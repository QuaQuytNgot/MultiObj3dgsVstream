"""Playback application orchestration, kept outside the small CLI entry point."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import asdict
import json
from collections import deque
from pathlib import Path
import time

import numpy as np

from .bandwidth import BandwidthEstimator, TransferObserver
from .decoder import ProgressiveDecoder
from .mpd_parser import MPDParser
from .playback_source import PlaybackSource
from .request_api import NetworkClient
from .scene import SceneManager
from .playback_buffer import plan_window, prime_window, presentation_steps, MAX_PLANNING_STEPS
from src.renderer.backend import Camera, create_renderer


def _objects(config, parser):
    options = config["playback"]
    source = parser.get_objects()[0]
    if "objects" in options:
        result = []
        count = options["objects"]
        for i in range(count):
            matrix = np.eye(4)
            matrix[0, 3] = (i - (count - 1) / 2) * 1.4
            result.append({"id": f"{source}-{i}", "source": source, "transform": matrix,
                           "quality": options.get("quality", 0)})
        return result
    result = config["scene"].get("objects") or [{"id": obj, "source": obj, "quality": 0}
                                               for obj in parser.get_objects()]
    return [{**obj, "quality": options.get("quality", obj.get("quality", 0))} for obj in result]


async def _responsive_wait(awaitable, viewer):
    task = asyncio.ensure_future(awaitable)
    try:
        while not task.done():
            if viewer and not viewer.poll():
                raise KeyboardInterrupt
            await asyncio.wait({task}, timeout=.01)
        return task.result()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def run_playback(config):
    options, rendering = config["playback"], config["rendering"]
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    scene, source, renderer, viewer, ahead, pool = SceneManager(), None, None, None, None, None
    estimator = BandwidthEstimator(**{key: value for key, value in config.get("bandwidth", {}).items()
                                      if key in ("window_size", "max_sample_age_s", "startup_bandwidth_bps")})
    observer = None
    rendered = frame_count = new_scene_frames = new_object_frames = repeats = misses = dropped = 0
    render_seconds = 0.
    rebuffer_events = rebuffer_seconds = 0
    rebuffer_started = underrun_since = None
    rebuffer_history=deque(maxlen=64)
    startup = {"requested_s":options.get("startup_buffer_s",0.),"status":"disabled","wait_s":0.}
    started, cpu_started = time.perf_counter(), time.process_time()
    async with AsyncExitStack() as stack:
        try:
            if options["mode"] == "streaming":
                client = await stack.enter_async_context(NetworkClient(**config.get("network", {})))
                if config.get("network", {}).get("concurrency", 1) == 1:
                    observer = TransferObserver(estimator)
                    stack.callback(client.subscribe(observer))
                parser = await MPDParser.from_http(options["mpd"], client)
                source_options = {"client": client}
            else:
                mpd = Path(options["mpd"])
                parser = MPDParser.from_local(mpd, media_base_url="http://offline.invalid/")
                source_options = {"local_root": mpd.parent}
            objects = _objects(config, parser)
            if not objects:
                raise ValueError("Playback requires at least one scene object")
            optimization = config.get("decoder_optimization")
            decoder = ProgressiveDecoder(config.get("decoder"), max_cached_frames=max(2, 2*len(objects)),
                                         optimization=optimization)
            policy = options.get("late_policy", "wait")
            selection_policy = None
            if options.get("decode_ahead", False):
                from .decode_workers import DecodeWorkers, select_workers, queue_capacity
                qualities = options.get("quality_cycle", []) or [obj["quality"] for obj in objects]
                if options["display"] == "interactive":
                    qualities = [3]  # Interactive controls may request any quality.
                shared_mib = sum((optimization or {}).get(k,128*1024**2) for k in
                                 ("stream_cache_bytes","array_cache_bytes")) // 1024**2
                worker_count, selection_policy = select_workers(
                    options.get("workers",1), max(qualities), max_workers=options.get("max_workers",6),
                    memory_mib=options.get("decode_memory_mib",2048), shared_mib=shared_mib,
                    profile=options.get("worker_profile"), verification=(optimization or {}).get("verification","strict"),
                    native=(optimization or {}).get("native","baseline"))
                pool = DecodeWorkers(worker_count, pending=worker_count, optimization=optimization,
                                     runtime_config=config.get("decoder"))
                window_steps = max(min(MAX_PLANNING_STEPS,options.get("ahead_frames",12)+1),
                                   presentation_steps(max(options.get("startup_buffer_s",0.),options.get("target_buffer_s",0.),
                                                          options.get("rebuffer_buffer_s",.5) if options.get("rebuffer_after_s",0.) else 0.),options["fps"])[0])
                capacity = queue_capacity(max(qualities),worker_count,selection_policy["admission_memory_mib"],
                                          shared_mib,window_steps*len(objects),
                                          options.get("ahead_capacity"))
                selection_policy["ahead_capacity"] = capacity
            else:
                capacity = max(2,len(objects))
            source = PlaybackSource(parser, decoder, **source_options,
                                    buffer_bytes=options.get("buffer_bytes",512*1024**2),
                                    stream_package=options.get("stream_package"),
                                    stream_base_url=options.get("stream_base_url"), workers=pool,
                                    result_capacity=capacity)
            if pool:
                from .decode_ahead import DecodeAhead
                ahead = DecodeAhead(source,capacity)
            for obj in objects:
                parser.get_representation(obj["source"],obj["quality"])
                scene.add_object(obj["id"],obj.get("transform"),playing=obj.get("playing",True))
            camera_sample = dict(config["scene"]["camera"])
            camera_settings = {key:value for key,value in rendering.items()
                               if key in ("width","height","up_axis","fov_degrees","near","far")}
            if not options["decode_only"]:
                renderer = create_renderer(rendering["backend"],**rendering.get("settings",{}))
            if options["display"] == "interactive":
                from .viewer import InteractiveViewer
                viewer = InteractiveViewer(camera_sample,scene.objects,up_axis=camera_settings.get("up_axis","y"))
            clocks = {obj["id"]:float(obj.get("start_time_s",0.)) for obj in objects}
            clock_origins=dict(clocks)
            clock_steps={identifier:0 for identifier in clocks}
            cycle = options.get("quality_cycle",[])
            last_states, last_identities, state_cycles = {}, {}, {}
            first_ready = anchor = None
            previous_tick = 0
            skipped_slots = 0
            per_object = {obj["id"]:{"new_video_frames":0,"deadline_misses":0,"repeated_presentations":0} for obj in objects}
            positions={name:{sample["frame"]:i for i,sample in enumerate(timeline[0])}
                       for name,timeline in source._timelines.items()}
            last_ordinals={}
            def quality_for(obj,step):
                q = cycle[(step//options["quality_interval"])%len(cycle)] if cycle else obj["quality"]
                return viewer.qualities.get(obj["id"],q) if viewer else q
            def advancing(obj):
                return scene.objects[obj["id"]].playing and (not viewer or not viewer.paused)
            def advance_clocks(steps):
                # Repeated float addition drifts enough to select the previous
                # media frame near 4.9 s at 30 Hz. Preserve arbitrary offsets
                # and pause state with an integer step count per object.
                for obj in objects:
                    identifier=obj["id"]
                    if advancing(obj):
                        clock_steps[identifier]+=steps
                        clocks[identifier]=clock_origins[identifier]+clock_steps[identifier]/options["fps"]
            target_s=max(min(MAX_PLANNING_STEPS,options.get("ahead_frames",12)+1)/options["fps"],
                         options.get("startup_buffer_s",0.),options.get("target_buffer_s",0.))
            def window_for(duration,tick):
                peers=objects[frame_count%len(objects):]+objects[:frame_count%len(objects)]
                return plan_window(source,peers,clocks,lambda obj,offset:quality_for(obj,tick+offset),advancing,
                                   fps=options["fps"],duration_s=duration,capacity=capacity,loop=options["loop"],
                                   max_steps=(options["frames"]-frame_count) if options["frames"] else None)
            if options.get("startup_buffer_s",0.) > 0:
                initial=window_for(options["startup_buffer_s"],0)
                if initial.reason in ("decoded queue memory/capacity limit","presentation planning limit") and options.get("startup_fallback","shrink") == "error":
                    raise ValueError("Requested startup decoded buffer exceeds memory/capacity limit")
                startup=await _responsive_wait(prime_window(ahead,initial,timeout_s=options.get("startup_timeout_s",30.),
                                                            fallback=options.get("startup_fallback","shrink")),viewer)
            startup["target_s"]=target_s
            with (output/"metrics.jsonl").open("w") as metrics:
                while options["frames"] == 0 or frame_count < options["frames"]:
                    frame_started = time.perf_counter()
                    if viewer:
                        if not viewer.poll(): break
                        for identifier in viewer.toggle_objects:
                            scene.set_playing(identifier,not scene.objects[identifier].playing)
                        viewer.toggle_objects.clear()
                    tick = frame_count
                    if policy == "hold" and anchor is not None:
                        threshold=options.get("rebuffer_after_s",0.)
                        if rebuffer_started is None and threshold and underrun_since is not None and frame_started-underrun_since >= threshold:
                            rebuffer_started=frame_started
                            rebuffer_events+=1
                            rebuffer_history.append({"requested_s":options.get("rebuffer_buffer_s",.5),"status":"filling"})
                        tick = previous_tick if rebuffer_started is not None else max(previous_tick+1,int((frame_started-anchor)*options["fps"]))
                        skipped_slots += max(0,tick-previous_tick-1)
                        advance_clocks(tick-previous_tick)
                    playback_time = tick/options["fps"]
                    current, requests = {}, []
                    try:
                        for obj in objects:
                            selection = source.select_frame(obj["source"],clocks[obj["id"]],loop=options["loop"])
                            current[obj["id"]] = (selection,quality_for(obj,tick))
                        if ahead:
                            ahead.collect()
                            # Consume late completions before retiring past
                            # lookahead slots; otherwise a slow decoder starves
                            # presentation despite producing usable states.
                            if policy == "hold" and first_ready is not None:
                                for obj in objects:
                                    selected,q = current[obj["id"]]
                                    ready = ahead.latest(obj["source"],selected.media_time_s,q)
                                    prior = last_states.get(obj["id"])
                                    wrapped = state_cycles.get(obj["id"]) != selected.cycle
                                    if ready is not None and (prior is None or wrapped or ready.timestamp >= prior.timestamp):
                                        last_states[obj["id"]] = ready
                                        state_cycles[obj["id"]] = selected.cycle
                            # Earliest presentation slot first; rotate peers to
                            # avoid always giving the same object the first worker.
                            window=window_for(max(target_s,options.get("rebuffer_buffer_s",.5)) if rebuffer_started is not None else target_s,tick)
                            # Rotate equal-deadline object peers before admission.
                            requests=list(window.requests)
                            peers=objects[frame_count%len(objects):]+objects[:frame_count%len(objects)]
                            current_keys=[(obj["source"],current[obj["id"]][0].frame,current[obj["id"]][1]) for obj in peers]
                            requests=list(dict.fromkeys(current_keys+requests))
                            admit=None if options.get("prefetch",True) or rebuffer_started is not None else set(current_keys)
                            await ahead.prefetch(requests,retain_running=policy=="hold",admit=admit)
                            if rebuffer_started is not None:
                                recovery=window_for(options.get("rebuffer_buffer_s",.5),tick)
                                if recovery.reason in ("decoded queue memory/capacity limit","presentation planning limit") and options.get("startup_fallback","shrink") == "error":
                                    raise ValueError("Requested rebuffer decoded window exceeds memory/capacity limit")
                                ready=recovery.availability(ahead)
                                rebuffer_history[-1].update(effective_target_s=recovery.effective_s,reason=recovery.reason,
                                                            decoded_duration_s=ready["decoded_duration_s"])
                                timed_out=time.perf_counter()-rebuffer_started >= options.get("startup_timeout_s",30.)
                                if ready["ready"] or timed_out:
                                    if timed_out and options.get("startup_fallback","shrink") == "error":
                                        raise RuntimeError("Rebuffer decoded window timed out")
                                    elapsed_rebuffer=time.perf_counter()-rebuffer_started
                                    anchor+=elapsed_rebuffer
                                    rebuffer_seconds+=elapsed_rebuffer
                                    rebuffer_history[-1].update(status="timeout" if timed_out else "ready",wait_s=elapsed_rebuffer)
                                    rebuffer_started=underrun_since=None
                    except EOFError:
                        break
                    if policy == "wait" or first_ready is None:
                        tasks = [(ahead or source).get_frame(obj["source"],current[obj["id"]][0].frame,current[obj["id"]][1]) for obj in objects]
                        decoded_frames = await _responsive_wait(asyncio.gather(*tasks),viewer)
                        last_states.update({obj["id"]:frame for obj,frame in zip(objects,decoded_frames)})
                        state_cycles.update({obj["id"]:current[obj["id"]][0].cycle for obj in objects})
                        if first_ready is None:
                            anchor = time.perf_counter()  # Final clock anchor follows first state preparation/render.
                    elif ahead:
                        ahead.collect()
                        for obj in objects:
                            selected,q = current[obj["id"]]
                            frame = ahead.latest(obj["source"],selected.media_time_s,q)
                            prior = last_states.get(obj["id"])
                            wrapped = state_cycles.get(obj["id"]) != selected.cycle
                            if frame is not None and (prior is None or wrapped or frame.timestamp >= prior.timestamp):
                                last_states[obj["id"]] = frame
                                state_cycles[obj["id"]] = selected.cycle
                    selections, all_new, any_new = {}, True, False
                    tick_missed=False
                    lateness = max(0.,time.perf_counter()-anchor-playback_time)
                    for obj in objects:
                        identifier = obj["id"]
                        desired,q = current[identifier]
                        decoded = last_states[identifier]
                        actual_cycle = state_cycles[identifier]
                        identity = (actual_cycle,decoded.frame_index)
                        fresh = last_identities.get(identifier) != identity
                        ordinal=actual_cycle*len(positions[obj["source"]])+positions[obj["source"]][decoded.frame_index]
                        skipped_media=max(0,ordinal-last_ordinals[identifier]-1) if fresh and identifier in last_ordinals else 0
                        last_ordinals[identifier]=ordinal
                        dropped+=skipped_media
                        per_object[identifier]["dropped_video_frames"]=per_object[identifier].get("dropped_video_frames",0)+skipped_media
                        exact = decoded.frame_index == desired.frame and decoded.quality == q and actual_cycle == desired.cycle
                        missed = not exact or (policy == "wait" and lateness > 1/options["fps"])
                        per_object[identifier]["new_video_frames"] += int(fresh)
                        per_object[identifier]["repeated_presentations"] += int(not fresh)
                        per_object[identifier]["deadline_misses"] += int(missed)
                        new_object_frames += int(fresh)
                        repeats += int(not fresh)
                        misses += int(missed)
                        tick_missed = tick_missed or (missed and advancing(obj))
                        active = scene.objects[identifier].playing and (not viewer or not viewer.paused)
                        if active:
                            all_new = all_new and fresh
                        any_new = any_new or fresh
                        last_identities[identifier] = identity
                        scene.update_object(identifier,decoded.state,playback_time,quality=decoded.quality,frame_index=decoded.frame_index)
                        selections[identifier] = {"frame":decoded.frame_index,"quality":decoded.quality,
                            "media_time_s":decoded.timestamp,"cycle":actual_cycle,"desired_cycle":desired.cycle,"gaussians":decoded.state.count,
                            "desired_frame":desired.frame,"desired_quality":q,"new_video_frame":fresh,
                            "dropped_video_frames":skipped_media,
                            "deadline_missed":missed,"delivery_lateness_s":lateness,
                            "media_lateness_s":max(0.,desired.media_time_s-decoded.timestamp+(desired.cycle-actual_cycle)*source._timelines[obj["source"]][2])}
                    new_scene_frames += int(all_new and any_new)
                    row = {"output_frame":frame_count,"presentation_time_s":playback_time,"objects":selections,
                           "decoded_bytes_retained":sum(d.retained_bytes for d in pool.decoders) if pool else decoder.retained_bytes,
                           "segment_bytes_retained":source.buffer.size_bytes,
                           "decode_queue":len(ahead.pending) if ahead else 0,"rebuffering":rebuffer_started is not None}
                    if ahead:
                        ahead.collect()
                        known={(obj["source"],last_states[obj["id"]].frame_index,last_states[obj["id"]].quality) for obj in objects}
                        availability=window.availability(ahead,known)
                        row["buffered_duration_s"]=availability["decoded_duration_s"]
                        row["decoded_buffer_per_object_s"]=availability["per_object"]
                        row["decoded_buffer_frames"]=len(ahead.ready)
                    if renderer:
                        image = scene.render(renderer,Camera.orbit(camera_sample,camera_settings),playback_time)
                        row.update(image.metrics)
                        row["render_seconds"] = image.metrics.get("render_ms",0.)/1000
                        row["upload_seconds"] = image.metrics.get("upload_ms",0.)/1000
                        row["gpu_readback_seconds"] = image.metrics.get("transfer_ms",0.)/1000
                        render_seconds += row["render_seconds"]
                        rendered += 1
                        if viewer:
                            viewer.show(image.rgb)
                        elif options.get("write_frames",True):
                            from PIL import Image
                            Image.fromarray(np.uint8(np.clip(image.rgb,0,1)*255)).save(output/f"frame_{frame_count:06d}.png")
                    if first_ready is None:
                        first_ready=time.perf_counter()
                        anchor=first_ready
                        startup["initial_decoded_duration_s"]=row.get("buffered_duration_s",0.)
                    if rebuffer_started is None:
                        underrun_since=(underrun_since or time.perf_counter()) if tick_missed else None
                    elapsed = time.perf_counter()-frame_started
                    row.update(frame_seconds=elapsed,fps=1/elapsed if elapsed else None)
                    metrics.write(json.dumps(row,allow_nan=False)+"\n")
                    metrics.flush()
                    frame_count += 1
                    previous_tick = tick
                    if policy == "wait" and (not viewer or not viewer.paused):
                        advance_clocks(1)
                    if policy == "hold":
                        deadline=frame_started+1/options["fps"] if rebuffer_started is not None else anchor+(tick+1)/options["fps"]
                        await asyncio.sleep(max(0.,deadline-time.perf_counter()))
                    elif viewer:
                        await asyncio.sleep(max(0.,1/options["fps"]-elapsed))
            presentation_ended = time.perf_counter()
            if rebuffer_started is not None:
                rebuffer_seconds+=presentation_ended-rebuffer_started
            if ahead:
                await ahead.close()
            await source.aclose()
            decoders = pool.decoders if pool else [decoder]
            wall = time.perf_counter()-started
            statistics = {}
            for item in decoders:
                if item.optimized:
                    for key,value in item.optimized.stats.items():
                        statistics[key] = statistics.get(key,0)+value
            summary = {"status":"passed","mode":options["mode"],"backend":None if renderer is None else rendering["backend"],
                "presentation_ticks":frame_count,"decoded_output_frames":new_scene_frames,
                "new_object_video_frames":new_object_frames,"repeated_object_presentations":repeats,
                "dropped_object_video_frames":dropped,
                "object_deadline_misses":misses,"per_object":per_object,"late_policy":policy,
                "skipped_presentation_slots":skipped_slots,
                "startup_buffer":startup,"prefetch_enabled":options.get("prefetch",True),
                "target_buffer_s":target_s,"rebuffer_events":rebuffer_events,"rebuffer_seconds":rebuffer_seconds,
                "rebuffer_history":list(rebuffer_history),"buffer_planning_step_limit":MAX_PLANNING_STEPS,
                "encoded_input_capacity":source.input_capacity,
                "rendered_frames":rendered,"decode_count":sum(d.decode_count for d in decoders),
                "decode_seconds":sum(d.total_decode_seconds for d in decoders),"render_seconds":render_seconds,
                "wall_seconds":wall,"cpu_seconds":time.process_time()-cpu_started,
                "fps":new_scene_frames/wall,"presentation_tick_fps":frame_count/max(1e-9,presentation_ended-started),
                "first_frame_startup_s":None if first_ready is None else first_ready-started,
                "workers":pool.workers if pool else 0,"worker_selection":selection_policy,
                "worker_statistics":dict(pool.stats) if pool else None,
                "completed_chain_fps":pool.stats["decoded_frame_chains"]/wall if pool else None,
                "scheduler_statistics":dict(ahead.stats) if ahead else None,
                "media_reads":source.stream_store.stats["stream_reads"] if source.stream_store else source.fetch_count,
                "media_bytes":source.stream_store.stats["read_bytes"] if source.stream_store else source.read_bytes,
                "segment_verification_task_seconds":source.segment_verification_seconds,
                "encoded_input_copy_seconds":source.input_copy_seconds,"encoded_input_copy_bytes":source.input_copy_bytes,
                "bandwidth":asdict(estimator.estimate_bandwidth()),
                "traffic":None if observer is None else asdict(observer.snapshot()),
                "decoder_optimization":{"statistics":statistics,"verification":(optimization or {}).get("verification","strict")},
                "stream_package_statistics":None if source.stream_store is None else dict(source.stream_store.stats)}
            active_s = max(1e-9,presentation_ended-first_ready) if first_ready is not None else None
            summary["scene_tick_fps_after_startup"] = (frame_count-1)/active_s if active_s else None
            summary["new_scene_fps_after_startup"] = max(0,new_scene_frames-1)/active_s if active_s else None
            summary["rendered_fps"] = rendered/wall
            summary["rendered_fps_after_startup"] = max(0,rendered-1)/active_s if active_s else None
            summary["decoder_cache_bytes"] = (pool.shared.streams.size+pool.shared.arrays.size+pool.shared.idmaps.size) if pool and pool.shared else 0
            try:
                import resource
                summary["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            except ImportError:
                summary["peak_rss_bytes"] = None
            (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
            return summary
        finally:
            if ahead: await ahead.close()
            if source: await source.aclose()
            if pool: await asyncio.to_thread(pool.close)
            if viewer: viewer.close()
            if renderer: renderer.close()
            scene.close()
