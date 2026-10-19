"""Actual codec/MPD/buffer integration using small encoded test fixtures."""
from dataclasses import replace
import asyncio
import json
from pathlib import Path
import threading

import pytest

from src.client.adaptation_inputs import QualityTarget
from src.client.decoder import ProgressiveDecoder
from src.client.mpd_parser import MPDParser
from src.client.playback import run_playback
from src.client.playback_config import load_config
from src.client.playback_source import PlaybackSource, SegmentBuffer
from src.client.request_api import NetworkClient
from src.content_metadata import digest
from src.content_mpd import build_mpd
from src.server.media_server import create_server
from tools.content_preparation.assets import synthetic_state, state_hash
from tools.content_preparation.codec_adapter import CodecAdapter
from tools.content_preparation.manifest import build_manifest
from tools.content_preparation.mpd import load_prepared
from tools.content_preparation.packaging import package_object


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


@pytest.fixture
def encoded_bundle(tmp_path,request):
    root = tmp_path / "actor"
    root.mkdir()
    parameters = getattr(request,"param",{})
    fps,count = parameters.get("fps",2),parameters.get("count",3)
    frames = [{"frame": 10 + i, "timestamp": i / fps} for i in range(count)]
    qualities = [f"Q{q}" for q in range(4)]
    records, references = [], {}
    codec = CodecAdapter()
    for frame in frames:
        parent = None
        for q in range(4):
            state = synthetic_state(frame["frame"], 4 + q * 3)
            state.metadata["quality"] = f"Q{q}"
            path = root / f"{frame['frame']}-Q{q}.cpgs"
            record = codec.encode(state, path, parent=parent)
            parent = codec.decode(path, parent=parent)
            references[frame["frame"], q] = parent
            records.append({**record, "id": f"progressive:{'Base' if q == 0 else f'E{q}'}:{frame['frame']}",
                            "mode": "progressive", "quality": f"Q{q}", "frame": frame["frame"],
                            "path": path.name})
    package = package_object("actor", frames, qualities, records,
                             {"gof_frames": 3, "segment_frames": 3}, root)
    manifest = build_manifest("actor", frames, qualities, package, output_dir=root)
    manifest["object"].update(fps=fps, duration_seconds=count/fps)
    write_json(root / "manifest.json", manifest)
    write_json(root / "package_index.json", package)
    raw = (root / "manifest.json").read_bytes()
    write_json(tmp_path / "manifest.json", {"schema_version": "content-preparation.server.v1", "objects": [
        {"id": "actor", "manifest_path": "actor/manifest.json", "bytes": len(raw), "sha256": digest(raw)}]})
    loaded, _ = load_prepared(tmp_path, metadata_only=True)
    mpd, index = build_mpd(loaded)
    (tmp_path / "manifest.mpd").write_bytes(mpd)
    index["mpd"] = {"path": "manifest.mpd", "bytes": len(mpd), "sha256": digest(mpd)}
    write_json(tmp_path / "content_index.json", index)
    return tmp_path, references


def make_source(root, **kwargs):
    parser = MPDParser.from_local(root / "manifest.mpd", media_base_url="http://offline.invalid/")
    return PlaybackSource(parser, ProgressiveDecoder(), local_root=root, **kwargs)


@pytest.mark.asyncio
async def test_actual_mpd_decode_progression_and_zero_cost_cache_switch(encoded_bundle):
    root, references = encoded_bundle
    source = make_source(root)
    try:
        for q in range(4):
            result = await source.get_frame("actor", 10, q)
            assert state_hash(result.state) == state_hash(references[10, q])
        assert source.decoder.decode_count == 4 and source.fetch_count == 4
        counts = source.decoder.decode_count, source.fetch_count
        for q in (0, 3, 1, 2):
            assert (await source.get_frame("actor", 10, q)).state is source.decoder.get_frame("actor", 10, q).state
        assert (source.decoder.decode_count, source.fetch_count) == counts
        remaining = source.resolver.required_units(QualityTarget("actor", (11,), 3), source.availability())
        assert remaining.valid and remaining.required_bytes == 0  # shared verified containers
        for frame in (11, 12):
            result = await source.get_frame("actor", frame, 3)
            assert state_hash(result.state) == state_hash(references[frame, 3])
        assert source.fetch_count == 4
        assert all(state.frame != 10 for state in source.decoder.decoded_states.values())
    finally:
        source.close()
    assert source.buffer.size_bytes == 0 and not source.decoder.decoded_states


def test_media_clock_frame_boundaries_and_infinite_loop(encoded_bundle):
    source = make_source(encoded_bundle[0])
    assert source.select_frame("actor", 0).frame == 10
    assert source.select_frame("actor", .499999).frame == 10
    assert source.select_frame("actor", .5).frame == 11
    wrapped = source.select_frame("actor", 1.5)
    assert (wrapped.frame, wrapped.cycle, wrapped.media_time_s, wrapped.presentation_time_s) == (10, 1, 0, 1.5)
    assert source.select_frame("actor", 15000.5).frame == 11
    with pytest.raises(EOFError):
        source.select_frame("actor", 1.5, loop=False)
    for time in (-1, float("nan"), True):
        with pytest.raises(ValueError):
            source.select_frame("actor", time)
    source.close()


def test_decimal_sequence_duration_wraps_at_exact_media_boundary(encoded_bundle):
    source = make_source(encoded_bundle[0])
    frames = ({"frame": 10, "timestamp": 0.}, {"frame": 11, "timestamp": .1})
    source._timelines["actor"] = (frames, (0., .1), .2)
    for time, cycle in ((.2, 1), (.6, 3), (1., 5), (1000., 5000)):
        selection = source.select_frame("actor", time)
        assert (selection.frame, selection.cycle) == (10, cycle)
        assert selection.presentation_time_s == pytest.approx(time)
    assert source.select_frame("actor", .6 - 1e-6).frame == 11
    source.close()


@pytest.mark.asyncio
async def test_missing_incomplete_corrupt_segments_never_publish_state(encoded_bundle):
    source = make_source(encoded_bundle[0])
    unit = next(iter(source.index.request_units.values()))
    path = encoded_bundle[0] / unit.request.url.removeprefix("http://offline.invalid/")
    raw = path.read_bytes()
    path.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="Incomplete"):
        await source.get_frame("actor", 10, 0)
    path.write_bytes(raw[:-1] + bytes((raw[-1] ^ 1,)))
    with pytest.raises(ValueError, match="hash mismatch"):
        await source.get_frame("actor", 10, 0)
    path.unlink()
    with pytest.raises(FileNotFoundError, match="transferring"):
        await source.get_frame("actor", 10, 0)
    assert not source.availability().decoded_states and not source.buffer.cached_units
    source.close()


def test_buffer_eviction_removes_planner_credit_and_rejects_bad_bytes(encoded_bundle):
    source = make_source(encoded_bundle[0])
    unit = next(iter(source.index.request_units.values()))
    body = (encoded_bundle[0] / unit.request.url.removeprefix("http://offline.invalid/")).read_bytes()
    buffer = SegmentBuffer(len(body))
    buffer.put(unit, body)
    assert buffer.cached_units == {unit.unit_id: unit.checksum}
    second = replace(unit, unit_id="another")
    buffer.put(second, body)
    assert buffer.cached_units == {"another": unit.checksum}
    assert buffer.get(unit.unit_id) is None
    with pytest.raises(ValueError, match="hash mismatch"):
        buffer.put(unit, b"invalid")
    buffer = SegmentBuffer(0)
    buffer.put(unit, body)
    assert not buffer.cached_units
    source.close()


@pytest.mark.asyncio
async def test_real_http_mpd_transport_decode_matches_offline(encoded_bundle):
    root, references = encoded_bundle
    server = create_server(root, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    source = None
    try:
        async with NetworkClient(http_version="h1") as client:
            url = f"http://127.0.0.1:{server.server_port}/manifest.mpd"
            parser = await MPDParser.from_http(url, client)
            source = PlaybackSource(parser, ProgressiveDecoder(), client=client)
            frames = await asyncio.gather(source.get_frame("actor", 10, 3),
                                          source.get_frame("actor", 11, 3))
            for frame in frames:
                assert state_hash(frame.state) == state_hash(references[frame.frame_index, 3])
            assert source.fetch_count == 4
    finally:
        if source:
            source.close()
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.asyncio
async def test_application_decode_only_duplicates_and_loops(encoded_bundle, tmp_path):
    root, _ = encoded_bundle
    config = load_config(["--mpd", str(root / "manifest.mpd"), "--frames", "5", "--fps", "2",
                          "--objects", "2", "--quality-cycle", "0,3,1", "--quality-interval", "1",
                          "--decode-only", "--output", str(tmp_path / "run")])
    summary = await run_playback(config)
    rows = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert summary["decoded_output_frames"] == 5 and summary["rendered_frames"] == 0
    assert rows[3]["objects"]["actor-0"]["frame"] == 10
    assert rows[3]["objects"]["actor-0"]["cycle"] == 1
    assert all(row["objects"]["actor-0"] == row["objects"]["actor-1"] for row in rows)
    assert [row["objects"]["actor-0"]["quality"] for row in rows] == [0, 3, 1, 0, 3]


@pytest.mark.parametrize("args", [["--frames", "-1"], ["--fps", "0"], ["--objects", "0"],
                                  ["--quality-cycle", "5"], ["--quality-interval", "0"]])
def test_invalid_application_config_is_rejected(args):
    with pytest.raises(SystemExit):
        load_config(args)


def test_yaml_decode_only_selection_is_preserved(tmp_path):
    path = tmp_path / "client.yaml"
    path.write_text("playback:\n  decode_only: true\n")
    assert load_config(["--config", str(path)])["playback"]["decode_only"] is True
    with pytest.raises(SystemExit):
        load_config(["--config", str(path), "--display", "interactive"])


def test_interactive_controls_keep_quality_and_playback_independent():
    from types import SimpleNamespace
    from src.client.viewer import InteractiveViewer
    viewer = InteractiveViewer.__new__(InteractiveViewer)
    viewer.object_ids, viewer.selected = ("left", "right"), 0
    viewer.camera_sample = {"azimuth": 0, "elevation": 0, "distance": 4}
    viewer.paused, viewer.qualities, viewer.toggle_objects = False, {}, []
    for key in ("4", "Tab", "2", "p", "Right", "Up", "plus", "w", "space"):
        viewer._key(SimpleNamespace(keysym=key))
    assert viewer.qualities == {"left": 3, "right": 1}
    assert viewer.toggle_objects == ["right"] and viewer.paused
    assert viewer.camera_sample["azimuth"] == 5 and viewer.camera_sample["elevation"] == 5
    assert viewer.camera_sample["distance"] < 4 and viewer.camera_sample["target"][1] > 0


@pytest.mark.asyncio
async def test_decode_ahead_quality_switch_cancellation_and_bounds(encoded_bundle):
    from src.client.decode_ahead import DecodeAhead
    source=make_source(encoded_bundle[0],decode_in_worker=True)
    ahead=DecodeAhead(source,2)
    try:
        await ahead.prefetch([("actor",10,3),("actor",11,0),("actor",12,0)])
        assert len(ahead.pending)==2
        first=await ahead.get_frame("actor",10,3)
        assert first.quality==3
        await ahead.prefetch([("actor",12,2)])
        last=await ahead.get_frame("actor",12,2)
        assert state_hash(last.state)==state_hash(encoded_bundle[1][12,2])
        await ahead.prefetch([("actor",10,3)])
    finally:
        await ahead.close()
        source.close()
    assert not ahead.pending


@pytest.mark.asyncio
async def test_application_decode_ahead_runs_and_cleans_up(encoded_bundle,tmp_path):
    config=load_config(["--mpd",str(encoded_bundle[0]/"manifest.mpd"),"--frames","5","--fps","2",
                        "--objects","2","--quality-cycle","0,3,1","--quality-interval","1",
                        "--decode-only","--decode-ahead","--decoder-mode","native-cached",
                        "--output",str(tmp_path/"ahead")])
    result=await run_playback(config)
    assert result["decoded_output_frames"]==5


@pytest.mark.asyncio
async def test_pooled_source_deduplicates_requests_and_preserves_seek_quality(encoded_bundle):
    from src.client.decode_workers import DecodeWorkers
    root,references = encoded_bundle
    parser = MPDParser.from_local(root/"manifest.mpd",media_base_url="http://offline.invalid/")
    pool = DecodeWorkers(2,optimization=None)
    source = PlaybackSource(parser,ProgressiveDecoder(),local_root=root,workers=pool)
    try:
        frames = await asyncio.gather(source.get_frame("actor",10,3),source.get_frame("actor",10,3),source.get_frame("actor",12,2))
        assert frames[0] is frames[1]
        assert pool.stats["submitted"] == 2
        for frame in frames:
            assert state_hash(frame.state)==state_hash(references[frame.frame_index,frame.quality])
        seek = await source.get_frame("actor",11,1)
        assert state_hash(seek.state)==state_hash(references[11,1])
    finally:
        await source.aclose()
        pool.close()


@pytest.mark.asyncio
async def test_main_multiworker_runtime_wait_and_hold_have_separate_video_counters(encoded_bundle,tmp_path):
    for policy in ("wait","hold"):
        config = load_config(["--mpd",str(encoded_bundle[0]/"manifest.mpd"),"--workers","2",
                              "--decoder-mode","native-cached","--verification","runtime",
                              "--late-policy",policy,"--fps","30","--objects","2","--frames","5",
                              "--decode-only","--output",str(tmp_path/policy)])
        result = await run_playback(config)
        assert result["workers"]==2 and result["presentation_ticks"]==5
        # Source is 2 FPS: rendering the same media sample at 30 Hz is not new video.
        assert result["decoded_output_frames"]==1
        assert result["new_object_video_frames"]==2
        assert result["repeated_object_presentations"]==8


@pytest.mark.asyncio
async def test_responsive_wait_polls_camera_while_decoder_is_delayed():
    from src.client.playback import _responsive_wait
    class Viewer:
        polls = 0
        def poll(self):
            self.polls += 1
            return True
    viewer = Viewer()
    await _responsive_wait(asyncio.sleep(.06),viewer)
    assert viewer.polls >= 3


def test_verified_segment_member_proof_checks_ownership_bounds_and_mutable_input(encoded_bundle):
    from src.client.verified_input import VerifiedSegment
    root,_ = encoded_bundle
    source = make_source(root)
    descriptor = source.index.payload_for("actor",10,0)
    unit = source.index.request_units[descriptor.request_unit_id]
    location = source.parser.get_payload_location(descriptor.payload_id)
    data = bytearray((root/unit.request.url.removeprefix("http://offline.invalid/")).read_bytes())
    verified = VerifiedSegment(unit,data)
    data[-1] ^= 1
    token = verified.member(location)
    assert token.data.readonly and hashlib_sha(token.data)==location.checksum
    with pytest.raises(ValueError,match="size/hash"):
        VerifiedSegment(unit,data)
    with pytest.raises(ValueError,match="another segment"):
        verified.member(replace(location,request_unit_id="wrong"))
    with pytest.raises(ValueError,match="extent"):
        verified.member(replace(location,offset=len(verified.data)))
    source.close()


def hashlib_sha(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


@pytest.mark.asyncio
async def test_obsolete_work_drains_without_blocking_camera_and_errors_propagate():
    from src.client.decode_ahead import DecodeAhead
    class Source:
        async def get_frame(self, obj, frame, q):
            if frame==3:
                raise ValueError("corrupt incoming packet")
            try:
                await asyncio.sleep(.2)
            except asyncio.CancelledError:
                await asyncio.sleep(.05)  # Running native stage needs drainage.
                raise
    ahead = DecodeAhead(Source(),2)
    try:
        await ahead.prefetch([("actor",1,0)])
        await asyncio.sleep(.005)
        began = asyncio.get_running_loop().time()
        await ahead.prefetch([("actor",3,0)])
        assert asyncio.get_running_loop().time()-began < .03
        await asyncio.sleep(.005)
        with pytest.raises(ValueError,match="corrupt"):
            ahead.collect()
    finally:
        await ahead.close()


@pytest.mark.asyncio
async def test_cancelled_duplicate_waiter_does_not_cancel_other_consumer(encoded_bundle,monkeypatch):
    from src.client.decode_workers import DecodeWorkers
    root,_=encoded_bundle
    pool=DecodeWorkers(1,optimization=None)
    source=PlaybackSource(MPDParser.from_local(root/"manifest.mpd",media_base_url="http://offline.invalid/"),
                          ProgressiveDecoder(),local_root=root,workers=pool)
    original=source._pooled_frame
    async def delayed(key):
        await asyncio.sleep(.05)
        return await original(key)
    monkeypatch.setattr(source,"_pooled_frame",delayed)
    try:
        first=asyncio.create_task(source.get_frame("actor",10,0))
        second=asyncio.create_task(source.get_frame("actor",10,0))
        await asyncio.sleep(.005)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):await first
        assert (await second).frame_index==10
        assert pool.stats["submitted"]==1
    finally:
        await source.aclose()
        pool.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('encoded_bundle',[{'fps':30,'count':8}],indirect=True)
async def test_hold_presents_late_completions_without_starvation(encoded_bundle,tmp_path,monkeypatch):
    import time
    from src.client.decode_workers import DecodeWorkers
    original=DecodeWorkers._decode
    def delayed(self,*args,**kwargs):
        time.sleep(.09)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(DecodeWorkers,'_decode',delayed)
    config=load_config(['--mpd',str(encoded_bundle[0]/'manifest.mpd'),'--workers','1',
                        '--decoder-mode','baseline','--late-policy','hold','--frames','8',
                        '--ahead-capacity','3',
                        '--fps','30','--objects','1','--no-loop','--decode-only',
                        '--output',str(tmp_path/'late')])
    result=await run_playback(config)
    assert 2 <= result['decoded_output_frames'] < 8
    assert result['repeated_object_presentations'] > 0
    assert result['scheduler_statistics']['peak_pending'] <= 3


@pytest.mark.asyncio
@pytest.mark.parametrize('encoded_bundle',[{'fps':30,'count':8}],indirect=True)
async def test_finite_output_does_not_decode_past_requested_end(encoded_bundle,tmp_path):
    config=load_config(['--mpd',str(encoded_bundle[0]/'manifest.mpd'),'--workers','2',
                        '--decoder-mode','baseline','--frames','2','--fps','30','--objects','1',
                        '--decode-only','--output',str(tmp_path/'finite')])
    result=await run_playback(config)
    assert result['decode_count']==2


@pytest.mark.asyncio
async def test_paused_object_does_not_suppress_active_scene_video_count(encoded_bundle,tmp_path):
    config=load_config(['--mpd',str(encoded_bundle[0]/'manifest.mpd'),'--workers','2','--decoder-mode','baseline',
                        '--frames','3','--fps','2','--decode-only','--output',str(tmp_path/'paused')])
    config['scene']['objects']=[{'id':'moving','source':'actor','quality':0},
                               {'id':'paused','source':'actor','quality':0,'playing':False}]
    result=await run_playback(config)
    assert result['decoded_output_frames']==3
    assert result['per_object']['paused']['new_video_frames']==1
    assert result['per_object']['moving']['new_video_frames']==3


@pytest.mark.asyncio
async def test_waiting_for_fetch_does_not_occupy_native_worker(encoded_bundle,monkeypatch):
    from src.client.decode_workers import DecodeWorkers
    root,_=encoded_bundle
    pool=DecodeWorkers(1,optimization=None)
    source=PlaybackSource(MPDParser.from_local(root/'manifest.mpd',media_base_url='http://offline.invalid/'),
                          ProgressiveDecoder(),local_root=root,workers=pool)
    release=asyncio.Event()
    original=source.prepare_chain
    async def prepare(obj,frame,q):
        if frame==10:
            await release.wait()
        return await original(obj,frame,q)
    monkeypatch.setattr(source,'prepare_chain',prepare)
    first=asyncio.create_task(source.get_frame('actor',10,0))
    try:
        await asyncio.sleep(.005)
        ready=await asyncio.wait_for(source.get_frame('actor',11,0),.5)
        assert ready.frame_index==11 and not first.done()
        release.set()
        assert (await first).frame_index==10
    finally:
        release.set()
        await source.aclose()
        pool.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('encoded_bundle',[{'fps':30,'count':20}],indirect=True)
async def test_startup_primes_actual_media_before_clock_and_preserves_order(encoded_bundle,tmp_path):
    config=load_config(['--mpd',str(encoded_bundle[0]/'manifest.mpd'),'--objects','2',
                        '--decoder-mode','baseline','--workers','2','--frames','20','--fps','30',
                        '--startup-buffer-s','.5','--target-buffer-s','.6','--decode-only',
                        '--output',str(tmp_path/'primed')])
    result=await run_playback(config)
    rows=[json.loads(row) for row in (tmp_path/'primed/metrics.jsonl').read_text().splitlines()]
    assert result['startup_buffer']['status']=='ready'
    assert result['startup_buffer']['decoded_duration_s']==pytest.approx(.5)
    assert result['startup_buffer']['required_frames']==15  # Shared source/frame/quality.
    assert result['first_frame_startup_s'] >= result['startup_buffer']['wait_s']
    assert result['decoded_output_frames']==20
    assert result['dropped_object_video_frames']==0
    assert [row['objects']['actor-0']['frame'] for row in rows]==list(range(10,30))
    assert rows[0]['presentation_time_s']==0.
    assert rows[0]['buffered_duration_s']>=.5


@pytest.mark.asyncio
@pytest.mark.parametrize('encoded_bundle',[{'fps':30,'count':12}],indirect=True)
async def test_startup_capacity_fallback_is_explicit_and_no_prefetch_still_advances(encoded_bundle,tmp_path):
    args=['--mpd',str(encoded_bundle[0]/'manifest.mpd'),'--objects','1','--workers','2',
          '--decoder-mode','baseline','--frames','12','--fps','30','--decode-only',
          '--startup-buffer-s','.5','--ahead-capacity','3','--no-prefetch',
          '--output',str(tmp_path/'small')]
    result=await run_playback(load_config(args))
    assert result['startup_buffer']['status']=='capacity_limited'
    assert result['startup_buffer']['decoded_duration_s']==pytest.approx(.1)
    assert result['decoded_output_frames']==12
    assert result['decode_count']==12
    assert not result['prefetch_enabled']
    with pytest.raises(ValueError,match='capacity limit'):
        await run_playback(load_config(args+['--startup-fallback','error']))


@pytest.mark.asyncio
@pytest.mark.parametrize('encoded_bundle',[{'fps':30,'count':20}],indirect=True)
async def test_optional_rebuffer_freezes_media_clock_and_renders_repeats(encoded_bundle,tmp_path,monkeypatch):
    import time
    from src.client.decode_workers import DecodeWorkers
    original=DecodeWorkers._decode
    def delayed(self,*args,**kwargs):
        time.sleep(.065)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(DecodeWorkers,'_decode',delayed)
    config=load_config(['--mpd',str(encoded_bundle[0]/'manifest.mpd'),'--objects','1','--workers','1',
                        '--decoder-mode','baseline','--frames','18','--fps','30','--decode-only',
                        '--late-policy','hold','--ahead-capacity','4','--rebuffer-after-s','.025',
                        '--rebuffer-buffer-s','.1','--output',str(tmp_path/'rebuffer')])
    result=await run_playback(config)
    rows=[json.loads(row) for row in (tmp_path/'rebuffer/metrics.jsonl').read_text().splitlines()]
    assert result['rebuffer_events']>=1 and result['rebuffer_seconds']>0
    assert any(row['rebuffering'] for row in rows)
    assert result['repeated_object_presentations']>0
    assert result['scheduler_statistics']['peak_pending']<=4
    assert all(a['presentation_time_s']<=b['presentation_time_s'] for a,b in zip(rows,rows[1:]))


@pytest.mark.parametrize('args',[['--startup-buffer-s','-1'],['--target-buffer-s','nan'],
                                 ['--startup-timeout-s','0'],['--rebuffer-after-s','inf'],
                                 ['--rebuffer-buffer-s','0']])
def test_invalid_startup_options_are_rejected(args):
    with pytest.raises(SystemExit):load_config(args)


@pytest.mark.asyncio
@pytest.mark.parametrize('encoded_bundle',[{'fps':30,'count':12}],indirect=True)
async def test_rebuffer_error_policy_rejects_insufficient_capacity(encoded_bundle,tmp_path,monkeypatch):
    import time
    from src.client.decode_workers import DecodeWorkers
    original=DecodeWorkers._decode
    def delayed(self,*args,**kwargs):
        time.sleep(.065)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(DecodeWorkers,'_decode',delayed)
    config=load_config(['--mpd',str(encoded_bundle[0]/'manifest.mpd'),'--objects','1','--workers','1',
                        '--decoder-mode','baseline','--frames','10','--fps','30','--decode-only',
                        '--late-policy','hold','--ahead-capacity','1','--rebuffer-after-s','.001',
                        '--rebuffer-buffer-s','.1','--startup-fallback','error','--output',str(tmp_path/'rebuffer_error')])
    with pytest.raises(ValueError,match='rebuffer.*capacity limit'):
        await run_playback(config)


@pytest.mark.asyncio
@pytest.mark.parametrize('encoded_bundle',[{'fps':30,'count':150}],indirect=True)
async def test_hold_clock_preserves_nominal_frame_boundary_after_147_steps(encoded_bundle,tmp_path):
    config=load_config(['--mpd',str(encoded_bundle[0]/'manifest.mpd'),'--objects','1','--workers','2',
                        '--decoder-mode','baseline','--frames','150','--fps','30','--decode-only',
                        '--late-policy','hold','--startup-buffer-s','.5','--target-buffer-s','1',
                        '--output',str(tmp_path/'integerclock')])
    result=await run_playback(config)
    rows=[json.loads(row) for row in (tmp_path/'integerclock/metrics.jsonl').read_text().splitlines()]
    assert len(rows)==150
    for row in rows:
        expected=10+(round(row['presentation_time_s']*30)%150)
        assert row['objects']['actor-0']['desired_frame']==expected


@pytest.mark.asyncio
@pytest.mark.parametrize('encoded_bundle',[{'fps':60,'count':20}],indirect=True)
async def test_wait_clock_preserves_arbitrary_start_offset(encoded_bundle,tmp_path):
    config=load_config(['--mpd',str(encoded_bundle[0]/'manifest.mpd'),'--workers','2',
                        '--decoder-mode','baseline','--frames','5','--fps','30','--decode-only',
                        '--output',str(tmp_path/'offsetclock')])
    config['scene']['objects']=[{'id':'offset','source':'actor','quality':0,'start_time_s':.02}]
    await run_playback(config)
    rows=[json.loads(row) for row in (tmp_path/'offsetclock/metrics.jsonl').read_text().splitlines()]
    assert [row['objects']['offset']['frame'] for row in rows]==[11,13,15,17,19]
