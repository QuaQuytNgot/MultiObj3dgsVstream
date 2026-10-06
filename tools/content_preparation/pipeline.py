"""Sequential offline preparation with per-frame and per-view recovery."""
from itertools import product
from pathlib import Path
from contextlib import nullcontext
import json
import os
import shutil
import sys
import tempfile
import numpy as np
from . import __version__
from .paths import UPSTREAM_ROOT
from .config import ROOT, STAGES, config_hash, frame_records, resolve_path
from .checkpoint import Journal, file_lock, sha256, write_json
from .upstream import build_training_command, run_command, source_snapshot, verify_checkpoint_lineage,runtime_provenance
# BEGIN PROGRESS DISPLAY ONLY
from tools.content_preparation_progress import implementation_hash, observe_stages
# END PROGRESS DISPLAY ONLY


def fixture_state(frame, quality_index, seed=0):
    """Tiny analytic moving scene, with shared-state and static refinements."""
    from .assets import GaussianState
    rng=np.random.default_rng(seed)
    n=8+quality_index*4
    xyz=rng.uniform(-.5,.5,(n,3)).astype("float32")
    xyz[:,0] += .05*np.sin(frame*.5)
    xyz[:8,1] += .015*quality_index
    rotation=np.zeros((n,4),dtype="float32"); rotation[:,0]=1
    color=rng.uniform(.2,.9,(n,3)).astype("float32")
    color[:8] += .01*quality_index
    sh=((color-.5)/.28209479177387814)[:,:,None]
    arrays={"xyz":xyz,"rotation":rotation,"scale":np.full((n,3),-2.2,dtype="float32"),
            "opacity":np.full((n,1),1.5+.1*quality_index,dtype="float32"),"sh":sh}
    return GaussianState(arrays, {"stable_ids":True,"lineage_verified":True,"synthetic":True,"sh_degree":0,"frame":frame},np.arange(n,dtype="int64"))


class Pipeline:
    def __init__(self, config, config_path, resume=False, overwrite=False):
        self.cfg=config
        self.config_path=Path(config_path).resolve()
        self.output=resolve_path(config["output"])
        self.resume,self.overwrite=resume,overwrite
        self.code_hash=config_hash({p.name:sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
        # BEGIN PROGRESS DISPLAY ONLY
        self.code_hash=implementation_hash(Path(__file__).parent)
        # END PROGRESS DISPLAY ONLY
        self.snapshot=source_snapshot()
        runtime=runtime_provenance(config)
        self.runtime_signature={k:v for k,v in runtime.items() if k not in {"gpu","cuda_runtime"}}

    def selected_stages(self, stage="all"):
        if stage == "proxy" and not self.cfg["proxy"]["enabled"]:
            raise ValueError("Proxy stage is disabled by proxy.enabled: false")
        return tuple(s for s in STAGES if s != "proxy" or self.cfg["proxy"]["enabled"]) if stage == "all" else (stage,)

    def plan(self, stage="all"):
        selected=self.selected_stages(stage)
        return {"dry_run":True,"output":str(self.output),"stages":list(selected),"delivery_modes":self.cfg["delivery_modes"],
                "objects":[{"id":o["id"],"frames":len(frame_records(o)),"qualities":[q["id"] for q in o["qualities"]],
                            "sampled_views_per_frame":len(self.cfg["sampling"]["azimuth"])*len(self.cfg["sampling"]["elevation"])*len(self.cfg["sampling"]["scales"]),
                            "dataset_kind":o["dataset"]["kind"]} for o in self.cfg["objects"]],
                "execution":"one object, quality, frame, view; LPIPS batch=1; CPU/disk reference cache",
                "training_backend":self.cfg["training"]["backend"],"renderer_backend":self.cfg["renderer"]["backend"]}

    def run(self, stage="all", dry_run=False):
        if dry_run:
            return self.plan(stage)
        selected=self.selected_stages(stage)
        marker=self.output/".preparation"/"owner.json"
        if self.output.exists() and any(self.output.iterdir()) and not marker.exists():
            raise FileExistsError(f"Output directory is not owned by this pipeline: {self.output}. Choose a new directory.")
        self.output.mkdir(parents=True,exist_ok=True)
        gpu_lock=file_lock(Path(tempfile.gettempdir())/f"content-preparation-gpu-{os.getuid()}.lock") if self.cfg["renderer"]["backend"]=="upstream_cuda" or self.cfg["training"]["backend"]=="upstream" else nullcontext()
        with file_lock(self.output/".preparation"/"run.lock"),gpu_lock:
            if marker.exists():
                prior=json.loads(marker.read_text())
                if prior["upstream_sources"]!=self.snapshot and not self.overwrite:
                    raise RuntimeError("Upstream sources changed since preparation; use a new output or explicit --overwrite")
            write_json(marker,{"pipeline_version":__version__,"upstream_sources":self.snapshot})
            self.journal=Journal(self.output,self.resume,self.overwrite)
            # BEGIN PROGRESS DISPLAY ONLY
            observe_stages(self)
            # END PROGRESS DISPLAY ONLY
            write_json(self.output/".preparation"/"resolved_config.json",self.cfg)
            try:
                for obj in self.cfg["objects"]:
                    self.obj=obj; self.root=self.output/obj["id"]; self.root.mkdir(parents=True,exist_ok=True)
                    for name in selected:
                        print(f"[{obj['id']}] {name}",flush=True)
                        with self.journal.stage(f"{obj['id']}/{name}"):
                            getattr(self,f"stage_{name}")()
                if stage in ("all","manifest"):
                    self._server_index()
            finally:
                after=source_snapshot()
                write_json(self.output/"validation_upstream.json",{"unchanged":after==self.snapshot,"before":self.snapshot,"after":after})
                if after!=self.snapshot:
                    raise RuntimeError("Protected upstream training/renderer/preprocessing source changed")
        return {"output":str(self.output),"tasks_executed":self.journal.executed,"tasks_skipped":self.journal.skipped,"stages":list(selected),"upstream_unchanged":True}

    def task(self,stage,key,inputs,action,command=None,extra=None):
        relevant={"preprocess":("runtime",),"train":("training","runtime"),"export":(),"encode":("encoding","delivery_modes"),
                  "package":("packaging",),"decode":(),"profile":("sampling","renderer","metrics"),"proxy":("sampling","renderer","proxy"),"manifest":("metrics","proxy",)}[stage]
        cfg={k:self.cfg[k] for k in relevant}
        cfg.update(object=self.obj,code_hash=self.code_hash,runtime_signature=self.runtime_signature,extra=extra)
        return self.journal.run(f"{self.obj['id']}/{stage}/{key}",cfg,inputs,action,command,__version__)

    def json_task(self,stage,key,path,value,inputs):
        def action():
            write_json(path,value)
            return value,[path]
        return self.task(stage,key,inputs,action)

    def read(self,name):
        path=self.root/name
        if not path.exists():
            raise FileNotFoundError(f"Missing prerequisite {path}; run preceding stages first")
        return json.loads(path.read_text())

    def rel(self,path):
        return Path(path).relative_to(self.root).as_posix()

    def _checkpoint_manifest(self):
        path=resolve_path(self.obj["dataset"]["checkpoint_manifest"])
        value=json.loads(path.read_text())
        for row in value["frames"]:
            for q in self.obj["qualities"]:
                if q["id"] in row:
                    row[q["id"]]=str(resolve_path(row[q["id"]],path.parent))
        return value,path

    def stage_preprocess(self):
        records=[]; dataset=self.obj["dataset"]; kind=dataset["kind"]
        if kind=="checkpoints":
            manifest,path=self._checkpoint_manifest()
            records=[{"frame":r["frame"],"checkpoint_import":True} for r in frame_records(self.obj)]
            self.json_task("preprocess","index",self.root/"preprocess.json",{"kind":kind,"frames":records,"source_sha256":sha256(path)},[path])
            return
        for record in frame_records(self.obj):
            frame=record["frame"]
            if kind=="synthetic":
                from .assets import save_state
                path=self.root/"input"/f"{frame}.npz"
                def action(path=path,frame=frame):
                    save_state(path,fixture_state(frame,len(self.obj["qualities"])-1,self.cfg["runtime"]["seed"]))
                    return {"frame":frame,"input":self.rel(path)},[path]
                result=self.task("preprocess",str(frame),[],action)
            elif kind=="prepared":
                sources={q["id"]:str(resolve_path(dataset["source_template"].format(frame=frame,quality=q["id"],scale=q["resolution_scale"],object=self.obj["id"]))) for q in self.obj["qualities"]}
                path=self.root/"input"/f"{frame}.json"
                def action(sources=sources,path=path,frame=frame):
                    from .upstream import initialize_prepared_points
                    owned={}
                    for q,source in sources.items():
                        target=self.root/"input"/str(frame)/q
                        if Path(source).resolve().is_relative_to(target.resolve()) or target.resolve().is_relative_to(Path(source).resolve()):
                            raise ValueError("Prepared input and pipeline-owned copy paths may not overlap")
                        if target.exists(): shutil.rmtree(target)
                        shutil.copytree(source,target)
                        for split in ("train","test"):
                            if not (target/f"transforms_{split}.json").exists():
                                raise ValueError(f"Prepared source lacks transforms_{split}.json: {source}")
                        initialize_prepared_points(target/"points3d.ply",self.cfg["runtime"]["seed"])
                        owned[q]=str(target)
                    value={"frame":frame,"sources":owned,"canonical":dataset.get("canonical",{"normalization":"provided prepared coordinates"})}
                    write_json(path,value)
                    return value,[path,*[p for q in owned.values() for p in sorted(Path(q).rglob("*")) if p.is_file()]]
                result=self.task("preprocess",str(frame),list(sources.values()),action)
            else:
                raw=resolve_path(dataset["raw_template"].format(frame=frame,object=self.obj["id"]))
                poses={s:str(resolve_path(dataset.get(f"{s}_poses",str(UPSTREAM_ROOT/f"transforms_{s}.json")))) for s in ("train","test")}
                dest=self.root/"input"/str(frame)
                spec={"raw_path":str(raw),"dest":str(dest),"poses":poses,"scales":[q["resolution_scale"] for q in self.obj["qualities"]],"width":dataset.get("width",1024),"seed":self.cfg["runtime"]["seed"]}
                spec_path=self.root/".work"/f"preprocess_{frame}.json"
                cmd=[sys.executable,"-m","tools.content_preparation.upstream","--preprocess-spec",str(spec_path)]
                def action(spec=spec,spec_path=spec_path,cmd=cmd,dest=dest,frame=frame):
                    # Failed tasks may leave partial images; restart this one frame only.
                    if dest.exists(): shutil.rmtree(dest)
                    write_json(spec_path,spec)
                    run_command(cmd,self.root/"logs"/f"preprocess_{frame}.log",self.cfg)
                    value={"frame":frame,"sources":{q["id"]:str(dest/f"res{q['resolution_scale']}") for q in self.obj["qualities"]},"canonical":json.loads((dest/"canonical.json").read_text())}
                    return value,[p for p in sorted(dest.rglob("*")) if p.is_file()]
                result=self.task("preprocess",str(frame),[raw,*poses.values()],action,cmd,extra=spec)
            records.append(result)
        files=[self.root/r["input"] for r in records if "input" in r]
        if kind!="synthetic": files += [Path(s) for r in records for s in r.get("sources",{}).values()]
        self.json_task("preprocess","index",self.root/"preprocess.json",{"kind":kind,"frames":records},files)

    def stage_train(self):
        from .assets import save_state
        prepared=self.read("preprocess.json"); backend=self.cfg["training"]["backend"]
        imported,import_path=self._checkpoint_manifest() if backend=="existing" else ({},None)
        frame_rows={r["frame"]:dict(r) for r in frame_records(self.obj)}; commands=[]
        for qi,quality in enumerate(self.obj["qualities"]):
            q=quality["id"]; previous=None
            for fi,record in enumerate(frame_records(self.obj)):
                frame=record["frame"]; p=next(r for r in prepared["frames"] if r["frame"]==frame)
                if backend=="synthetic_fixture":
                    path=self.root/"checkpoints"/q/f"{frame}.npz"
                    def action(path=path,frame=frame,qi=qi):
                        save_state(path,fixture_state(frame,qi,self.cfg["runtime"]["seed"]))
                        return {"frame":frame,"level":q,"path":str(path),"sha256":sha256(path),"synthetic_fixture":True,"argv":[]},[path]
                    result=self.task("train",f"{q}/{frame}",[self.root/p["input"]],action)
                elif backend=="existing":
                    source=Path(next(r for r in imported["frames"] if r["frame"]==frame)[q])
                    original=next((c for c in imported.get("commands",[]) if c["frame"]==frame and c["level"]==q),{})
                    if "progressive" in self.cfg["delivery_modes"] and (not original.get("sha256") or not original.get("argv")):
                        raise ValueError(f"Progressive checkpoint import requires ORIGINAL training argv and sha256 for {q}/{frame}; newly hashing a checkpoint does not prove lineage")
                    if original.get("sha256") and original["sha256"]!=sha256(source): raise ValueError("Imported checkpoint provenance mismatch")
                    path=self.root/"checkpoints"/q/f"{frame}.json"
                    result=self.json_task("train",f"{q}/{frame}",path,{"frame":frame,"level":q,"path":str(source),"sha256":sha256(source),"argv":original.get("argv",[])},[import_path,source])
                else:
                    source=Path(p["sources"][q]); model=self.root/"models"/q/str(frame)
                    steps=self.cfg["training"]["initial_iterations" if fi==0 else "dynamic_iterations"]
                    foundation=frame_rows[frame][self.obj["qualities"][qi-1]["id"]] if fi==0 and qi>0 else None
                    cmd=build_training_command(source,model,quality,self.cfg["training"],steps,previous,foundation)
                    path=model/"point_cloud"/f"iteration_{steps}"/"point_cloud.ply"
                    def action(cmd=cmd,path=path,model=model,frame=frame,q=q):
                        if model.exists(): shutil.rmtree(model)
                        run_command(cmd,self.root/"logs"/f"train_{q}_{frame}.log",self.cfg)
                        if not path.is_file(): raise RuntimeError(f"Training produced no checkpoint: {path}")
                        return {"frame":frame,"level":q,"path":str(path),"sha256":sha256(path),"argv":cmd},[path,self.root/"logs"/f"train_{q}_{frame}.log"]
                    result=self.task("train",f"{q}/{frame}",[source,*([previous] if previous else []),*([foundation] if foundation else [])],action,cmd)
                frame_rows[frame][q]=result["path"]; previous=result["path"]; commands.append(dict(result))
        manifest={"frames":list(frame_rows.values()),"commands":commands,"backend":backend,"upstream_sources":self.snapshot}
        if ("progressive" in self.cfg["delivery_modes"] and backend!="synthetic_fixture") or backend=="upstream":
            verify_checkpoint_lineage(manifest,[q["id"] for q in self.obj["qualities"]])
            manifest["lineage_verified"]=True
        else: manifest["lineage_verified"]=backend=="synthetic_fixture"
        self.json_task("train","index",self.root/"training.json",manifest,[Path(c["path"]) for c in commands])

    def stage_export(self):
        from .assets import load_state,read_ply,save_state,state_hash
        training=self.read("training.json"); exported=[]
        for quality in self.obj["qualities"]:
            q=quality["id"]
            for row in training["frames"]:
                frame=row["frame"]; source=Path(row[q]); path=self.root/"qualities"/q/f"{frame}.npz"
                def action(source=source,path=path,frame=frame,q=q):
                    state=read_ply(source) if source.suffix==".ply" else load_state(source)
                    if training["lineage_verified"]:
                        state.ids=np.arange(len(state.ids),dtype="int64")
                        state.metadata.update(stable_ids=True,lineage_verified=True)
                    state.metadata.update(frame=frame,quality=q,checkpoint_sha256=sha256(source))
                    save_state(path,state)
                    return {"quality":q,"frame":frame,"path":self.rel(path),"state_hash":state_hash(state),"checkpoint_sha256":sha256(source)},[path]
                exported.append(self.task("export",f"{q}/{frame}",[source,self.root/"training.json"],action))
        self.json_task("export","index",self.root/"export.json",{"states":exported},[self.root/r["path"] for r in exported])

    def stage_encode(self):
        from .assets import load_state
        from .codec_adapter import get_codec
        states=self.read("export.json")["states"]; records=[]
        for mode in self.cfg["delivery_modes"]:
            for qi,quality in enumerate(self.obj["qualities"]):
                q=quality["id"]; layer=q if mode=="independent" else ("Base" if qi==0 else f"E{qi}")
                config=dict(self.cfg["encoding"],**quality.get("encoding",{}))
                codec=get_codec(config)
                for frame in frame_records(self.obj):
                    f=frame["frame"]; source=self.root/next(r["path"] for r in states if r["quality"]==q and r["frame"]==f)
                    parent_layer=("Base" if qi==1 else f"E{qi-1}") if mode=="progressive" and qi>0 else None
                    parent_path=self.root/".work"/"encoder_decoded"/mode/parent_layer/f"{f}.npz" if parent_layer else None
                    path=self.root/"encoded"/mode/layer/f"{f}.cpgs"; decoded=self.root/".work"/"encoder_decoded"/mode/layer/f"{f}.npz"
                    def action(source=source,path=path,decoded=decoded,parent_path=parent_path,config=config,codec=codec,q=q,layer=layer,mode=mode,f=f,parent_layer=parent_layer):
                        from .assets import save_state
                        parent=load_state(parent_path) if parent_path else None
                        meta=codec.encode(load_state(source),path,config,parent)
                        state=codec.decode(path,parent)
                        save_state(decoded,state)
                        record=dict(meta,id=f"{mode}:{layer}:{f}",mode=mode,quality=q,layer=layer,frame=f,path=self.rel(path),
                                    dependencies=[f"{mode}:{parent_layer}:{f}"] if parent_layer else [],parent_layer=parent_layer)
                        record["encoder_decoded_path"]=self.rel(decoded)
                        return record,[path,decoded]
                    records.append(self.task("encode",f"{mode}/{layer}/{f}",[source,*([parent_path] if parent_path else [])],action,extra=config))
        self.json_task("encode","index",self.root/"encoding.json",{"payloads":records},[self.root/r["path"] for r in records])

    def stage_package(self):
        from .packaging import package_object
        records=self.read("encoding.json")["payloads"]
        def action():
            index=package_object(self.obj["id"],frame_records(self.obj),self.obj["qualities"],records,self.cfg["packaging"],self.root)
            path=self.root/"package.json";write_json(path,index)
            return {"path":self.rel(path)},[path,self.root/"package_index.json",*[self.root/s["path"] for s in index["segments"]]]
        self.task("package","index",[self.root/"encoding.json",*[self.root/r["path"] for r in records]],action)

    def stage_decode(self):
        from .assets import load_state,save_state,state_hash
        from .codec_adapter import get_decoder
        from .packaging import extract_segment_member
        index=self.read("package.json"); decoded_records=[]
        (self.root/".work").mkdir(parents=True,exist_ok=True)
        for record in index["payloads"]:
            mode,layer,f=record["mode"],record["layer"],record["frame"]
            parent_layer=record.get("parent_layer")
            parent_path=self.root/"decoded"/mode/parent_layer/f"{f}.npz" if parent_layer else None
            path=self.root/"decoded"/mode/layer/f"{f}.npz"
            segment=next(s for s in index["segments"] if record["id"] in [m["id"] for m in s["members"]])
            quality=next(q for q in self.obj["qualities"] if q["id"]==record["quality"])
            runtime_config=dict(self.cfg["encoding"],**quality.get("encoding",{}))
            def action(record=record,path=path,parent_path=parent_path,segment=segment,runtime_config=runtime_config):
                parent=load_state(parent_path) if parent_path else None
                data=extract_segment_member(self.root/segment["path"],record["id"])
                if sha256(self.root/segment["path"])!=segment["sha256"]: raise ValueError("Segment checksum changed")
                with tempfile.NamedTemporaryFile(dir=self.root/".work",suffix=".cpgs") as tmp:
                    tmp.write(data);tmp.flush()
                    state=get_decoder(record["codec"],runtime_config).decode(Path(tmp.name),parent)
                if state_hash(state)!=record["decoded_state_hash"]: raise ValueError("Decoded state hash differs from encoded reconstruction")
                save_state(path,state)
                return {"mode":record["mode"],"quality":record["quality"],"layer":record["layer"],"frame":record["frame"],"path":self.rel(path),"state_hash":state_hash(state)},[path]
            decoded_records.append(self.task("decode",f"{mode}/{layer}/{f}",[self.root/segment["path"],*([parent_path] if parent_path else [])],action))
        self.json_task("decode","index",self.root/"decoding.json",{"states":decoded_records,"source":"requestable segment containers only"},[self.root/r["path"] for r in decoded_records])

    def samples(self):
        cfg=self.cfg["sampling"]
        for record in frame_records(self.obj):
            if cfg["times"]!="all" and record["frame"] not in cfg["times"]: continue
            for ai,ei,si in product(range(len(cfg["azimuth"])),range(len(cfg["elevation"])),range(len(cfg["scales"]))):
                yield dict(record,azimuth=cfg["azimuth"][ai],elevation=cfg["elevation"][ei],scale=cfg["scales"][si],distance=cfg["distance"],
                           view_bin=f"a{ai}_e{ei}",scale_bin=f"s{si}",time_bin=str(record["frame"]),sample_id=f"{record['frame']}_a{ai}_e{ei}_s{si}")

    def _render_task(self,stage,key,state_path,path,sample):
        from .assets import load_state
        from .renderer_adapter import render
        def action():
            settings=dict(self.cfg["renderer"])
            extension=self.cfg["runtime"].get("extension_path")
            if extension:
                extension_path = resolve_path(extension)
                if not extension_path.is_dir():
                    raise FileNotFoundError(f"Configured extension_path does not exist: {extension_path}")
                settings.setdefault("vendor_path", str(extension_path))
            image=render(load_state(state_path),sample,settings)
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez(path,rgb=image["rgb"],alpha=image["alpha"],depth=image["depth"])
            return {"path":self.rel(path),"renderer":image.get("metadata",{})},[path]
        return self.task(stage,key,[state_path],action,extra=sample)

    def stage_profile(self):
        from .metrics import MetricEvaluator
        from .quality_profile import QualityProfile
        states=self.read("decoding.json")["states"]; samples=list(self.samples()); rows=[]
        highest=self.obj["qualities"][-1]["id"]
        evaluator=MetricEvaluator(self.cfg["metrics"])
        try:
            for mode in self.cfg["delivery_modes"]:
                for quality in [highest]+[q["id"] for q in self.obj["qualities"] if q["id"]!=highest]:
                    for sample in samples:
                        key=f"{mode}/{quality}/{sample['sample_id']}"
                        state=self.root/next(r["path"] for r in states if r["mode"]==mode and r["quality"]==quality and r["frame"]==sample["frame"])
                        rendered=self.root/"profiles"/"images"/mode/quality/f"{sample['sample_id']}.npz"
                        self._render_task("profile",f"render/{key}",state,rendered,sample)
                        ref=self.root/"profiles"/"images"/mode/highest/f"{sample['sample_id']}.npz"
                        metric_path=self.root/"profiles"/"samples"/mode/quality/f"{sample['sample_id']}.json"
                        def action(rendered=rendered,ref=ref,sample=sample,quality=quality,mode=mode,metric_path=metric_path):
                            with np.load(rendered) as image,np.load(ref) as reference:
                                metric=evaluator.compare(image["rgb"],reference["rgb"]).to_dict()
                            row=dict(sample,object=self.obj["id"],quality=quality,mode=mode,metrics=metric,
                                     reference_quality=highest,reference_state="decoded",image_path=self.rel(rendered))
                            write_json(metric_path,row)
                            return row,[metric_path]
                        rows.append(self.task("profile",f"metrics/{key}",[rendered,ref],action,extra=sample))
        finally: evaluator.close()
        profile=QualityProfile(rows,metadata={"object":self.obj["id"],"reference_quality":highest,"reference":"decoded requestable payload", "renderer_backend":self.cfg["renderer"]["backend"],"renderer_settings":self.cfg["renderer"],"metrics":self.cfg["metrics"],"planner_distortion":self.cfg["metrics"]["distortion"]})
        paths=[self.root/"profiles"/"profile.json",self.root/"profiles"/"gains.json",self.root/"profiles"/"aggregates.json"]
        def action():
            profile.save(paths[0]);write_json(paths[1],profile.transition_gains([q["id"] for q in self.obj["qualities"]]));write_json(paths[2],profile.summarize())
            return {"rows":len(rows),"reference_quality":highest},paths
        self.task("profile","index",[self.root/r["image_path"] for r in rows]+sorted((self.root/"profiles"/"samples").rglob("*.json")),action)

    def stage_proxy(self):
        from .assets import load_state,save_state
        from .proxy import generate_proxy,stable_subset,validate_proxy_render
        states=self.read("decoding.json")["states"]; mode=self.cfg["delivery_modes"][0]; highest=self.obj["qualities"][-1]["id"]
        records=[]; ids=None;frames=frame_records(self.obj)
        for i,row in enumerate(frames):
            f=row["frame"];source=self.root/next(s["path"] for s in states if s["mode"]==mode and s["quality"]==highest and s["frame"]==f)
            path=self.root/"proxy"/f"{f}.npz"
            def action(source=source,path=path,ids=ids,f=f):
                state=load_state(source)
                if ids is None:
                    from .proxy import load_proxy
                    with tempfile.TemporaryDirectory(dir=self.root/".work",prefix="proxy_") as tmp:
                        proxy=generate_proxy(state,self.cfg["proxy"],tmp,frame=f)
                        state=load_proxy(proxy["descriptor_path"],frame=f)
                else:
                    if not state.metadata.get("stable_ids"):
                        raise ValueError("Temporal proxy requires verified stable IDs; supply training lineage or explicit Gaussian IDs")
                    state=stable_subset(state,ids)
                save_state(path,state)
                return {"frame":f,"path":self.rel(path),"ids":state.ids.tolist(),"source_quality":highest,"selection":"frozen first-frame stable IDs"},[path]
            record=self.task("proxy",f"state/{f}",[source,*([self.root/records[0]["path"]] if records else [])],action)
            if ids is None: ids=np.asarray(record["ids"],dtype="int64")
            records.append(record)
        validation=[]
        for sample in self.samples():
            state=self.root/next(r["path"] for r in records if r["frame"]==sample["frame"])
            path=self.root/"proxy"/"renders"/f"{sample['sample_id']}.npz"
            self._render_task("proxy",f"render/{sample['sample_id']}",state,path,sample)
            ref=self.root/"profiles"/"images"/mode/highest/f"{sample['sample_id']}.npz"
            if not ref.exists():
                high=self.root/next(r["path"] for r in states if r["mode"]==mode and r["quality"]==highest and r["frame"]==sample["frame"])
                ref=self.root/"proxy"/"reference"/f"{sample['sample_id']}.npz"
                self._render_task("proxy",f"reference/{sample['sample_id']}",high,ref,sample)
            metric_path=self.root/"proxy"/"validation"/f"{sample['sample_id']}.json"
            def action(path=path,ref=ref,sample=sample,metric_path=metric_path):
                with np.load(path) as p,np.load(ref) as r:
                    row=dict(sample,metrics=validate_proxy_render(dict(p),dict(r),self.cfg["proxy"]["alpha_threshold"]))
                write_json(metric_path,row)
                return row,[metric_path]
            validation.append(self.task("proxy",f"validation/{sample['sample_id']}",[path,ref],action,extra=sample))
        from .assets import state_hash
        descriptor={"schema":"dynamic-lapisgs.proxy.v1","backend":"fixed_gaussian_subset","quality_independent":True,"display_asset":False,
                    "selection_quality":highest,"selection_frame":frames[0]["frame"],"configured_count":self.cfg["proxy"]["max_gaussians"],
                    "selected_ids":ids.tolist(),"selected_count":len(ids),"selection_policy":"first_highest_decoded_frame_opacity_descending_ID_tiebreak",
                    "depth_convention":"coverage_conditioned_expected_camera_z", "frames":[{"frame":r["frame"],"path":Path(r["path"]).name,
                    "bytes":(self.root/r["path"]).stat().st_size,"sha256":sha256(self.root/r["path"]),"state_sha256":state_hash(load_state(self.root/r["path"]))} for r in records]}
        self.json_task("proxy","index",self.root/"proxy"/"index.json",descriptor,[self.root/r["path"] for r in records])
        self.json_task("proxy","validation_index",self.root/"proxy"/"validation.json",{"samples":validation,"reference":"highest decoded", "optimization_performed":False},sorted((self.root/"proxy"/"validation").glob("*.json")))

    def stage_manifest(self):
        from .manifest import build_manifest,write_manifest,load_manifest
        index=self.read("package.json");training=self.read("training.json");prepared=self.read("preprocess.json")
        proxy_enabled=self.cfg["proxy"]["enabled"]
        proxy=self.read("proxy/index.json") if proxy_enabled else None
        proxy_path="proxy/index.json" if proxy_enabled else self.cfg["proxy"].get("external_descriptor")
        if proxy_path:
            from .validation import safe_relative_path
            safe_relative_path(proxy_path)
            descriptor=(self.root/proxy_path).resolve()
            if not descriptor.is_relative_to(self.root.resolve()) or not descriptor.is_file():
                raise ValueError("External proxy descriptor must already be published within the object output directory")
        profile=self.read("profiles/profile.json")
        canonical={"matrix":np.eye(4).tolist(),"scale":1.,"coordinate_system":"prepared renderer coordinates","up_axis":self.cfg["renderer"].get("up_axis","y"),"frame_transforms":[{"frame":r["frame"],**r["canonical"]} for r in prepared["frames"] if "canonical" in r]}
        provenance={"pipeline_version":__version__,"implementation_sha256":self.code_hash,"config_sha256":config_hash(self.cfg),"upstream_sources":self.snapshot,
                    "training_backend":training["backend"],"renderer_backend":self.cfg["renderer"]["backend"],"lineage_verified":training["lineage_verified"],
                    "checkpoints":[{"quality":r["level"],"frame":r["frame"],"sha256":r["sha256"]} for r in training["commands"]]}
        from .upstream import runtime_provenance
        provenance["runtime"]=runtime_provenance(self.cfg)
        path=self.root/"manifest.json"
        def action():
            manifest=build_manifest(self.obj["id"],frame_records(self.obj),self.obj["qualities"],index,provenance,canonical,proxy_path,"profiles/profile.json",self.root)
            manifest["object"]["proxy_preparation"]="generated" if proxy_enabled else ("external" if proxy_path else "disabled")
            manifest["object"].update(fps=self.obj["fps"],sample_duration_seconds=1./self.obj["fps"],
                                    duration_seconds=manifest["object"]["time_range"][1]-manifest["object"]["time_range"][0]+1./self.obj["fps"],
                                    duration_convention="last timestamp - first timestamp + one source frame interval")
            manifest["planner_distortion"]=self.cfg["metrics"]["distortion"]
            # Only generated proxies have frame/validation artifacts in this run.
            assets={a["path"]:a for a in manifest["assets"]}
            extra_assets=["profiles/gains.json","profiles/aggregates.json"]
            if proxy_enabled:
                extra_assets += ["proxy/"+r["path"] for r in proxy["frames"]]+["proxy/validation.json"]
            for relative in extra_assets:
                p=self.root/relative;assets[relative]={"path":relative,"bytes":p.stat().st_size,"sha256":sha256(p)}
            manifest["assets"]=list(assets.values());manifest["accounting"]["auxiliary_asset_bytes"]=sum(a["bytes"] for a in assets.values())
            write_manifest(path,manifest);load_manifest(path,validate_files=True)
            valid=self.root/"validation.json"
            from .renderer_adapter import GPUExecutionGuard
            renders=[v["result"]["renderer"] for k,v in self.journal.data["tasks"].items() if k.startswith(self.obj["id"]+"/") and v["status"]=="complete" and isinstance(v.get("result"),dict) and "renderer" in v["result"]]
            write_json(valid,{"passed":True,"decoded_from_segments":True,"profile_rows":len(profile.get("rows",profile.get("samples",[]))),"proxy_frames":len(proxy["frames"]) if proxy_enabled else 0,"proxy_preparation":manifest["object"]["proxy_preparation"],"upstream_unchanged":source_snapshot()==self.snapshot,"gpu_execution":"serialized process lock + per-view renderer guard","gpu_guard":GPUExecutionGuard.statistics(),"render_peak_cuda_allocated_bytes":max((r.get("peak_cuda_allocated_bytes",0) for r in renders),default=0),"lpips_batch_size":1})
            return {"manifest":self.rel(path)},[path,valid]
        inputs=[self.root/"package.json",self.root/"training.json",self.root/"profiles/profile.json",self.root/"profiles/gains.json",self.root/"profiles/aggregates.json"]
        if proxy_path:
            inputs.append(self.root/proxy_path)
        if proxy_enabled:
            inputs += [self.root/"proxy"/a["path"] for a in proxy["frames"]]+[self.root/"proxy/validation.json"]
        self.task("manifest","index",inputs,action)

    def _server_index(self):
        manifests=[self.output/o["id"]/"manifest.json" for o in self.cfg["objects"]]
        value={"schema_version":"content-preparation.server.v1","objects":[{"id":o["id"],"manifest_path":str(p.relative_to(self.output)),"sha256":sha256(p),"bytes":p.stat().st_size} for o,p in zip(self.cfg["objects"],manifests)]}
        self.journal.run("server/manifest",{"code_hash":self.code_hash},manifests,lambda:(write_json(self.output/"manifest.json",value) or value,[self.output/"manifest.json"]))
