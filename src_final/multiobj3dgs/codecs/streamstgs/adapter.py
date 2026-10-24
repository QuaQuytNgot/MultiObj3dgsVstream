"""Staged decoder extracted from the validated Soldier runtime.

CPU stage preserves upstream float64 normalization then float32 tensor conversion;
no model is constructed and CUDA is never touched by preparation workers.
"""
from dataclasses import dataclass, field
from concurrent.futures import Future
from collections import deque
import csv
import json
from pathlib import Path
import subprocess
import threading
import time
import weakref
from typing import Any
import numpy as np
from ...shared.gaussian import GaussianTensorState
from .vendor import model_class, vendor_root

MLPS=('mlp_time','mlp_deform','mlp_cov','mlp_opacity','mlp_color')
SHAPES={'_xyz':3,'_features_dc':3,'_rotation':4,'_scaling':3,'_opacity':1}


def check_cancel(cancel):
    if cancel is not None and (cancel.is_set() if hasattr(cancel,'is_set') else cancel()):
        from concurrent.futures import CancelledError
        raise CancelledError('GOP preparation cancelled')


def safe_file(package, name):
    path=(package/name).resolve()
    if not path.is_relative_to(package.resolve()) or not path.is_file():
        raise ValueError(f'Missing or unsafe codec member: {name}')
    return path


def decode_video(video, expected, grid, feat_dim=12, threads=1, cancel=None, atlas_shape=None):
    check_cancel(cancel)
    stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(video)]))['streams'][0]
    w,h=int(stream['width']),int(stream['height'])
    expected_columns=feat_dim//4; expected_rows=4
    if (h,w) not in ((expected_rows*grid,expected_columns*grid),(expected_columns*grid,expected_rows*grid)):
        raise ValueError(f'Unsupported atlas dimensions {w}x{h} for grid {grid}, features {feat_dim}')
    if atlas_shape is not None and (h,w)!=tuple(int(n)*grid for n in atlas_shape):
        raise ValueError('Feature video dimensions disagree with declared atlas_rows/atlas_columns')
    # Decode one atlas at a time so cancellation does not retain a whole raw GOP.
    cmd=['ffmpeg','-nostdin','-v','error','-threads',str(threads),'-i',str(video),'-f','rawvideo','-pix_fmt','gray16be','pipe:1']
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    size=w*h*2
    try:
        for _ in range(expected):
            check_cancel(cancel)
            chunks=[];remaining=size
            while remaining:
                chunk=p.stdout.read(remaining)
                if not chunk: raise ValueError('Truncated feature video')
                chunks.append(chunk);remaining-=len(chunk)
            raw=b''.join(chunks)
            atlas=np.frombuffer(raw,dtype='>u2').reshape(h,w)
            yield atlas
        if p.stdout.read(1): raise ValueError('Unexpected extra feature atlases')
        if p.wait()!=0: raise ValueError('FFmpeg feature decode failed')
    finally:
        p.stdout.close()
        if p.poll() is None:
            p.terminate()
            try:p.wait(timeout=2)
            except subprocess.TimeoutExpired:p.kill();p.wait()


def unpack_features(atlas, grid, feat_dim, minimum, maximum, normalized=True):
    # Tiles are row-major in the encoded picture. Swapping atlas axes also
    # transposes every tile and permutes channels, breaking the correspondence
    # between each Gaussian's canonical attributes and temporal features.
    if atlas.ndim!=2 or any(n%grid for n in atlas.shape) or atlas.size!=grid*grid*feat_dim:
        raise ValueError('Feature atlas must contain exactly feat_dim square tiles')
    result=np.empty((feat_dim,grid,grid),dtype=np.float32)
    scratch=np.empty((grid,grid),dtype=np.float64)
    columns=atlas.shape[1]//grid
    for channel in range(feat_dim):
        row,col=divmod(channel,columns)
        np.divide(atlas[row*grid:(row+1)*grid,col*grid:(col+1)*grid],65535.,out=scratch)
        if normalized:
            # Preserve float64 divide/multiply/add order, with one tile scratch.
            scratch*=maximum-minimum;scratch+=minimum
        result[channel]=scratch
    return np.ascontiguousarray(result.reshape(feat_dim,-1).T)


@dataclass
class PreparedGopHost:
    package: Path
    reconstruction: dict
    cameras: list
    attributes: dict
    features: tuple
    weights: dict
    stages: dict = field(default_factory=dict)
    complete: bool=True
    error: Any=None
    produced_features: int=0

    @property
    def nbytes(self):
        m=self.reconstruction['model']
        features=int(m['grid_sidelen'])**2*int(m['feat_dim'])*int(m['time_embedding_num'])*4
        return sum(x.nbytes for x in self.attributes.values())+features+sum(t.numel()*t.element_size() for state in self.weights.values() for t in state.values())


@dataclass
class ReadyGopHandle:
    model: Any
    reconstruction: dict
    cameras: list
    nbytes: int
    device: str
    stages: dict=field(default_factory=dict)
    released: bool=False
    available_features: Any=None
    transfer: Any=None


@dataclass
class PendingGopUpload:
    prepared: PreparedGopHost
    model: Any
    plan: tuple
    started: float
    stages: dict
    cursor: int=0
    completion: Any=None
    feature_events: Any=field(default_factory=deque)
    available_features: int=0
    handle: Any=None
    done: bool=False
    offset: int=0


class _UploadBuffer:
    def __init__(self,tensor,event,start):
        self.tensor,self.event,self.start=tensor,event,start
        self.busy=False
        self.pending=None


class StreamSTGSAdapter:
    def __init__(self,vendor_root=None,device='cuda',ffmpeg_threads=1):
        self.vendor_root=vendor_root;self.device=device;self.ffmpeg_threads=ffmpeg_threads;self._owner=None
        self._upload_stream=None;self._upload_buffers=[]
        if ffmpeg_threads<1:raise ValueError('ffmpeg_threads must be positive')

    def preparation_key(self,segment):
        """Only immutable CPU results may be shared; GPU models stay independent."""
        return (segment.resource.sha256,tuple(sorted(a.sha256 for a in segment.shared_assets)))

    def _assert_owner(self):
        current=threading.get_ident()
        if self._owner is None:self._owner=current
        if current!=self._owner:raise RuntimeError('StreamSTGS GPU operations must run on their owner thread')

    def estimate_host_bytes(self,package_dir):
        package=Path(package_dir);m=json.loads(safe_file(package,'reconstruction.json').read_text())['model']
        n=int(m['grid_sidelen'])**2;d=int(m['feat_dim']);t=int(m['time_embedding_num'])
        resident=n*(14+d*t)*4+sum(p.stat().st_size*2 for p in package.glob('mlp_*.pth'))
        # Two float64 atlases + raw big-endian frame + float64 tile workspace.
        transient=n*d*(8*3+2*2)
        return resident+transient+16*1024**2

    def estimate_memory(self,package_dir):
        package=Path(package_dir);m=json.loads(safe_file(package,"reconstruction.json").read_text())["model"]
        n=int(m["grid_sidelen"])**2;d=int(m["feat_dim"]);t=int(m["time_embedding_num"])
        weights=sum(p.stat().st_size*2 for p in package.glob("mlp_*.pth"))
        resident=n*(14+d*t)*4+weights
        return {"raw_bytes":n*d*28+16*1024**2,"host_bytes":resident,"gpu_bytes":resident*2+16*1024**2}

    def estimate_device_bytes(self,prepared):
        # Resident model plus constructor/transfer duplication; frame workspaces are separate.
        return prepared.nbytes*2+16*1024**2

    def estimate_upload_bytes(self,prepared):
        # Incremental copies allocate each destination once; pinned staging is
        # charged to the raw host budget, and frame workspaces have their own
        # transient GPU reserve. Reserve the complete GOP even before it arrives.
        return prepared.nbytes+16*1024**2

    def prepare_cpu(self,package_dir,cancel=None):
        return self._prepare_cpu(package_dir,cancel)

    def start_cpu(self,package_dir,cancel,pool):
        """Publish canonical state early; the persistent CPU worker fills tiles.

        Host admission reserves the complete GOP before this starts. The worker
        owns its materialized package lease until producer completion, including
        cancellation. No CUDA is called on the preparation worker.
        """
        published=Future()
        def produce():
            try:return self._prepare_cpu(package_dir,cancel,published)
            except BaseException as exc:
                if not published.done():published.set_exception(exc)
                raise
        producer=pool.submit(produce)
        def stopped(future):
            if future.cancelled() and not published.done():published.cancel()
        producer.add_done_callback(stopped)
        return published,producer

    def _prepare_cpu(self,package_dir,cancel=None,published=None):
        import imagecodecs
        import torch
        package=Path(package_dir).resolve();started=time.perf_counter();check_cancel(cancel)
        reconstruction=json.loads(safe_file(package,'reconstruction.json').read_text());m=reconstruction['model']
        if reconstruction.get('upstream_commit')!='ca8602516352efd83e9e69e665b2e26ed03cb5cc':raise ValueError('Unsupported StreamSTGS upstream revision')
        # The pinned constructor fixes feat_dim/time_batch=12/3. Reject incompatible content.
        if (int(m['feat_dim']),int(m['time_batch']))!=(12,3):raise ValueError('Pinned StreamSTGS model requires feat_dim=12/time_batch=3')
        with safe_file(package,'compression_info.csv').open(newline='') as f:
            info={r['']:r for r in csv.DictReader(f)}
        experiment=reconstruction.get('experiment')
        if experiment is None:
            import yaml
            experiment=yaml.safe_load(safe_file(package,'compression_config.yaml').read_text())['experiments'][0]
        attributes={};stages={'metadata_cpu_s':time.perf_counter()-started};feature_config=None
        with torch.inference_mode():
            for config in experiment['attributes']:
                check_cancel(cancel);name=config['name'];row=info[name];t=time.perf_counter()
                if 'point_feat' in name:
                    feature_config=config
                    continue
                else:
                    if name not in SHAPES or config['method']!='jpeg-xl':raise ValueError(f'Unsupported canonical attribute {name}/{config["method"]}')
                    array=imagecodecs.imread(safe_file(package,row['file']))
                    if config.get('normalize',False):array=array*(float(row['max'])-float(row['min']))+float(row['min'])
                    if config.get('contract',False):array=np.sign(array)*np.expm1(np.abs(array))
                    attributes[name]=np.asarray(array).reshape(-1,SHAPES[name]).astype(np.float32)
                stages[name+'_cpu_s']=time.perf_counter()-t
            expected=int(m['grid_sidelen'])**2
            if set(attributes)!=set(SHAPES) or any(len(v)!=expected or not np.isfinite(v).all() for v in attributes.values()):raise ValueError('Canonical attribute shape/nonfinite mismatch')
            if feature_config is None:raise ValueError('Missing temporal feature configuration')
            weights={}
            weights_started=time.perf_counter()
            for name in MLPS:
                check_cancel(cancel)
                # weights_only refuses arbitrary pickle code; archives contain plain state_dicts.
                weights[name]=torch.load(safe_file(package,name+'.pth'),map_location='cpu',weights_only=True)
            stages['weights_cpu_s']=time.perf_counter()-weights_started
        config=feature_config;row=info[config['name']]
        minimum=float(row['min']) if config.get('normalize') else 0.;maximum=float(row['max']) if config.get('normalize') else 1.
        atlas_shape=(m['atlas_rows'],m['atlas_columns']) if 'atlas_rows' in m and 'atlas_columns' in m else None
        if m.get('channel_order','row-major tiles')!='row-major tiles':raise ValueError('Unsupported feature atlas channel_order')
        host=PreparedGopHost(package,reconstruction,json.loads(safe_file(package,'cameras.json').read_text()),attributes,[None]*int(m['time_embedding_num']),weights,stages,complete=False)
        stages['canonical_weights_ready_s']=time.perf_counter()-started
        if published is not None:published.set_result(host)
        t=time.perf_counter();read_s=unpack_s=0.
        try:
            iterator=iter(decode_video(safe_file(package,'_point_feats.mp4'),int(m['time_embedding_num']),int(m['grid_sidelen']),int(m['feat_dim']),self.ffmpeg_threads,cancel,atlas_shape))
            for index in range(int(m['time_embedding_num'])):
                before=time.perf_counter();atlas=next(iterator);read_s+=time.perf_counter()-before
                before=time.perf_counter()
                feature=unpack_features(atlas,int(m['grid_sidelen']),int(m['feat_dim']),minimum,maximum,config.get('normalize',False))
                if len(feature)!=expected or not np.isfinite(feature).all():raise ValueError('Feature shape/nonfinite mismatch')
                unpack_s+=time.perf_counter()-before
                host.features[index]=feature;host.produced_features=index+1
            # Consume EOF so extra pictures and FFmpeg failures are still checked.
            try:next(iterator);raise ValueError('Unexpected extra feature atlas')
            except StopIteration:pass
            stages[config['name']+'_cpu_s']=time.perf_counter()-t
            stages['feature_read_wait_s']=read_s;stages['feature_unpack_s']=unpack_s
            stages['prepare_cpu_s']=time.perf_counter()-started
            host.complete=True
            return host
        except BaseException as exc:
            host.error=exc;host.complete=True
            raise

    def realize_gpu(self,prepared):
        import torch
        self._assert_owner();started=time.perf_counter();m=prepared.reconstruction['model']
        if not torch.cuda.is_available():raise RuntimeError('StreamSTGS requires an available NVIDIA CUDA device')
        import_started=time.perf_counter();model_type=model_class(self.vendor_root)
        import_s=time.perf_counter()-import_started
        stages=dict(prepared.stages)
        stages['codec_import_s']=import_s
        with torch.cuda.device(self.device),torch.inference_mode():
            stage_started=time.perf_counter()
            pc=model_type(int(m['max_sh_degree']),bool(m['disable_xyz_log_activation']),eval=True)
            pc.time_line=int(m['time_line']);pc.time_embedding_num=int(m['time_embedding_num'])
            pc.feat_dim=int(m['feat_dim']);pc.time_batch=int(m['time_batch']);pc.total_feat_dim=pc.feat_dim*pc.time_batch
            pc.active_sh_degree=int(m['active_sh_degree']);pc.grid_sidelen=int(m['grid_sidelen'])
            torch.cuda.synchronize(self.device)
            stages['model_initialization_s']=time.perf_counter()-stage_started;stage_started=time.perf_counter()
            for name,array in prepared.attributes.items():setattr(pc,name,torch.tensor(array,device=self.device,dtype=torch.float32))
            for index,array in enumerate(prepared.features):
                tensor=torch.tensor(array,device=self.device,dtype=torch.float32)
                setattr(pc,f'_point_feat_{index}',tensor);pc._point_feats.append(tensor)
            torch.cuda.synchronize(self.device)
            stages['tensor_allocation_upload_s']=time.perf_counter()-stage_started;stage_started=time.perf_counter()
            stages['canonical_feature_upload_bytes']=sum(a.nbytes for a in prepared.attributes.values())+sum(a.nbytes for a in prepared.features)
            for name in MLPS:getattr(pc,name).load_state_dict(prepared.weights[name])
            pc.eval();torch.cuda.synchronize(self.device)
            stages['weight_binding_upload_s']=time.perf_counter()-stage_started
        stages['realize_gpu_s']=time.perf_counter()-started
        return ReadyGopHandle(pc,prepared.reconstruction,prepared.cameras,prepared.nbytes,self.device,stages)

    def begin_gpu(self,prepared):
        """Initialize one model; enqueue bulk copies incrementally on a copy stream.

        All methods stay on the GPU owner. At most two 8 MiB pinned staging
        buffers exist per adapter, and each is reused only after its CUDA event.
        """
        import torch
        self._assert_owner();started=time.perf_counter();m=prepared.reconstruction['model']
        import_started=time.perf_counter();model_type=model_class(self.vendor_root)
        import_s=time.perf_counter()-import_started
        if self._upload_stream is None:
            with torch.cuda.device(self.device):
                self._upload_stream=torch.cuda.Stream(device=self.device,priority=0)
                self._upload_buffers=[_UploadBuffer(torch.empty(2*1024**2,dtype=torch.float32,pin_memory=True),torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(2)]
        stages=dict(prepared.stages)
        stages['codec_import_s']=import_s
        stages.update(h2d_copy_gpu_s=0.,host_pack_cpu_s=0.,destination_allocation_cpu_s=0.)
        with torch.cuda.device(self.device),torch.cuda.stream(self._upload_stream),torch.inference_mode():
            t=time.perf_counter()
            pc=model_type(int(m['max_sh_degree']),bool(m['disable_xyz_log_activation']),eval=True)
            pc.time_line=int(m['time_line']);pc.time_embedding_num=int(m['time_embedding_num'])
            pc.feat_dim=int(m['feat_dim']);pc.time_batch=int(m['time_batch']);pc.total_feat_dim=pc.feat_dim*pc.time_batch
            pc.active_sh_degree=int(m['active_sh_degree']);pc.grid_sidelen=int(m['grid_sidelen'])
            stages['model_initialization_s']=time.perf_counter()-t;t=time.perf_counter()
            for name in MLPS:getattr(pc,name).load_state_dict(prepared.weights[name])
            pc.eval();stages['weight_binding_upload_s']=time.perf_counter()-t
        plan=tuple(prepared.attributes.items())+tuple((f'_point_feat_{i}',i) for i in range(len(prepared.features)))
        stages['canonical_feature_upload_bytes']=prepared.nbytes-sum(t.numel()*t.element_size() for state in prepared.weights.values() for t in state.values())
        stages['upload_owner_cpu_s']=time.perf_counter()-started
        return PendingGopUpload(prepared,pc,plan,started,stages)

    def poll_gpu(self,pending,budget_s=.002,max_bytes=16*1024**2):
        """Expose complete temporal windows; never expose unfinished H2D copies."""
        import torch
        self._assert_owner();started=time.perf_counter();copied=0
        if pending.prepared.error is not None:raise pending.prepared.error
        self._refresh_transfers(pending)
        with torch.cuda.device(self.device),torch.cuda.stream(self._upload_stream),torch.inference_mode():
            while pending.cursor<len(pending.plan) and copied<max_bytes and time.perf_counter()-started<budget_s:
                slot=next((s for s in self._upload_buffers if not s.busy or s.event.query()),None)
                if slot is None:break
                if slot.busy:
                    slot.pending.stages['h2d_copy_gpu_s']+=slot.start.elapsed_time(slot.event)/1000
                    slot.pending=None;slot.busy=False
                name,array=pending.plan[pending.cursor]
                feature_index=array if isinstance(array,int) else None
                if feature_index is not None:
                    array=pending.prepared.features[feature_index]
                    if array is None:break
                # Very large content is split across bounded staging slices too.
                tensor=getattr(pending.model,name,None)
                offset=pending.offset
                if offset==0:
                    t=time.perf_counter()
                    tensor=torch.empty(array.shape,device=self.device,dtype=torch.float32)
                    pending.stages['destination_allocation_cpu_s']+=time.perf_counter()-t
                    setattr(pending.model,name,tensor)
                    if name.startswith('_point_feat_'):pending.model._point_feats.append(tensor)
                flat=array.reshape(-1);n=min(len(flat)-offset,slot.tensor.numel())
                t=time.perf_counter();slot.tensor[:n].copy_(torch.from_numpy(flat[offset:offset+n]))
                pending.stages['host_pack_cpu_s']+=time.perf_counter()-t
                slot.start.record(self._upload_stream)
                tensor.view(-1)[offset:offset+n].copy_(slot.tensor[:n],non_blocking=True)
                slot.event.record(self._upload_stream);slot.busy=True;slot.pending=pending
                copied+=n*4;offset+=n
                if offset==len(flat):
                    if feature_index is not None:
                        event=torch.cuda.Event();event.record(self._upload_stream)
                        pending.feature_events.append((feature_index,event))
                    pending.cursor+=1;offset=0
                pending.offset=offset
            if pending.cursor==len(pending.plan) and pending.prepared.complete and pending.completion is None:
                pending.completion=torch.cuda.Event();pending.completion.record(self._upload_stream)
        pending.stages['upload_owner_cpu_s']+=time.perf_counter()-started
        finished=pending.completion is not None and pending.completion.query()
        if finished:pending.available_features=len(pending.prepared.features)
        if pending.handle is None and pending.available_features>=pending.model.time_batch:
            p=pending.prepared
            pending.handle=ReadyGopHandle(pending.model,p.reconstruction,p.cameras,p.nbytes,self.device,pending.stages,available_features=pending.available_features,transfer=weakref.ref(pending))
            pending.stages['first_window_gpu_ready_s']=time.perf_counter()-pending.started
        if not finished:return pending.handle
        pending.done=True
        for slot in self._upload_buffers:
            if slot.pending is pending:
                pending.stages['h2d_copy_gpu_s']+=slot.start.elapsed_time(slot.event)/1000
                slot.pending=None;slot.busy=False
        stages=pending.stages
        stages.update(pending.prepared.stages)
        stages['upload_pipeline_elapsed_s']=time.perf_counter()-pending.started
        stages['realize_gpu_s']=time.perf_counter()-pending.started
        pending.handle.available_features=len(pending.prepared.features)
        return pending.handle

    def _refresh_transfers(self,pending):
        while pending.feature_events and pending.feature_events[0][1].query():
            index,_=pending.feature_events.popleft();pending.available_features=index+1
        if pending.handle is not None:pending.handle.available_features=pending.available_features

    def frame_ready(self,handle,model_time):
        if handle.transfer is not None:
            pending=handle.transfer()
            if pending is not None:self._refresh_transfers(pending)
        available=handle.available_features if handle.available_features is not None else len(handle.model._point_feats)
        return int(float(model_time)*handle.model.time_line)+handle.model.time_batch<=available

    def cancel_gpu(self,pending):
        self._assert_owner()
        self._upload_stream.synchronize()
        for slot in self._upload_buffers:
            if slot.pending is pending:slot.pending=None;slot.busy=False
        pending.model=None

    def reconstruct(self,handle,model_time,camera):
        import torch
        self._assert_owner()
        if handle.released:raise RuntimeError('GOP handle has been released')
        pc=handle.model;t=float(model_time);index=int(t*pc.time_line)
        if not np.isfinite(t) or t<0 or index+pc.time_batch>len(pc._point_feats):raise ValueError('Model time lies outside available temporal window')
        if not self.frame_ready(handle,t):raise RuntimeError('Temporal feature transfer is not complete for this frame')
        with torch.cuda.device(handle.device),torch.inference_mode():
            if hasattr(camera,'to_torch'):center=camera.to_torch(handle.device).camera_center
            elif hasattr(camera,'center'):center=camera.center
            elif hasattr(camera,'camera_center'):center=camera.camera_center
            else:center=camera
            center=torch.as_tensor(center,device=handle.device,dtype=torch.float32)
            p=torch.ones((len(pc._xyz),1),device=handle.device)*t
            feature=torch.cat([pc.get_time_features(t,False,noise=0),pc.get_time_pe(p)],1)
            feature=pc.get_time_mlp(feature)
            delta,cov=pc.get_deform_mlp(feature),pc.get_cov_mlp(feature)
            xyz=pc.get_xyz+delta;scales=pc.scaling_activation(pc._scaling+cov[:,:3]);rotations=pc.rotation_activation(pc._rotation+cov[:,3:])
            direction=pc.get_xyz-center;distance=direction.norm(dim=1,keepdim=True)
            conditioned=torch.cat([feature,direction/distance,distance],1)
            opacity=pc.opacity_activation(pc._opacity+pc.get_opacity_mlp(conditioned))
            color=torch.relu(pc._features_dc)+pc.get_color_mlp(conditioned)[:,:3]
            return GaussianTensorState(xyz,scales,rotations,opacity,color,f'streamstgs:t={t}')

    def release(self,handle):
        import torch
        self._assert_owner()
        if not handle.released:
            # GL completion is the renderer owner's responsibility before retirement.
            torch.cuda.current_stream(handle.device).synchronize();handle.model=None;handle.released=True
