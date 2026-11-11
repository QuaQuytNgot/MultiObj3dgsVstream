"""Production StreamSTGS reuse plus a separate CRF40 contribution-only proxy.

No renderer is instantiated by import, planning, freezing, allocation, or tests.
No opacities/geometry are manipulated. RGB-basis passes preserve native Gaussian
sorting and transmittance. Their object aggregate ratio is an effective
attenuation proxy, not a single physical layer for every Gaussian of an object.
"""
import math,time
from pathlib import Path
import numpy as np
from experiments.nava_mse.rendering import StreamSTGSRenderer,_GPUBackend,validate_render_job
from experiments.nava_mse.workflow import validate_frozen_image
from experiments.nava_mse.common import digest,PROJECT


class ContributionBackend(_GPUBackend):
    mode='rgb'
    last_proxy=None
    def render(self,states,camera):
        if self.mode=='rgb':return super().render(states,camera)
        if self.mode!='proxy':raise ValueError('Unknown rendering purpose')
        torch=self.torch;start=time.perf_counter()
        with torch.inference_mode():
            scene=self.scene_type(states,camera)
            contributions=[]
            for i in range(0,len(states),3):
                contributions.extend(scene.render_joint(i).detach().cpu().numpy())
            solo=np.stack([scene.render_solo(i).detach().cpu().numpy() for i in range(len(states))])
            args=scene.joint.raster_args();args['colors_precomp']=torch.ones_like(scene.joint.colors)
            joint=scene.rasterizer(**args)[0][0].detach().cpu().numpy()
        self.last_proxy=dict(solo_alpha=solo,contributions=np.stack(contributions),joint_alpha=np.clip(joint,0,1))
        self.last_proxy_timing=dict(visibility_analysis_s=time.perf_counter()-start,
                                   contribution_passes=math.ceil(len(states)/3),solo_alpha_passes=len(states),joint_alpha_passes=1,
                                   method='native_joint_Gaussian_RGB_basis_all_CRF40')
        # Target joint RGB is deliberately not produced by this stage.
        return np.zeros((camera.height,camera.width,3),np.float32),joint


def package_paths(frozen):
    c=frozen['config'];base=PROJECT/c['packages_root']
    return {(oid,c['gop'],q):base/oid/f'gop_{c["gop"]:03d}/encoded/crf_{q}' for oid in c['source_objects'] for q in c['crfs']}


def make_renderer(frozen,*,purpose='rgb'):
    c=frozen['config'];runtime=c['runtime']
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no rendering performed')
    memory_mb=runtime['max_gpu_memory_mb'];total=torch.cuda.get_device_properties(0).total_memory
    if memory_mb<=0:raise ValueError('Positive VRAM limit required')
    torch.cuda.set_per_process_memory_fraction(min(1.,memory_mb*1024**2/total),0)
    backend=ContributionBackend(device='cuda',ffmpeg_threads=runtime.get('ffmpeg_threads',1));backend.mode=purpose
    return StreamSTGSRenderer(package_paths(frozen),metadata_root=PROJECT/c['metadata_root'],max_resident_models=runtime['max_resident_models'],bounds_source=PROJECT/c['bounds_source'],backend=backend)


def image_job(scene,timestamp,allocation,*,instance_id=None):
    chosen=[i for i in scene['instances'] if instance_id is None or i['instance_id']==instance_id]
    if not chosen:raise ValueError('Unknown instance')
    geometry=[{k:i[k] for k in ('instance_id','object_id','object_to_world','source_camera','conditioning_mode')} for i in chosen]
    return dict(kind='object' if instance_id else 'joint',camera_id=scene['camera_id'],timestamp_s=timestamp,allocation=dict(allocation),object_id=chosen[0]['object_id'] if instance_id else None,
                instance_id=instance_id,context_sha256=digest(dict(camera=scene['camera'],instances=geometry)))


def verify_image(frozen,scene,job,image):
    selected=[i for i in scene['instances'] if job['instance_id'] is None or i['instance_id']==job['instance_id']]
    validate_render_job(image['metadata'],job,frozen['config']['gop'],expected_camera=scene['camera'],expected_instances=selected)
    validate_frozen_image(frozen,image)


def render_image(frozen,scene,timestamp,allocation,store,renderer=None,*,instance_id=None):
    from .scenes import camera_from_metadata
    job=image_job(scene,timestamp,allocation,instance_id=instance_id)
    found=store.load(job)
    if found is not None:verify_image(frozen,scene,job,found);return found,False
    if renderer is None:return None,False
    instances=[i for i in scene['instances'] if instance_id is None or i['instance_id']==instance_id]
    started=time.perf_counter()
    result=renderer.render(camera_from_metadata(scene['camera']),instances,timestamp,frozen['config']['gop'],allocation)
    result.metadata['execution']={'render_and_decode_wall_s':time.perf_counter()-started,'purpose':'exact_offline_solo' if instance_id else 'independent_joint_evaluation'}
    backend=getattr(renderer,'_backend',None)
    if hasattr(backend,'torch'):result.metadata['execution']['peak_gpu_allocated_bytes']=int(backend.torch.cuda.max_memory_allocated())
    verify_image(frozen,scene,job,dict(metadata=result.metadata,rgb=result.rgb,raw_rgb=result.raw_rgb,alpha=result.alpha))
    saved=store.save(job,result)
    if frozen['config'].get('save_png',False):
        from experiments.nava_mse.storage import save_preview
        save_preview(Path(saved['provenance']['arrays_path']).with_suffix('.png'),saved['rgb'])
    return saved,True


def proxy_context(scene,timestamp):
    return dict(scene_geometry=digest(dict(camera=scene['camera'],instances=[{k:i[k] for k in ('instance_id','object_id','object_to_world','source_camera','conditioning_mode')} for i in scene['instances']])),camera_id=scene['camera_id'],timestamp_s=timestamp,quality=40,purpose='LOW_COST_OCCLUSION_PROXY',object_ids=[i['instance_id'] for i in scene['instances']])


def render_proxy(frozen,scene,timestamp,store,renderer=None):
    from .scenes import camera_from_metadata
    context=proxy_context(scene,timestamp);found=store.load('proxy',context)
    if found is not None:
        if found['metadata']['quality']!=40 or found['metadata']['purpose']!='LOW_COST_OCCLUSION_PROXY':raise ValueError('Cached proxy provenance mismatch')
        allocation={i['instance_id']:40 for i in scene['instances']}
        verify_image(frozen,scene,image_job(scene,timestamp,allocation),dict(metadata=found['metadata']['render_metadata']))
        return found,False
    if renderer is None:return None,False
    if renderer._backend.mode!='proxy':raise ValueError('Proxy requires contribution-only backend')
    allocation={i['instance_id']:40 for i in scene['instances']}
    start=time.perf_counter();result=renderer.render(camera_from_metadata(scene['camera']),scene['instances'],timestamp,frozen['config']['gop'],allocation)
    job=image_job(scene,timestamp,allocation)
    verify_image(frozen,scene,job,dict(metadata=result.metadata,rgb=result.rgb,raw_rgb=result.raw_rgb,alpha=result.alpha))
    timing=dict(renderer._backend.last_proxy_timing)
    if hasattr(renderer._backend,'torch'):timing['peak_gpu_allocated_bytes']=int(renderer._backend.torch.cuda.max_memory_allocated())
    return store.save('proxy',context,renderer._backend.last_proxy,dict(render_metadata=result.metadata,quality=40,purpose='LOW_COST_OCCLUSION_PROXY',preparation_wall_s=time.perf_counter()-start,**timing)),True
