"""Initial scene framing using the existing calibrated camera's field of view."""
import numpy as np


def fit_scene_camera(controller,catalog):
    """Extend single-content calibrated framing to arbitrary instance placement.

    Preserve the source camera's per-object framing allowance and add projected
    center offsets using actual intrinsics. The old fit's distance-only extent
    ignores narrow FOVs and can clip the outer instances of a larger scene.
    This is an initialization policy; normal orbit/pan/seek controls are reused.
    """
    instances=catalog.instances
    centers=np.asarray([np.asarray(i.translation)+np.asarray(catalog.contents[i.content_id].pivot) for i in instances])
    allowance=controller.radius
    controller.fit(centers,[i.scale for i in instances])
    camera=controller.camera()
    offsets=(camera.world_to_camera[:3,:3]@(centers-controller.target).T).T
    tanx=max(1e-6,min(camera.K[0,2]+.5,camera.width-camera.K[0,2]-.5)/camera.K[0,0])
    tany=max(1e-6,min(camera.K[1,2]+.5,camera.height-camera.K[1,2]-.5)/camera.K[1,1])
    radius=max(allowance*i.scale+max(abs(p[0])/tanx,abs(p[1])/tany)-p[2] for i,p in zip(instances,offsets))
    controller.radius=max(controller.radius,radius*1.05)
    controller._invalidate()
