import math


def _roundoff_bound(*values):
    # Bound a few binary64 subtraction/addition/division rounding steps. This
    # scales with the actual operands, rather than using a scene-time epsilon.
    return 4*max(math.ulp(value) for value in values)


def _snap_frame_roundoff(value,fps,tolerance):
    canonical=round(value*fps)/fps
    return canonical if abs(value-canonical)<=tolerance else value


def media_time(instance, content, scene_time):
    """Return media seconds, or None outside the instance's active interval."""
    elapsed = scene_time - instance.scene_start
    tolerance=_roundoff_bound(scene_time,instance.scene_start,elapsed)
    if abs(elapsed)<=tolerance:elapsed=0.
    if instance.active_duration is not None:
        if abs(elapsed-instance.active_duration)<=max(tolerance,4*math.ulp(instance.active_duration)):
            elapsed=instance.active_duration
    if elapsed < 0 or (instance.active_duration is not None and elapsed >= instance.active_duration):
        return None
    value = instance.media_offset + elapsed
    tolerance=_roundoff_bound(scene_time,instance.scene_start,instance.media_offset,value)
    value=_snap_frame_roundoff(value,content.fps,tolerance)
    if instance.loop:
        # No subtraction occurs inside the first content interval. Including
        # duration's much larger ULP here could erase a real fractional phase.
        if 0<=value<content.duration:return value
        wrapped=value%content.duration
        wrapped=_snap_frame_roundoff(wrapped,content.fps,
            _roundoff_bound(value,content.duration,wrapped))
        # A remainder just below duration can round to the exclusive endpoint.
        return 0. if wrapped==content.duration else wrapped
    return value if 0 <= value < content.duration else None


def transform_matrix(instance, pivot=(0,0,0)):
    """Stable pivot is content metadata, never recomputed from a GOP mean."""
    from ...shared.geometry import similarity_matrix
    return similarity_matrix(instance.translation,instance.rotation,instance.scale,pivot)
