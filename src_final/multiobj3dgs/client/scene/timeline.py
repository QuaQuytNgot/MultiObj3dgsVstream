

def media_time(instance, content, scene_time):
    """Return media seconds, or None outside the instance's active interval."""
    elapsed = scene_time - instance.scene_start
    if elapsed < 0 or (instance.active_duration is not None and elapsed >= instance.active_duration):
        return None
    value = instance.media_offset + elapsed
    if instance.loop:
        return value % content.duration
    return value if 0 <= value < content.duration else None


def transform_matrix(instance, pivot=(0,0,0)):
    """Stable pivot is content metadata, never recomputed from a GOP mean."""
    from ...shared.geometry import similarity_matrix
    return similarity_matrix(instance.translation,instance.rotation,instance.scale,pivot)
