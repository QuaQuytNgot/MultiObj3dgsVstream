"""Current per-object playback states, independent of decoder and renderer.

The scheduler/buffer selects frames. This manager publishes the selected states,
checks synchronization, and preserves other objects across a quality update.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from src.renderer.backend import (Camera, RenderObject, RendererBackend,
                                  finite_timestamp, validate_transform)
from tools.content_preparation.assets import GaussianState


@dataclass(slots=True)
class SceneObject:
    object_id: str
    object_to_world: np.ndarray = field(default_factory=lambda: np.eye(4))
    gaussian_state: GaussianState | None = None
    timestamp: float | None = None
    quality: int = 0
    frame_index: int | None = None
    playing: bool = True
    visible: bool = True
    revision: int = 0


class SceneManager:
    def __init__(self):
        self.objects: dict[str, SceneObject] = {}

    def add_object(self, object_id: str, object_to_world=None, *, playing=True,
                   visible=True) -> SceneObject:
        if not isinstance(object_id, str) or not object_id.strip():
            raise ValueError("object_id must be a nonempty string")
        if object_id in self.objects:
            raise ValueError(f"Duplicate object_id: {object_id}")
        transform = validate_transform(np.eye(4) if object_to_world is None else object_to_world)
        obj = SceneObject(object_id, transform, playing=bool(playing), visible=bool(visible))
        self.objects[object_id] = obj
        return obj

    def update_object(self, object_id: str, gaussian_state: GaussianState,
                      timestamp: float, *, quality=0, frame_index=None,
                      allow_rewind=False) -> SceneObject:
        obj = self.objects[object_id]
        timestamp = finite_timestamp(timestamp)
        if not isinstance(gaussian_state, GaussianState):
            raise TypeError("Scene updates require a decoded GaussianState")
        if isinstance(quality, bool) or int(quality) != quality or not 0 <= quality <= 3:
            raise ValueError("quality must be Q0..Q3, expressed as an integer 0..3")
        if obj.timestamp is not None and timestamp < obj.timestamp - 1e-9 and not allow_rewind:
            raise ValueError("Out-of-order object timestamp; use allow_rewind for explicit seeks")
        # Quality changes at the same timestamp are valid, including on pause.
        if obj.gaussian_state is not gaussian_state:
            gaussian_state.validate()
            obj.gaussian_state = gaussian_state
            obj.revision += 1
        obj.timestamp, obj.quality, obj.frame_index = timestamp, int(quality), frame_index
        return obj

    def set_transform(self, object_id: str, object_to_world):
        obj = self.objects[object_id]
        transform = validate_transform(object_to_world)
        if not np.array_equal(transform, obj.object_to_world):
            obj.object_to_world = transform
            obj.revision += 1

    def set_playing(self, object_id: str, playing: bool):
        self.objects[object_id].playing = bool(playing)

    def remove_object(self, object_id: str):
        del self.objects[object_id]

    def snapshot(self, timestamp: float, *, require_synchronized=True) -> tuple[RenderObject, ...]:
        timestamp = finite_timestamp(timestamp)
        result = []
        for obj in self.objects.values():
            if not obj.visible or obj.gaussian_state is None:
                continue
            if require_synchronized and obj.playing and abs(obj.timestamp - timestamp) > 1e-9:
                raise ValueError(f"{obj.object_id}: frame timestamp {obj.timestamp} differs from scene timestamp {timestamp}")
            result.append(RenderObject(obj.object_id, obj.gaussian_state, obj.timestamp,
                                       obj.object_to_world, obj.revision))
        return tuple(result)

    def render(self, backend: RendererBackend, camera: Camera, timestamp: float,
               *, require_synchronized=True):
        return backend.render(self.snapshot(timestamp, require_synchronized=require_synchronized),
                              camera, timestamp)

    def close(self):
        self.objects.clear()
