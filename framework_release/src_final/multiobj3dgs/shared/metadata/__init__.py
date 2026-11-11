"""Immutable independent-GOP catalog. No codec, numerical or transport imports."""
from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping
import math


def _finite(value, name, *, minimum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if (minimum is not None and value < minimum) or (positive and value <= 0):
        raise ValueError(f'{name} is out of range')


def _identifier(value):
    if not isinstance(value, str) or not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-:.' for c in value):
        raise ValueError(f'Invalid identifier: {value!r}')


@dataclass(frozen=True, slots=True)
class AssetRef:
    id: str
    url: str
    bytes: int
    sha256: str
    def __post_init__(self):
        _identifier(self.id)
        if not isinstance(self.url, str) or not self.url:
            raise ValueError('Asset URL is required')
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 0:
            raise ValueError('Invalid asset bytes')
        if len(self.sha256) != 64 or any(c not in '0123456789abcdef' for c in self.sha256):
            raise ValueError('Invalid SHA256')


@dataclass(frozen=True, slots=True)
class FrameMapping:
    source_frame: int
    seconds: float
    model_time: float
    def __post_init__(self):
        if isinstance(self.source_frame, bool) or not isinstance(self.source_frame, int):
            raise ValueError('source_frame must be integer')
        _finite(self.seconds, 'seconds', minimum=0)
        _finite(self.model_time, 'model_time', minimum=0)


@dataclass(frozen=True, slots=True)
class GopSegment:
    id: str
    gop_index: int
    start: float
    duration: float
    resource: AssetRef
    members: Mapping[str, str]
    shared_assets: tuple[AssetRef, ...]
    frames: tuple[FrameMapping, ...]
    codec_metadata: str = 'reconstruction.json'
    def __post_init__(self):
        _identifier(self.id)
        if isinstance(self.gop_index, bool) or not isinstance(self.gop_index, int) or self.gop_index < 0:
            raise ValueError('Invalid GOP index')
        _finite(self.start, 'GOP start', minimum=0)
        _finite(self.duration, 'GOP duration', positive=True)
        object.__setattr__(self, 'members', MappingProxyType(dict(self.members)))
        object.__setattr__(self, 'shared_assets', tuple(self.shared_assets))
        object.__setattr__(self, 'frames', tuple(self.frames))
        if not self.frames or self.frames[0].seconds != self.start:
            raise ValueError('GOP requires explicit output-frame mapping starting at GOP start')
        if any(b.seconds <= a.seconds or b.source_frame != a.source_frame + 1 for a, b in zip(self.frames, self.frames[1:])):
            raise ValueError('GOP frame mapping must be contiguous')
        if self.frames[-1].seconds >= self.end:
            raise ValueError('Frame lies beyond exclusive GOP end')
        shared_ids = {a.id for a in self.shared_assets}
        if len(shared_ids) != len(self.shared_assets) or {v for v in self.members.values() if v} != shared_ids:
            raise ValueError('Unknown or duplicate shared asset')
        if self.codec_metadata not in self.members:
            raise ValueError('Codec metadata must be materialized')
        for name in self.members:
            if not name or name in {'.', '..'} or '/' in name or '\\' in name:
                raise ValueError('Codec member must be a basename')
    @property
    def end(self):
        return self.start + self.duration
    @property
    def resources(self):
        return (self.resource, *self.shared_assets)
    @property
    def cold_bytes(self):
        return sum(a.bytes for a in {a.sha256: a for a in self.resources}.values())
    def frame_at(self, seconds: float):
        if not self.start <= seconds < self.end:
            raise ValueError('Time outside GOP')
        from bisect import bisect_right
        return self.frames[bisect_right([f.seconds for f in self.frames], seconds) - 1]


@dataclass(frozen=True, slots=True)
class RepresentationInfo:
    id: str
    crf: int
    rank: int
    bandwidth: int
    segments: tuple[GopSegment, ...]
    def __post_init__(self):
        _identifier(self.id)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (self.crf, self.rank, self.bandwidth)) or self.bandwidth == 0:
            raise ValueError('Invalid representation metadata')
        object.__setattr__(self, 'segments', tuple(self.segments))
        if not self.segments or self.segments[0].start != 0:
            raise ValueError('Representation timeline must start at zero')
        if [s.gop_index for s in self.segments] != list(range(len(self.segments))):
            raise ValueError('GOP indices must be contiguous')
        if any(not math.isclose(a.end, b.start, abs_tol=1e-8) for a,b in zip(self.segments,self.segments[1:])):
            raise ValueError('GOP timeline has gap/overlap')


@dataclass(frozen=True, slots=True)
class ObjectContent:
    id: str
    fps: float
    duration: float
    representations: Mapping[str, RepresentationInfo]
    pivot: tuple[float, float, float] = (0., 0., 0.)
    codec: str = 'streamstgs'
    def __post_init__(self):
        _identifier(self.id)
        _finite(self.fps, 'fps', positive=True)
        _finite(self.duration, 'content duration', positive=True)
        object.__setattr__(self, 'pivot', tuple(self.pivot))
        object.__setattr__(self, 'representations', MappingProxyType(dict(self.representations)))
        if len(self.pivot) != 3:
            raise ValueError('pivot must contain three coordinates')
        for v in self.pivot: _finite(v, 'pivot')
        if self.codec != 'streamstgs' or not self.representations:
            raise ValueError('Unsupported codec or empty representations')
        baseline = None
        for key, rep in self.representations.items():
            if key != rep.id or not math.isclose(rep.segments[-1].end, self.duration, abs_tol=1e-8):
                raise ValueError('Representation identity/duration mismatch')
            mapping = tuple((s.start, s.duration, tuple((f.source_frame, f.seconds, f.model_time) for f in s.frames)) for s in rep.segments)
            if baseline is not None and baseline != mapping:
                raise ValueError('Representations must have identical presentation/model-time mappings')
            baseline = mapping
            frames = [f for s in rep.segments for f in s.frames]
            if any(b.source_frame != a.source_frame+1 for a,b in zip(frames,frames[1:])):
                raise ValueError('Source frame mapping must remain contiguous across GOP boundaries')
            if any(not math.isclose(f.seconds, i / self.fps, abs_tol=1e-8) for i,f in enumerate(frames)) or not math.isclose(len(frames)/self.fps,self.duration,abs_tol=1e-8):
                raise ValueError('Frame coverage does not match fps/duration')
    def segment_at(self, representation_id: str, seconds: float):
        if not 0 <= seconds < self.duration:
            raise ValueError('Time outside content')
        return next(s for s in self.representations[representation_id].segments if s.start <= seconds < s.end)


@dataclass(frozen=True, slots=True)
class SceneInstance:
    id: str
    content_id: str
    translation: tuple[float, float, float] = (0., 0., 0.)
    rotation: tuple[float, float, float, float] = (1., 0., 0., 0.)
    scale: float = 1.
    scene_start: float = 0.
    media_offset: float = 0.
    active_duration: float | None = None
    loop: bool = False
    def __post_init__(self):
        _identifier(self.id); _identifier(self.content_id)
        object.__setattr__(self, 'translation', tuple(self.translation))
        object.__setattr__(self, 'rotation', tuple(self.rotation))
        if len(self.translation) != 3 or len(self.rotation) != 4:
            raise ValueError('Expected XYZ translation and WXYZ rotation')
        for v in (*self.translation, *self.rotation): _finite(v, 'transform')
        if not math.isclose(sum(v*v for v in self.rotation), 1., abs_tol=1e-5):
            raise ValueError('Rotation quaternion must be normalized')
        _finite(self.scale, 'scale', positive=True)
        _finite(self.scene_start, 'scene_start', minimum=0)
        _finite(self.media_offset, 'media_offset', minimum=0)
        if self.active_duration is not None: _finite(self.active_duration, 'active_duration', positive=True)
        if not isinstance(self.loop, bool): raise ValueError('loop must be boolean')


@dataclass(frozen=True, slots=True)
class ContentCatalog:
    revision: str
    contents: Mapping[str, ObjectContent]
    instances: tuple[SceneInstance, ...]
    duration: float
    base_url: str = ''
    def __post_init__(self):
        if len(self.revision) != 64 or any(c not in '0123456789abcdef' for c in self.revision):
            raise ValueError('Invalid catalog revision')
        object.__setattr__(self, 'contents', MappingProxyType(dict(self.contents)))
        object.__setattr__(self, 'instances', tuple(self.instances))
        _finite(self.duration, 'scene duration', positive=True)
        if not self.contents or not self.instances or len({i.id for i in self.instances}) != len(self.instances):
            raise ValueError('Nonempty contents and unique instances required')
        for key, content in self.contents.items():
            if key != content.id: raise ValueError('Content ID mismatch')
        for instance in self.instances:
            if instance.content_id not in self.contents: raise ValueError('Unknown instance content')
            content = self.contents[instance.content_id]
            if instance.media_offset >= content.duration: raise ValueError('Media offset outside content')
            available = content.duration-instance.media_offset
            active = instance.active_duration if instance.active_duration is not None else available
            if not instance.loop and active > available + 1e-8: raise ValueError('Nonlooping instance exceeds content')
            if instance.scene_start + active > self.duration + 1e-8: raise ValueError('Instance exceeds scene timeline')
