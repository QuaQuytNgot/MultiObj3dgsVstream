from dataclasses import dataclass, field
from typing import Mapping, Protocol
from types import MappingProxyType


@dataclass(frozen=True)
class Candidate:
    instance_id: str
    gop_id: str
    representation_id: str
    missing_bytes: int
    duration: float
    deadline: float
    utility: float | None = None


@dataclass(frozen=True)
class PlannerInput:
    revision: str
    candidates: tuple[Candidate, ...]
    predicted_bandwidth_bps: float | None
    visibility: Mapping[str, float] = field(default_factory=dict)
    buffer_seconds: Mapping[str, float] = field(default_factory=dict)
    observations: Mapping[str, object] = field(default_factory=dict)


    def __post_init__(self):
        object.__setattr__(self, 'candidates', tuple(self.candidates))
        for name in ('visibility','buffer_seconds','observations'):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


@dataclass(frozen=True)
class Selection:
    instance_id: str
    gop_id: str
    representation_id: str
    deadline: float


@dataclass(frozen=True)
class SelectionPlan:
    revision: str
    selections: tuple[Selection, ...]
    reason: str = 'manual'


class Planner(Protocol):
    def select(self, inputs: PlannerInput) -> SelectionPlan: ...


class ManualSelector:
    """Opaque representation identities; selections do not infer utility from CRF."""
    def __init__(self, catalog, defaults=None):
        self.catalog = catalog
        self.selected = {}
        defaults = defaults or {}
        for instance in catalog.instances:
            representations = catalog.contents[instance.content_id].representations
            self.set(instance.id, defaults.get(instance.id, next(iter(representations))))

    def set(self, instance_id, representation_id):
        instance = next((i for i in self.catalog.instances if i.id == instance_id), None)
        if instance is None:
            raise KeyError(instance_id)
        if representation_id not in self.catalog.contents[instance.content_id].representations:
            raise KeyError(f'unknown representation {representation_id!r} for {instance_id}')
        self.selected[instance_id] = representation_id

    def select(self, inputs):
        selected = tuple(Selection(c.instance_id, c.gop_id, c.representation_id, c.deadline)
                         for c in inputs.candidates if self.selected[c.instance_id] == c.representation_id)
        return SelectionPlan(inputs.revision, selected)
