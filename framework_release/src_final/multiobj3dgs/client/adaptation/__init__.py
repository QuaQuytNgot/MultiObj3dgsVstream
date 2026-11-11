"""Whole-object/GOP selection contracts; no optimization solver is installed."""
from .interfaces import Candidate, PlannerInput, Selection, SelectionPlan, ManualSelector

__all__ = ['Candidate', 'PlannerInput', 'Selection', 'SelectionPlan', 'ManualSelector']
