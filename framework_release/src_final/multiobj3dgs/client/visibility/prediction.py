"""Explicit current-pose hold baseline; no trained camera predictor is implied."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class PosePrediction:
    camera: object
    observed_at: float
    target_at: float
    evidence: str = 'current_pose_hold'

class CurrentPoseHold:
    def predict(self,camera,observed_at,target_at):
        if not all(math.isfinite(t) for t in (observed_at,target_at)) or target_at<observed_at:
            raise ValueError('Invalid prediction horizon')
        return PosePrediction(camera,observed_at,target_at)
