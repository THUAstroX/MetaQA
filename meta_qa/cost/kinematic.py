"""Kinematic cost interface with vehicle dynamics limits."""

from dataclasses import dataclass
import numpy as np

from .base import CostFunction, CostState, CostResult


@dataclass
class KinematicLimits:
    """Vehicle kinematic limits."""
    max_yaw_rate: float = 0.5         # rad/s
    max_sideslip: float = 0.1         # rad (~6 deg)
    max_lateral_accel: float = 4.0    # m/s^2
    
    yaw_rate_weight: float = 0.5
    sideslip_weight: float = 0.5


class KinematicCost(CostFunction):
    """Kinematic cost based on yaw rate and sideslip angle."""
    
    def __init__(self, limits: KinematicLimits = None):
        super().__init__()
        self.limits = limits or KinematicLimits()
    
    def compute(self, state: CostState) -> CostResult:
        """Compute cost from yaw rate and sideslip violations."""
        ego = state.ego
        
        yaw_rate_ratio = abs(ego.yaw_rate) / self.limits.max_yaw_rate
        sideslip_ratio = abs(ego.sideslip_angle) / self.limits.max_sideslip
        
        # Weighted sum, clipped to [0, 1]
        cost = (
            self.limits.yaw_rate_weight * min(1.0, yaw_rate_ratio) +
            self.limits.sideslip_weight * min(1.0, sideslip_ratio)
        ) / (self.limits.yaw_rate_weight + self.limits.sideslip_weight)
        
        return CostResult(
            value=cost,
            info={
                "yaw_rate": ego.yaw_rate,
                "sideslip": ego.sideslip_angle,
                "lateral_accel": ego.lateral_acceleration,
            }
        )
