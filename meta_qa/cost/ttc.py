"""Time-to-Collision cost interface."""

from typing import Optional
import numpy as np

from .base import CostFunction, CostState, CostResult, SurroundingVehicle


def compute_ttc(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    other_pos: np.ndarray,
    other_vel: np.ndarray,
    collision_radius: float = 3.0,
) -> float:
    """
    Compute Time-to-Collision between two vehicles.
    
    Returns TTC in seconds (inf if no collision predicted).
    """
    rel_pos = other_pos - ego_pos
    rel_vel = other_vel - ego_vel
    
    distance = np.linalg.norm(rel_pos)
    if distance < 1e-6:
        return 0.0
    
    closing_speed = -np.dot(rel_vel, rel_pos) / distance
    
    if closing_speed <= 0:
        return float("inf")
    
    return max(0.0, (distance - collision_radius) / closing_speed)


class TTCCost(CostFunction):
    """TTC-based cost interface."""
    
    def __init__(self, threshold: float = 3.0):
        super().__init__()
        self.threshold = threshold
    
    def compute(self, state: CostState) -> CostResult:
        """Compute cost based on minimum TTC."""
        if not state.surrounding:
            return CostResult(value=0.0, info={"min_ttc": float("inf")})
        
        min_ttc = float("inf")
        for other in state.surrounding:
            ttc = compute_ttc(
                state.ego.position, state.ego.velocity,
                other.position, other.velocity
            )
            min_ttc = min(min_ttc, ttc)
        
        # Linear interpolation: ttc=0 -> cost=1, ttc>=threshold -> cost=0
        cost = max(0.0, 1.0 - min_ttc / self.threshold) if min_ttc < float("inf") else 0.0
        
        return CostResult(value=cost, info={"min_ttc": min_ttc})
