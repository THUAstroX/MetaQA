"""Collision cost interface."""

from .base import CostFunction, CostState, CostResult


class CollisionCost(CostFunction):
    """Binary collision cost from environment info."""
    
    def compute(self, state: CostState) -> CostResult:
        """Returns 1.0 if crash occurred, 0.0 otherwise."""
        crash = state.info.get("crash", False) or state.info.get("crash_vehicle", False)
        return CostResult(value=1.0 if crash else 0.0)
