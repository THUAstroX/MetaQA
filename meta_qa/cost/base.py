"""
Cost Function Base Interface.

Provides base classes and vehicle state data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np


@dataclass
class EgoState:
    """Ego vehicle kinematic state."""
    
    position: np.ndarray       # (2,) [x, y] meters
    velocity: np.ndarray       # (2,) [vx, vy] m/s
    heading: float             # radians
    speed: float               # m/s (scalar)
    
    acceleration: float = 0.0       # m/s^2
    yaw_rate: float = 0.0           # rad/s
    steering_angle: float = 0.0     # radians
    
    length: float = 4.5
    width: float = 1.8
    
    @property
    def sideslip_angle(self) -> float:
        """Sideslip angle in vehicle frame."""
        vx = self.velocity[0] * np.cos(self.heading) + self.velocity[1] * np.sin(self.heading)
        vy = -self.velocity[0] * np.sin(self.heading) + self.velocity[1] * np.cos(self.heading)
        if abs(vx) > 0.1:
            return np.arctan2(vy, abs(vx))
        return 0.0
    
    @property
    def lateral_acceleration(self) -> float:
        """Lateral acceleration: ay = v * yaw_rate."""
        return self.speed * abs(self.yaw_rate)
    
    @classmethod
    def from_env(cls, env) -> "EgoState":
        """Create from MetaDrive environment."""
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        
        ego = base_env.agent
        velocity = np.array([ego.velocity[0], ego.velocity[1]])
        
        return cls(
            position=np.array([ego.position[0], ego.position[1]]),
            velocity=velocity,
            heading=ego.heading_theta,
            speed=np.linalg.norm(velocity),
        )


@dataclass
class SurroundingVehicle:
    """Single surrounding vehicle."""
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    distance: float = 0.0
    length: float = 4.5
    width: float = 1.8


@dataclass
class CostState:
    """State container for cost computation."""
    ego: EgoState
    surrounding: List[SurroundingVehicle] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostResult:
    """Result of cost computation."""
    value: float
    info: Dict[str, Any] = field(default_factory=dict)


class CostFunction(ABC):
    """Abstract base class for cost functions."""
    
    def __init__(self):
        self.name = self.__class__.__name__
    
    @abstractmethod
    def compute(self, state: CostState) -> CostResult:
        """Compute cost from state. Returns CostResult with value in [0, 1]."""
        pass
    
    def __call__(self, state: CostState) -> CostResult:
        return self.compute(state)
