"""
Data Structures for Offline RL.

Defines MDP and CMDP transition tuples and supporting data classes.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class SurroundingVehicleInfo:
    """
    Information about surrounding vehicles.
    
    Stores both world-frame and ego-centric information for flexibility.
    
    Attributes:
        num_vehicles: Number of surrounding vehicles
        positions: World frame positions (N, 2)
        velocities: World frame velocities (N, 2)
        headings: Vehicle headings in radians (N,)
        relative_positions: Ego-centric positions (N, 2)
        relative_velocities: Ego-centric velocities (N, 2)
        distances: Distances to ego vehicle (N,)
        lengths: Vehicle lengths (N,)
        widths: Vehicle widths (N,)
        vehicle_types: Vehicle type strings
    """
    num_vehicles: int = 0
    
    # World frame information
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    headings: np.ndarray = field(default_factory=lambda: np.zeros(0))
    
    # Ego-centric (local frame) information
    relative_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    relative_velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    distances: np.ndarray = field(default_factory=lambda: np.zeros(0))
    
    # Additional attributes
    lengths: np.ndarray = field(default_factory=lambda: np.zeros(0))
    widths: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vehicle_types: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_vehicles": self.num_vehicles,
            "positions": self.positions.tolist(),
            "velocities": self.velocities.tolist(),
            "headings": self.headings.tolist(),
            "relative_positions": self.relative_positions.tolist(),
            "relative_velocities": self.relative_velocities.tolist(),
            "distances": self.distances.tolist(),
            "lengths": self.lengths.tolist(),
            "widths": self.widths.tolist(),
            "vehicle_types": self.vehicle_types,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SurroundingVehicleInfo":
        """Create from dictionary."""
        return cls(
            num_vehicles=data["num_vehicles"],
            positions=np.array(data["positions"]),
            velocities=np.array(data["velocities"]),
            headings=np.array(data["headings"]),
            relative_positions=np.array(data["relative_positions"]),
            relative_velocities=np.array(data["relative_velocities"]),
            distances=np.array(data["distances"]),
            lengths=np.array(data.get("lengths", [])),
            widths=np.array(data.get("widths", [])),
            vehicle_types=data.get("vehicle_types", []),
        )
    
    def to_flat_array(self, max_vehicles: int = 10, per_vehicle_dim: int = 6) -> np.ndarray:
        """
        Convert to flat array for observation.
        
        Args:
            max_vehicles: Maximum number of vehicles to include
            per_vehicle_dim: Dimensions per vehicle
            
        Returns:
            Flat array of shape (max_vehicles * per_vehicle_dim,)
        """
        arr = np.zeros(max_vehicles * per_vehicle_dim, dtype=np.float32)
        n = min(self.num_vehicles, max_vehicles)
        
        for i in range(n):
            base = i * per_vehicle_dim
            arr[base + 0] = self.relative_positions[i, 0]
            arr[base + 1] = self.relative_positions[i, 1]
            arr[base + 2] = self.relative_velocities[i, 0]
            arr[base + 3] = self.relative_velocities[i, 1]
            arr[base + 4] = self.headings[i]
            arr[base + 5] = self.distances[i]
        
        return arr


@dataclass
class MDPTransition:
    """
    Single MDP transition tuple.
    
    Standard (s, a, r, s', done) format for reinforcement learning.
    
    Attributes:
        state: Current observation
        action: Action taken
        reward: Reward received
        next_state: Next observation
        done: Episode termination flag
        info: Additional information
    """
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.tolist(),
            "action": self.action.tolist(),
            "reward": self.reward,
            "next_state": self.next_state.tolist(),
            "done": self.done,
            "info": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.info.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MDPTransition":
        """Create from dictionary."""
        return cls(
            state=np.array(data["state"]),
            action=np.array(data["action"]),
            reward=data["reward"],
            next_state=np.array(data["next_state"]),
            done=data["done"],
            info=data.get("info", {}),
        )


@dataclass
class CMDPTransition:
    """
    Single CMDP (Constrained MDP) transition tuple.
    
    Extends MDP with safety cost for safe reinforcement learning.
    Format: (s, a, r, c, s', done)
    
    Attributes:
        state: Current observation
        action: Action taken
        reward: Reward received
        cost: Safety cost (0 = safe, positive = unsafe)
        next_state: Next observation
        done: Episode termination flag
        surrounding_vehicles: Optional surrounding vehicle info
        info: Additional information
    """
    state: np.ndarray
    action: np.ndarray
    reward: float
    cost: float
    next_state: np.ndarray
    done: bool
    surrounding_vehicles: Optional[SurroundingVehicleInfo] = None
    info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "state": self.state.tolist(),
            "action": self.action.tolist(),
            "reward": self.reward,
            "cost": self.cost,
            "next_state": self.next_state.tolist(),
            "done": self.done,
            "info": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.info.items()
            },
        }
        if self.surrounding_vehicles is not None:
            result["surrounding_vehicles"] = self.surrounding_vehicles.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CMDPTransition":
        """Create from dictionary."""
        surrounding = None
        if "surrounding_vehicles" in data:
            surrounding = SurroundingVehicleInfo.from_dict(data["surrounding_vehicles"])
        
        return cls(
            state=np.array(data["state"]),
            action=np.array(data["action"]),
            reward=data["reward"],
            cost=data["cost"],
            next_state=np.array(data["next_state"]),
            done=data["done"],
            surrounding_vehicles=surrounding,
            info=data.get("info", {}),
        )
    
    def to_mdp(self) -> MDPTransition:
        """Convert to MDP transition (drops cost)."""
        return MDPTransition(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            done=self.done,
            info=self.info,
        )


@dataclass
class Episode:
    """
    A complete episode of transitions.
    
    Attributes:
        transitions: List of transitions
        total_reward: Sum of rewards
        total_cost: Sum of costs (CMDP only)
        metadata: Episode metadata (scenario_id, etc.)
    """
    transitions: List[MDPTransition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def length(self) -> int:
        return len(self.transitions)
    
    @property
    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)
    
    @property
    def total_cost(self) -> float:
        return sum(t.cost for t in self.transitions if hasattr(t, "cost"))
    
    @property
    def is_cmdp(self) -> bool:
        return len(self.transitions) > 0 and isinstance(self.transitions[0], CMDPTransition)
