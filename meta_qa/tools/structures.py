"""
Data Structures for Offline RL.

Defines MDP and CMDP transition tuples.
"""

import numpy as np
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .surrounding import SurroundingVehicleInfo


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
    surrounding_vehicles: Optional["SurroundingVehicleInfo"] = None
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
        from .surrounding import SurroundingVehicleInfo
        
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
