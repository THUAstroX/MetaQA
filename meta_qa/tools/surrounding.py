"""
Surrounding Vehicle Information Extraction.

Utilities to extract surrounding vehicle information from MetaDrive environment.
"""

import numpy as np
from typing import List, Optional
from meta_qa.data.structures import SurroundingVehicleInfo


class SurroundingVehicleGetter:
    """
    Utility class to extract surrounding vehicle information from MetaDrive environment.
    
    Handles various MetaDrive versions and extracts vehicle information in both
    world-frame and ego-centric coordinates.
    
    Args:
        max_vehicles: Maximum number of surrounding vehicles to track
        detection_radius: Detection radius in meters
        include_parked: Whether to include parked vehicles (speed < 0.1 m/s)
        
    Usage:
        getter = SurroundingVehicleGetter(max_vehicles=10, detection_radius=50.0)
        info = getter.get_surrounding_vehicles(env)
    """
    
    def __init__(
        self,
        max_vehicles: int = 10,
        detection_radius: float = 50.0,
        include_parked: bool = False,
    ):
        self.max_vehicles = max_vehicles
        self.detection_radius = detection_radius
        self.include_parked = include_parked
    
    def get_surrounding_vehicles(self, env) -> SurroundingVehicleInfo:
        """
        Get surrounding vehicle information from MetaDrive environment.
        
        Args:
            env: MetaDrive environment (ScenarioEnv or TrajectoryEnv)
            
        Returns:
            SurroundingVehicleInfo object
        """
        # Unwrap if necessary to get base environment
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        
        try:
            # Get ego vehicle
            ego = base_env.agent
            ego_pos = np.array([ego.position[0], ego.position[1]])
            ego_heading = ego.heading_theta
            
            # Get all traffic vehicles (excluding ego)
            traffic_vehicles = self._get_traffic_vehicles(base_env, ego_vehicle=ego)
            
            if len(traffic_vehicles) == 0:
                return SurroundingVehicleInfo()
            
            # Collect vehicle information
            positions = []
            velocities = []
            headings = []
            lengths = []
            widths = []
            vehicle_types = []
            
            for vehicle in traffic_vehicles:
                try:
                    pos = np.array([vehicle.position[0], vehicle.position[1]])
                    
                    # Check if within detection radius
                    distance = np.linalg.norm(pos - ego_pos)
                    if distance > self.detection_radius:
                        continue
                    
                    # Check if parked (speed < threshold)
                    vel = np.array([vehicle.velocity[0], vehicle.velocity[1]])
                    speed = np.linalg.norm(vel)
                    if not self.include_parked and speed < 0.1:
                        continue
                    
                    positions.append(pos)
                    velocities.append(vel)
                    headings.append(vehicle.heading_theta)
                    
                    # Get vehicle dimensions
                    if hasattr(vehicle, "LENGTH"):
                        lengths.append(vehicle.LENGTH)
                    elif hasattr(vehicle, "length"):
                        lengths.append(vehicle.length)
                    else:
                        lengths.append(4.5)  # Default
                    
                    if hasattr(vehicle, "WIDTH"):
                        widths.append(vehicle.WIDTH)
                    elif hasattr(vehicle, "width"):
                        widths.append(vehicle.width)
                    else:
                        widths.append(1.8)  # Default
                    
                    # Get vehicle type
                    if hasattr(vehicle, "type"):
                        vehicle_types.append(str(vehicle.type))
                    else:
                        vehicle_types.append("VEHICLE")
                        
                except Exception:
                    continue
            
            if len(positions) == 0:
                return SurroundingVehicleInfo()
            
            # Convert to arrays
            positions = np.array(positions)
            velocities = np.array(velocities)
            headings = np.array(headings)
            lengths = np.array(lengths)
            widths = np.array(widths)
            
            # Sort by distance and limit to max_vehicles
            distances = np.linalg.norm(positions - ego_pos, axis=1)
            sorted_indices = np.argsort(distances)[: self.max_vehicles]
            
            positions = positions[sorted_indices]
            velocities = velocities[sorted_indices]
            headings = headings[sorted_indices]
            lengths = lengths[sorted_indices]
            widths = widths[sorted_indices]
            distances = distances[sorted_indices]
            vehicle_types = [vehicle_types[i] for i in sorted_indices]
            
            # Convert to ego-centric frame
            relative_positions = self._world_to_local(positions, ego_pos, ego_heading)
            relative_velocities = self._world_to_local_velocity(velocities, ego_heading)
            
            return SurroundingVehicleInfo(
                num_vehicles=len(positions),
                positions=positions,
                velocities=velocities,
                headings=headings,
                relative_positions=relative_positions,
                relative_velocities=relative_velocities,
                distances=distances,
                lengths=lengths,
                widths=widths,
                vehicle_types=vehicle_types,
            )
            
        except Exception as e:
            print(f"Error getting surrounding vehicles: {e}")
            return SurroundingVehicleInfo()
    
    def _get_traffic_vehicles(self, env, ego_vehicle=None) -> List:
        """
        Get list of traffic vehicles from environment (excluding ego vehicle).
        
        Args:
            env: MetaDrive environment
            ego_vehicle: Ego vehicle to exclude from the list
            
        Returns:
            List of traffic vehicles (excluding ego)
        """
        vehicles = []
        
        try:
            # Get ego vehicle ID for filtering
            ego_id = None
            if ego_vehicle is not None:
                ego_id = id(ego_vehicle)
                if hasattr(ego_vehicle, "id"):
                    ego_id = ego_vehicle.id
                elif hasattr(ego_vehicle, "name"):
                    ego_id = ego_vehicle.name
            
            # Method 1: From traffic manager
            if hasattr(env, "engine") and hasattr(env.engine, "traffic_manager"):
                tm = env.engine.traffic_manager
                if hasattr(tm, "vehicles"):
                    for v in tm.vehicles:
                        if not self._is_ego_vehicle(v, ego_vehicle, ego_id):
                            vehicles.append(v)
                elif hasattr(tm, "spawned_objects"):
                    for v in tm.spawned_objects.values():
                        if not self._is_ego_vehicle(v, ego_vehicle, ego_id):
                            vehicles.append(v)
            
            # Method 2: From object manager
            if len(vehicles) == 0 and hasattr(env, "engine"):
                if hasattr(env.engine, "object_manager"):
                    om = env.engine.object_manager
                    if hasattr(om, "spawned_objects"):
                        for obj in om.spawned_objects.values():
                            if hasattr(obj, "velocity"):
                                if not self._is_ego_vehicle(obj, ego_vehicle, ego_id):
                                    vehicles.append(obj)
            
            # Method 3: From scene objects
            if len(vehicles) == 0 and hasattr(env, "engine"):
                if hasattr(env.engine, "get_objects"):
                    for obj in env.engine.get_objects():
                        if "vehicle" in str(type(obj)).lower():
                            if not self._is_ego_vehicle(obj, ego_vehicle, ego_id):
                                vehicles.append(obj)
                            
        except Exception as e:
            print(f"Error getting traffic vehicles: {e}")
        
        return vehicles
    
    def _is_ego_vehicle(self, vehicle, ego_vehicle, ego_id) -> bool:
        """Check if a vehicle is the ego vehicle."""
        if ego_vehicle is None:
            return False
        
        # Check by object identity
        if vehicle is ego_vehicle:
            return True
        
        # Check by id/name
        if ego_id is not None:
            if hasattr(vehicle, "id") and vehicle.id == ego_id:
                return True
            if hasattr(vehicle, "name") and vehicle.name == ego_id:
                return True
        
        # Check by position (same position = same vehicle)
        try:
            ego_pos = np.array([ego_vehicle.position[0], ego_vehicle.position[1]])
            veh_pos = np.array([vehicle.position[0], vehicle.position[1]])
            if np.linalg.norm(ego_pos - veh_pos) < 0.1:  # Less than 10cm apart
                return True
        except:
            pass
        
        return False
    
    def _world_to_local(
        self, positions: np.ndarray, ego_pos: np.ndarray, ego_heading: float
    ) -> np.ndarray:
        """Convert world positions to ego-centric local frame."""
        # Translate
        relative = positions - ego_pos
        
        # Rotate (negative heading to align x-axis with ego heading)
        cos_h = np.cos(-ego_heading)
        sin_h = np.sin(-ego_heading)
        
        local_x = relative[:, 0] * cos_h - relative[:, 1] * sin_h
        local_y = relative[:, 0] * sin_h + relative[:, 1] * cos_h
        
        return np.stack([local_x, local_y], axis=1)
    
    def _world_to_local_velocity(
        self, velocities: np.ndarray, ego_heading: float
    ) -> np.ndarray:
        """Convert world velocities to ego-centric local frame."""
        cos_h = np.cos(-ego_heading)
        sin_h = np.sin(-ego_heading)
        
        local_vx = velocities[:, 0] * cos_h - velocities[:, 1] * sin_h
        local_vy = velocities[:, 0] * sin_h + velocities[:, 1] * cos_h
        
        return np.stack([local_vx, local_vy], axis=1)
