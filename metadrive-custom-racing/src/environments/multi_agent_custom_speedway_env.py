"""
Multi-agent environment for custom speedway track with complex layout.

This extends the single-agent custom speedway environment to support multiple agents racing simultaneously.
"""
from __future__ import annotations

from typing import Dict, Any

try:
    from metadrive.envs import MultiAgentMetaDrive  # type: ignore
    from metadrive.component.map.pg_map import PGMap  # type: ignore
    from metadrive.component.pgblock.first_block import FirstPGBlock  # type: ignore
    from metadrive.component.pgblock.straight import Straight  # type: ignore
    from metadrive.component.pgblock.curve import Curve  # type: ignore
    from metadrive.component.pg_space import Parameter  # type: ignore
    from metadrive.constants import PGLineType  # type: ignore
    from metadrive.manager.pg_map_manager import PGMapManager  # type: ignore
except ImportError:
    # Fallback for local workspace where MetaDrive source is a sibling folder
    import os, sys
    _here = os.path.dirname(__file__)
    _workspace_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
    _metadrive_src = os.path.join(_workspace_root, "metadrive")
    if _metadrive_src not in sys.path:
        sys.path.insert(0, _metadrive_src)
    from metadrive.envs import MultiAgentMetaDrive  # type: ignore
    from metadrive.component.map.pg_map import PGMap  # type: ignore
    from metadrive.component.pgblock.first_block import FirstPGBlock  # type: ignore
    from metadrive.component.pgblock.straight import Straight  # type: ignore
    from metadrive.component.pgblock.curve import Curve  # type: ignore
    from metadrive.component.pg_space import Parameter  # type: ignore
    from metadrive.constants import PGLineType  # type: ignore
    from metadrive.manager.pg_map_manager import PGMapManager  # type: ignore


class MultiAgentCustomSpeedwayMap(PGMap):
    """A multi-agent PGMap that creates the complex custom speedway layout.

    This includes hairpin turns, S-sections, complex curves, and various straights
    to create a challenging racing circuit for multiple agents.
    """

    def _generate(self):
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "Map not empty; create fresh map."

        lane_num = self.config.get("lane_num", 2)  # 2 lanes for side-by-side racing
        lane_width = self.config.get("lane_width", 4.0)  # Standard lane width
        self.random_seed = 42

        # First (spawn) block
        last_block = FirstPGBlock(
            self.road_network,
            lane_width=lane_width,
            lane_num=lane_num,
            render_root_np=parent_node_path,
            physics_world=physics_world,
            remove_negative_lanes=True,
            side_lane_line_type=PGLineType.CONTINUOUS,
            center_line_type=PGLineType.BROKEN,
        )
        self.blocks.append(last_block)

        block_index = 1

        # Define the custom speedway layout (same as single-agent version)
        layout = [
            # Start straight
            ("Straight", {Parameter.length: 250}),
            
            # Turn 1: hairpin turn
            ("Curve", {Parameter.radius: 40, Parameter.angle: 180, Parameter.dir: 1}),
            
            # Short straight into Turn 2 (left)
            ("Straight", {Parameter.length: 150}),
            ("Curve", {Parameter.radius: 100, Parameter.angle: 45, Parameter.dir: 0}),
            
            # S-section: left-right-left-right
            ("Straight", {Parameter.length: 100}),
            ("Curve", {Parameter.radius: 50, Parameter.angle: 45, Parameter.dir: 0}),
            ("Curve", {Parameter.radius: 50, Parameter.angle: 45, Parameter.dir: 1}),
            ("Curve", {Parameter.radius: 50, Parameter.angle: 45, Parameter.dir: 0}),
            ("Curve", {Parameter.radius: 50, Parameter.angle: 45, Parameter.dir: 1}),
            
            # Long straight (back straight)
            ("Straight", {Parameter.length: 200}),
            
            # Complex right turn
            ("Curve", {Parameter.radius: 90, Parameter.angle: 120, Parameter.dir: 1}),
            
            # Short final straight into the start/finish
            ("Straight", {Parameter.length: 100}),

            ("Curve", {Parameter.radius: 100, Parameter.angle: 90, Parameter.dir: 1}),
            ("Straight", {Parameter.length: 75}),
            ("Curve", {Parameter.radius: 50, Parameter.angle: 170, Parameter.dir: 0}),

            ("Straight", {Parameter.length: 75}),
            ("Curve", {Parameter.radius: 180, Parameter.angle: 90, Parameter.dir: 1}),
            ("Straight", {Parameter.length: 100}),
            ("Curve", {Parameter.radius: 90, Parameter.angle: 90, Parameter.dir: 1}),
            ("Curve", {Parameter.radius: 180, Parameter.angle: 90, Parameter.dir: 1}),

            ("Curve", {Parameter.radius: 40, Parameter.angle: 120, Parameter.dir: 0}),
            ("Straight", {Parameter.length: 83}),
            ("Curve", {Parameter.radius: 60, Parameter.angle: 35, Parameter.dir: 1}),
            ("Straight", {Parameter.length: 150}),
        ]

        for seg_type, params in layout:
            if seg_type == "Straight":
                block = Straight(
                    block_index, 
                    last_block.get_socket(0), 
                    self.road_network, 
                    self.random_seed,
                    remove_negative_lanes=True
                )
            else:
                block = Curve(
                    block_index, 
                    last_block.get_socket(0), 
                    self.road_network, 
                    self.random_seed,
                    remove_negative_lanes=True
                )

            block.construct_from_config(params, parent_node_path, physics_world)
            self.blocks.append(block)
            last_block = block
            block_index += 1


class MultiAgentCustomSpeedwayMapManager(PGMapManager):
    """A map manager that always loads MultiAgentCustomSpeedwayMap once for multi-agent racing."""

    def reset(self):
        # Clear any previously spawned map
        for obj_id in list(self.spawned_objects.keys()):
            self.destroy_object(obj_id)
        # Spawn new map instance
        config = self.engine.global_config
        map_config = dict(config.get("map_config", {}))
        new_map = self.spawn_object(MultiAgentCustomSpeedwayMap, map_config=map_config, random_seed=None)
        self.load_map(new_map)


class MultiAgentCustomSpeedwayEnv(MultiAgentMetaDrive):
    """Multi-agent MetaDrive environment that uses the custom speedway track."""

    def __init__(self, config=None):
        super().__init__(config)

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        try:
            # Real racetrack settings
            cfg['crash_only_done'] = False
            # Set max speed to 120 km/h for racing
            cfg['vehicle_config']['max_speed_km_h'] = 120
            
            # REAL RACETRACK: Crash into other vehicles = episode over
            cfg['crash_vehicle_done'] = True  # Crash into another car = done
            cfg['crash_object_done'] = True   # Crash into barriers = done
            
            # Allow some off-road but penalize heavily
            cfg['out_of_road_done'] = False  # Don't immediately end, but heavy penalty
            
            # Reasonable episode length for racing
            cfg['horizon'] = 1500  # ~1-2 minutes of racing at 60fps
            
            # Success reward for staying on track
            cfg['success_reward'] = 10.0
            cfg['out_of_road_penalty'] = 5.0  # Heavy penalty for leaving track
            
        except Exception:
            pass
        return cfg

    def setup_engine(self):
        """Register our custom map manager with the engine."""
        super().setup_engine()
        # Replace the default map manager with our custom speedway one
        self.engine.update_manager("map_manager", MultiAgentCustomSpeedwayMapManager())

    def step(self, actions):
        observations, rewards, terminateds, truncateds, infos = super().step(actions)

        # Add detailed info and improve rewards for each agent
        try:
            for agent_id, vehicle in self.agents.items():
                agent_info = infos.get(agent_id, {})
                
                # Lane line collision detection
                wl = getattr(vehicle, 'on_white_continuous_line', False)
                yl = getattr(vehicle, 'on_yellow_continuous_line', False)
                bl = getattr(vehicle, 'on_broken_line', False)
                agent_info['white_line_collision'] = bool(wl)
                agent_info['yellow_line_collision'] = bool(yl)
                agent_info['broken_line_collision'] = bool(bl)
                agent_info['lane_line_collision'] = bool(wl or yl or bl)
                
                # Crash detection
                agent_info['crashed'] = bool(
                    getattr(vehicle, 'crash_vehicle', False) or 
                    getattr(vehicle, 'crash_object', False) or
                    getattr(vehicle, 'crash_building', False) or 
                    getattr(vehicle, 'crash_sidewalk', False) or
                    getattr(vehicle, 'crash_human', False)
                )
                
                # Heading information
                try:
                    heading = getattr(vehicle, 'heading', None)
                    if heading is None:
                        heading = getattr(vehicle, 'heading_theta', 0.0)
                    agent_info['heading'] = float(heading)
                except Exception:
                    pass
                
                # Lane offset information
                try:
                    lane = getattr(vehicle, 'lane', None)
                    if lane is not None and hasattr(lane, 'local_coordinates'):
                        s_l = lane.local_coordinates(getattr(vehicle, 'position', [0.0, 0.0]))
                        lane_offset = float(s_l[1]) if isinstance(s_l, (list, tuple)) and len(s_l) > 1 else 0.0
                        agent_info['lane_offset'] = lane_offset
                        lw = getattr(lane, 'width', None)
                        if lw is not None:
                            agent_info['lane_width'] = float(lw)
                except Exception:
                    pass
                
                # RACING REWARD SYSTEM - POSITIVE REWARDS FOR RACING!
                current_reward = rewards.get(agent_id, 0.0)
                
                # MetaDrive base reward is driving_reward = speed * cos(heading_diff)
                # This can be negative if going backwards. We'll override it completely.
                
                speed_kmh = getattr(vehicle, 'speed_km_h', 0.0)
                max_speed = getattr(vehicle, 'max_speed_km_h', 120.0)
                on_road = bool(getattr(vehicle, 'on_lane', True))
                
                # START FRESH - Build positive reward from scratch
                total_reward = 0.0
                
                # 1. MOVEMENT REWARD: Just moving forward is good! (0 to 5.0)
                speed_ratio = min(speed_kmh / max_speed, 1.0)
                movement_reward = speed_ratio * 5.0  # HUGE reward for speed!
                total_reward += movement_reward
                
                # 2. STANDING STILL PENALTY: Not moving = BAD!
                if speed_kmh < 1.0:  # Basically stopped
                    total_reward -= 2.0  # Big penalty for not moving
                
                # 3. ON ROAD BONUS: Stay on track (+1.0)
                if on_road:
                    total_reward += 1.0
                else:
                    # Off road = lose most bonuses
                    total_reward = -2.0
                
                # 4. CRASH = penalty but can recover
                crash_penalty = 0.0
                if agent_info['crashed']:
                    crash_penalty = -3.0
                    total_reward += crash_penalty
                
                # Use our total reward instead of MetaDrive's
                rewards[agent_id] = total_reward
                
                # Vehicle state summary
                agent_info['vehicle_state'] = {
                    'speed': speed_kmh / max(1e-6, max_speed),
                    'position': list(getattr(vehicle, 'position', [0.0, 0.0])),
                    'on_road': on_road,
                    'crashed': agent_info['crashed'],
                }
                agent_info['reward_components'] = {
                    'movement': movement_reward,
                    'standing_penalty': -2.0 if speed_kmh < 1.0 else 0.0,
                    'on_road': 1.0 if on_road else -2.0,
                    'crash': crash_penalty,
                    'total': rewards[agent_id]
                }
                
                infos[agent_id] = agent_info
        except Exception:
            pass

        # Ensure __all__ keys are properly set
        try:
            truncateds["__all__"] = all(v for k, v in truncateds.items() if k != "__all__")
            terminateds["__all__"] = all(v for k, v in terminateds.items() if k != "__all__")
        except Exception:
            pass
            
        return observations, rewards, terminateds, truncateds, infos
    
    def done_function(self, vehicle_id: str):
        """Check if an agent is done - REAL RACETRACK rules."""
        cfg = getattr(self, 'config', {}) or {}
        vehicle = self.vehicles[vehicle_id]
        
        # REAL RACETRACK TERMINATION CONDITIONS:
        
        # 1. Crashed into another vehicle (racing incident)
        vehicle_crash = vehicle.crash_vehicle
        
        # 2. Crashed into barriers/objects (off-track crash)
        object_crash = vehicle.crash_object or vehicle.crash_sidewalk
        
        # 3. Severely off-road (completely left the track area)
        severely_off_road = False
        try:
            # If way off the lane (>2x lane width), consider it off-track
            lane = getattr(vehicle, 'lane', None)
            if lane is not None and hasattr(lane, 'local_coordinates'):
                s_l = lane.local_coordinates(getattr(vehicle, 'position', [0.0, 0.0]))
                lane_offset = abs(float(s_l[1])) if isinstance(s_l, (list, tuple)) and len(s_l) > 1 else 0.0
                lane_width = getattr(lane, 'width', 8.0)
                # If more than 2x lane width off center = completely off track
                if lane_offset > lane_width * 2.0:
                    severely_off_road = True
        except:
            pass
        
        # Episode ends on: crashes OR going completely off-track
        done = vehicle_crash or object_crash or severely_off_road
        
        done_info = {
            "crash_vehicle": vehicle.crash_vehicle,
            "crash_object": vehicle.crash_object,
            "crash_sidewalk": vehicle.crash_sidewalk,
            "severely_off_road": severely_off_road,
            "out_of_road": not vehicle.on_lane,
            "arrive_dest": False,
            "max_step": False,
            "lane_line_collision": False,
            "white_line_collision": False,
            "yellow_line_collision": False,
        }
        return done, done_info

    def reset(self, **kwargs):
        """Reset the environment with optional debug logging."""
        try:
            cfg = getattr(self, 'config', {}) or {}
            map_cfg = cfg.get('map_config', {}) or {}
            debug = bool(cfg.get('debug', False))
            if debug:
                lane_num = map_cfg.get('lane_num', None)
                lane_width = map_cfg.get('lane_width', None)
                agent_cfgs = cfg.get('agent_configs', {}) or {}
                print(f"[SpawnDebug] Custom Speedway: lane_num={lane_num}, lane_width={lane_width}, agents={len(agent_cfgs)}")
                for aid, ac in agent_cfgs.items():
                    sli = ac.get('spawn_lane_index', None)
                    slong = ac.get('spawn_longitude', None)
                    slat = ac.get('spawn_lateral', None)
                    print(f"[SpawnDebug] {aid}: longitude={slong}, lateral={slat}, lane_index={sli}")
        except Exception:
            pass
        return super().reset(**kwargs)
