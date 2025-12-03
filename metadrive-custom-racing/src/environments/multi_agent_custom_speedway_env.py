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
        # Track forward progress for each agent (for reward calculation)
        self._previous_lane_progress = {}

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
            
            # DISABLE ALL MetaDrive built-in rewards - we use 100% custom rewards
            cfg['success_reward'] = 0.0          # Disabled
            cfg['out_of_road_penalty'] = 0.0     # Disabled
            cfg['crash_vehicle_penalty'] = 0.0   # Disabled
            cfg['crash_object_penalty'] = 0.0    # Disabled
            cfg['crash_sidewalk_penalty'] = 0.0  # Disabled
            cfg['driving_reward'] = 0.0          # Disabled
            cfg['speed_reward'] = 0.0            # Disabled

            # Boundary handling mode (True = training/lenient, False = racing/strict)
            cfg['boundary_training_mode'] = True  # Default to training mode

            # Ghost mode for multi-agent training (True = small crash penalty, False = large penalty)
            # When True: agents can "see" each other but collisions only give -5 penalty
            # When False: collisions give -50 penalty (competitive racing)
            cfg['ghost_mode'] = False  # Default to competitive mode

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
                
                # AGGRESSIVE RACING REWARD SYSTEM - SPEED IS EVERYTHING!
                current_reward = rewards.get(agent_id, 0.0)

                speed_kmh = getattr(vehicle, 'speed_km_h', 0.0)
                max_speed = getattr(vehicle, 'max_speed_km_h', 120.0)
                on_road = bool(getattr(vehicle, 'on_lane', True))

                # Check ONLY continuous lines (boundaries) - broken lines are OK!
                yellow_continuous = bool(getattr(vehicle, 'on_yellow_continuous_line', False))
                white_continuous = bool(getattr(vehicle, 'on_white_continuous_line', False))

                # Calculate forward progress along track (CRITICAL FOR RACING!)
                lane = getattr(vehicle, 'lane', None)
                progress_reward = 0.0
                if lane is not None and hasattr(lane, 'local_coordinates'):
                    try:
                        current_progress = float(lane.local_coordinates(getattr(vehicle, 'position', [0.0, 0.0]))[0])
                        previous_progress = self._previous_lane_progress.get(agent_id, current_progress)
                        progress_delta = current_progress - previous_progress
                        self._previous_lane_progress[agent_id] = current_progress

                        # Reward forward movement (even if slow!)
                        if progress_delta > 0:
                            progress_reward = progress_delta * 2.0  # 2 points per meter forward
                    except:
                        pass

                total_reward = 0.0

                # 0. FORWARD PROGRESS - MOST IMPORTANT!
                total_reward += progress_reward

                # 1. STAY ON TRACK BONUS
                if not yellow_continuous and not white_continuous and on_road:
                    total_reward += 1.0  # Bonus for staying on track

                # 2. SPEED REWARD - Gentler at low speeds, big rewards for fast!
                if speed_kmh >= 60:
                    total_reward += 10.0  # Excellent speed
                elif speed_kmh >= 40:
                    total_reward += 5.0   # Good speed
                elif speed_kmh >= 20:
                    total_reward += 2.0   # OK speed - POSITIVE!
                elif speed_kmh >= 10:
                    total_reward += 0.5   # Slow but moving - POSITIVE!
                elif speed_kmh >= 5:
                    total_reward += 0.0   # Very slow - neutral
                else:
                    total_reward -= 1.0   # Stationary - small penalty (was -10!)

                # 3. OUT OF BOUNDS penalties (solid yellow/white ONLY!)
                # Broken lines (lane dividers) are OK - no penalty!
                if yellow_continuous:  # Solid yellow = track boundary
                    total_reward -= 100.0  # Reduced from 200
                if white_continuous:   # Solid white = track edge
                    total_reward -= 100.0  # Reduced from 200

                # 3. OFF ROAD = bad
                if not on_road:
                    total_reward -= 5.0

                # 4. CRASH = depends on mode
                if agent_info['crashed']:
                    ghost_mode = cfg.get('ghost_mode', False)
                    if ghost_mode:
                        # Ghost mode: small penalty, agents learning to race together
                        total_reward -= 5.0
                    else:
                        # Competitive mode: large penalty, real racing
                        total_reward -= 50.0

                # Use our total reward
                rewards[agent_id] = total_reward
                
                # Vehicle state summary
                agent_info['vehicle_state'] = {
                    'speed': speed_kmh / max(1e-6, max_speed),
                    'position': list(getattr(vehicle, 'position', [0.0, 0.0])),
                    'on_road': on_road,
                    'crashed': agent_info['crashed'],
                }
                # Calculate individual reward components for logging (match actual rewards!)
                speed_reward = 0.0
                if speed_kmh >= 60:
                    speed_reward = 10.0
                elif speed_kmh >= 40:
                    speed_reward = 5.0
                elif speed_kmh >= 20:
                    speed_reward = 2.0
                elif speed_kmh >= 10:
                    speed_reward = 0.5
                elif speed_kmh >= 5:
                    speed_reward = 0.0
                else:
                    speed_reward = -1.0

                # Calculate on-track bonus
                on_track_bonus = 1.0 if (not yellow_continuous and not white_continuous and on_road) else 0.0

                # Calculate boundary violation penalty (ONLY continuous lines!)
                boundary_penalty = 0.0
                if yellow_continuous:
                    boundary_penalty -= 100.0
                if white_continuous:
                    boundary_penalty -= 100.0

                # Calculate crash penalty for logging (match actual)
                crash_penalty = 0.0
                if agent_info['crashed']:
                    ghost_mode = cfg.get('ghost_mode', False)
                    crash_penalty = -5.0 if ghost_mode else -50.0

                agent_info['reward_components'] = {
                    'forward_progress': progress_reward,
                    'on_track_bonus': on_track_bonus,
                    'speed_reward': speed_reward,
                    'boundary_penalty': boundary_penalty,
                    'off_road_penalty': -5.0 if not on_road else 0.0,
                    'crash_penalty': crash_penalty,
                    'total': rewards[agent_id]
                }

                # Add detailed metrics for wandb
                agent_info['metrics'] = {
                    'speed_kmh': speed_kmh,
                    'on_road': on_road,
                    'on_boundary': yellow_continuous or white_continuous,
                    'crashed': agent_info['crashed'],
                }
                
                infos[agent_id] = agent_info
        except Exception as e:
            # DON'T SILENTLY FAIL! Print errors so we can debug!
            print(f"⚠️  ERROR in custom reward calculation: {e}")
            import traceback
            traceback.print_exc()

        # Ensure __all__ keys are properly set
        try:
            truncateds["__all__"] = all(v for k, v in truncateds.items() if k != "__all__")
            terminateds["__all__"] = all(v for k, v in terminateds.items() if k != "__all__")
        except Exception:
            pass
            
        return observations, rewards, terminateds, truncateds, infos
    
    def done_function(self, vehicle_id: str):
        """Check if an agent is done - respects config settings for training vs racing."""
        cfg = getattr(self, 'config', {}) or {}
        vehicle = self.vehicles[vehicle_id]

        # Respect config settings for crash termination
        crash_vehicle_done = cfg.get('crash_vehicle_done', False)
        crash_object_done = cfg.get('crash_object_done', False)
        out_of_road_done = cfg.get('out_of_road_done', False)
        lane_line_done = cfg.get('lane_line_done', True)  # Terminate on boundary crossing

        # 1. Crashed into another vehicle (only if enabled)
        vehicle_crash = vehicle.crash_vehicle if crash_vehicle_done else False

        # 2. Crashed into barriers/objects (only if enabled)
        object_crash = (vehicle.crash_object or vehicle.crash_sidewalk) if crash_object_done else False

        # 3. OUT OF BOUNDS = TERMINATE (solid yellow/white lines ONLY!)
        # BUT during training, give penalty without terminating (let agent learn to recover)
        out_of_bounds = False
        if lane_line_done:
            try:
                # ONLY solid/continuous lines = track boundaries
                yellow_continuous = getattr(vehicle, 'on_yellow_continuous_line', False)
                white_continuous = getattr(vehicle, 'on_white_continuous_line', False)

                # Check if we're in training mode (lenient) or evaluation mode (strict)
                training_mode = cfg.get('boundary_training_mode', True)  # Default: lenient during training

                if yellow_continuous or white_continuous:
                    if not training_mode:
                        # STRICT MODE: Terminate immediately (for racing/evaluation)
                        out_of_bounds = True
                    # else: TRAINING MODE: Penalty already applied in reward, don't terminate
                    #       Let agent learn to recover from mistakes
            except:
                pass

        # 4. Severely off-road (only if enabled)
        severely_off_road = False
        if out_of_road_done:
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
        
        # Episode ends on: crashes OR out of bounds OR going completely off-track
        done = vehicle_crash or object_crash or out_of_bounds or severely_off_road
        
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
        # Reset progress tracking
        self._previous_lane_progress = {}

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
