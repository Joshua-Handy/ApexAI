"""
Multi-agent environment for custom speedway track with SIMPLE CLEAN REWARD SYSTEM.

This version uses a straightforward reward structure without complex corner bonuses or velocity deltas:

POSITIVE REWARDS:
  +20.0 * progress     - Forward movement (delta completion)
  +1.0 * completion    - Small bonus for being farther along track
  +1.0 * speed_norm    - Small speed bonus (not dominant)

CONSTANT TIME PENALTY:
  -0.02 per step       - Encourages continuous movement

NEGATIVE PENALTIES (major events):
  -20.0 off-road       - Leaving the track
  -20.0 crash          - Vehicle collision
  -20.0 white-line     - Boundary violation

KEY FEATURES:
  - Simple and clean: No conflicting signals
  - Progress-focused: Main reward is forward movement
  - Clear penalties: Major negative events have equal weight
  - No exploit opportunities: Removed complex corner bonuses and velocity deltas
  - Agent learns naturally: Speed and technique emerge from progress optimization

TRAINING MODE: out_of_road_done = False (lenient boundaries, allows off-road penalty learning)
RACING MODE: out_of_road_done = True (strict boundaries, competition mode)
"""
from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np
import math

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


class MultiAgentCustomSpeedwayMapCurve(PGMap):
    """Custom speedway map that tracks segment information for curve detection."""

    def _generate(self):
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "Map not empty; create fresh map."

        lane_num = self.config.get("lane_num", 2)
        lane_width = self.config.get("lane_width", 12.0)  # FIXED: Match training default
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

        # F1-STYLE TRACK LAYOUT (same as original)
        layout = [
            ("Straight", {Parameter.length: 200}),
            ("Curve", {Parameter.radius: 50, Parameter.angle: 90, Parameter.dir: 1}),
            ("Straight", {Parameter.length: 150}),
            ("Curve", {Parameter.radius: 50, Parameter.angle: 90, Parameter.dir: 1}),
            ("Straight", {Parameter.length: 100}),
            ("Curve", {Parameter.radius: 40, Parameter.angle: 90, Parameter.dir: 1}),
            ("Straight", {Parameter.length: 80}),
            ("Curve", {Parameter.radius: 50, Parameter.angle: 90, Parameter.dir: 0}),
            ("Straight", {Parameter.length: 60}),
            ("Curve", {Parameter.radius: 50, Parameter.angle: 90, Parameter.dir: 1}),
            ("Straight", {Parameter.length: 100}),
            ("Curve", {Parameter.radius: 60, Parameter.angle: 180, Parameter.dir: 1}),
            ("Straight", {Parameter.length: 30}),
            ("Curve", {Parameter.radius: 40, Parameter.angle: 90, Parameter.dir: 0}),
            ("Straight", {Parameter.length: 25}),
        ]

        # Build track and calculate segment info for curve detection
        self.segment_info = []
        cumulative_length = 0.0

        for seg_type, params in layout:
            if seg_type == "Straight":
                seg_length = params[Parameter.length]
                self.segment_info.append({
                    'type': 'Straight',
                    'length': seg_length,
                    'start': cumulative_length,
                    'end': cumulative_length + seg_length,
                    'curve_strength': 0.0  # No curve
                })
                cumulative_length += seg_length

                block = Straight(
                    block_index,
                    last_block.get_socket(0),
                    self.road_network,
                    self.random_seed,
                    remove_negative_lanes=True
                )
            else:  # Curve
                radius = params[Parameter.radius]
                angle = params[Parameter.angle]
                seg_length = abs(angle * math.pi / 180.0 * radius)  # Arc length

                # Calculate curve strength: turn_rate = abs(angle) / seg_len, normalized
                turn_rate = abs(angle) / seg_length if seg_length > 0 else 0
                curve_strength = min(turn_rate / 1.5, 1.0)

                self.segment_info.append({
                    'type': 'Curve',
                    'radius': radius,
                    'angle': angle,
                    'length': seg_length,
                    'start': cumulative_length,
                    'end': cumulative_length + seg_length,
                    'curve_strength': curve_strength
                })
                cumulative_length += seg_length

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

        # Store total track length for progress calculation
        self.total_track_length = cumulative_length


class MultiAgentCustomSpeedwayMapManagerCurve(PGMapManager):
    """Map manager for curve-aware speedway."""

    def reset(self):
        for obj_id in list(self.spawned_objects.keys()):
            self.destroy_object(obj_id)
        config = self.engine.global_config
        map_config = dict(config.get("map_config", {}))
        new_map = self.spawn_object(MultiAgentCustomSpeedwayMapCurve, map_config=map_config, random_seed=None)
        self.load_map(new_map)


class MultiAgentCustomSpeedwayCurveEnv(MultiAgentMetaDrive):
    """Multi-agent environment with CURVE-AWARE reward system.

    Implements adaptive speed rewards based on track curvature.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._last_completion = {}
        self._step_count = 0
        self._debug_log_file = None
        self._debug_log_path = None
        self._action_history = {}  # For steering jitter detection
        self._steering_history = {}
        self._last_speed = {}  # For acceleration bonus
        self._terminated_agents = set()
        self._offroad_steps = {}  # Track consecutive off-road steps per agent
        self._max_offroad_steps = 30  # Terminate after 30 consecutive off-road steps

    @staticmethod
    def default_config():
        cfg = MultiAgentMetaDrive.default_config()
        cfg.update({
            'num_agents': 2,
            'use_render': False,
            'map_config': {
                'lane_num': 2,
                'lane_width': 12.0,  # Wider lanes for curve navigation
            },
            'start_seed': 42,
            'horizon': 1500,
            'vehicle_config': {
                'max_speed_km_h': 100,  # Reduced from 120 to improve curve stability
            },
            # LENIENT TRAINING MODE: Heavy penalty but no termination
            # This allows agents to explore and learn without instant death
            # (Same as agent_0v2 which succeeded with this setting)
            'crash_vehicle_done': True,
            'crash_object_done': True,
            'out_of_road_done': False,  # Training mode: penalty only, no termination
            'show_terrain': True,
            'show_sidewalk': True,
        })

        try:
            cfg['boundary_training_mode'] = False  # Strict boundaries
            cfg['ghost_mode'] = False
        except Exception:
            pass
        return cfg

    def setup_engine(self):
        """Register curve-aware map manager."""
        super().setup_engine()
        self.engine.update_manager("map_manager", MultiAgentCustomSpeedwayMapManagerCurve())

    # REMOVED: _get_curve_strength() method - agents can't reach curves yet, unnecessary overhead

    def step(self, actions):
        self._step_count += 1

        # Freeze terminated agents
        filtered_actions = {}
        for agent_id, action in actions.items():
            if agent_id not in self._terminated_agents:
                filtered_actions[agent_id] = action
            else:
                filtered_actions[agent_id] = np.array([0.0, 0.0])

        observations, rewards, terminateds, truncateds, infos = super().step(filtered_actions)

        # CURVE-AWARE REWARD CALCULATION
        try:
            for agent_id, vehicle in self.agents.items():
                agent_info = infos.get(agent_id, {})

                # Basic vehicle state
                wl = getattr(vehicle, 'on_white_continuous_line', False)
                yl = getattr(vehicle, 'on_yellow_continuous_line', False)
                bl = getattr(vehicle, 'on_broken_line', False)
                agent_info['white_line_collision'] = bool(wl)
                agent_info['yellow_line_collision'] = bool(yl)
                agent_info['broken_line_collision'] = bool(bl)
                agent_info['lane_line_collision'] = bool(wl or yl or bl)

                agent_info['crashed'] = bool(
                    getattr(vehicle, 'crash_vehicle', False) or
                    getattr(vehicle, 'crash_object', False) or
                    getattr(vehicle, 'crash_building', False) or
                    getattr(vehicle, 'crash_sidewalk', False) or
                    getattr(vehicle, 'crash_human', False)
                )

                # Get speed
                speed_kmh = getattr(vehicle, 'speed_km_h', None)
                if speed_kmh is None:
                    speed_ms = getattr(vehicle, 'speed', 0.0)
                    speed_kmh = speed_ms * 3.6

                max_speed = getattr(vehicle, 'max_speed_km_h', 120.0)
                on_road = bool(getattr(vehicle, 'on_lane', True))

                # Route completion and progress
                try:
                    # FIX: Use map's consistent total_track_length instead of navigation's variable total_length
                    # Navigation total_length varies between agents causing incorrect completion jumps
                    current_map = self.engine.current_map

                    if current_map and hasattr(current_map, 'total_track_length') and hasattr(vehicle.navigation, 'travelled_length'):
                        # Use navigation's travelled_length but divide by map's CONSISTENT total_track_length
                        map_total_length = current_map.total_track_length
                        travelled = vehicle.navigation.travelled_length
                        completion = (travelled % map_total_length) / map_total_length
                    else:
                        # Fallback to navigation completion
                        completion = vehicle.navigation.route_completion

                    # DEBUG: Check navigation tracking on first step
                    if self.episode_step == 1:
                        nav_travelled = getattr(vehicle.navigation, 'travelled_length', None)
                        nav_total = getattr(vehicle.navigation, 'total_length', None)
                        map_total = getattr(current_map, 'total_track_length', None) if current_map else None
                        print(f"[NAV DEBUG] {agent_id} | nav_travelled={nav_travelled:.2f}m | nav_total={nav_total:.2f}m | map_total={map_total:.2f}m | fixed_completion={completion:.4f}")

                    completion = max(0.0, min(1.0, completion))
                    last_completion = self._last_completion.get(agent_id, 0.0)

                    # Handle lap completion
                    if completion < 0.2 and last_completion > 0.8:
                        progress = (1.0 - last_completion) + completion
                    else:
                        progress = completion - last_completion
                        if progress < 0:
                            progress = 0.0

                    self._last_completion[agent_id] = completion
                except Exception as e:
                    print(f"⚠️  Warning: Failed to calculate completion for {agent_id}: {e}")
                    progress = 0.0
                    completion = 0.0

                # Removed curve_strength - agents can't reach curves yet, no point calculating it

                # ========================================
                # SIMPLE CLEAN REWARD SYSTEM
                # No velocity delta, no corner bonuses, no complex exploit fixes
                # Just: progress, speed, and clear penalties
                # ========================================
                reward = 0.0
                speed_normalized = speed_kmh / max_speed

                # Get action for logging
                current_action = filtered_actions.get(agent_id, np.array([0.0, 0.0]))
                steering_action = current_action[0] if len(current_action) > 0 else 0.0
                throttle_action = current_action[1] if len(current_action) > 1 else 0.0

                # === POSITIVE REWARDS ===
                # Reward for NEW progress (delta completion)
                reward += progress * 20.0

                # Small bonus for being farther along the track
                reward += completion * 1.0

                # Small bonus for moving fast (not dominant)
                reward += 1.0 * speed_normalized

                # === CONSTANT TIME PENALTY (so stopping = bad) ===
                reward -= 0.02

                # === MAJOR NEGATIVE EVENTS ===
                # Out of road - track consecutive steps (only if not already terminated)
                if not on_road:
                    reward -= 20.0

                    # Only track/check termination if agent not already terminated
                    if agent_id not in self._terminated_agents:
                        # Track consecutive off-road steps
                        self._offroad_steps[agent_id] = self._offroad_steps.get(agent_id, 0) + 1

                        # Terminate after 30 consecutive off-road steps
                        if self._offroad_steps[agent_id] >= self._max_offroad_steps:
                            terminateds[agent_id] = True
                            self._terminated_agents.add(agent_id)
                            print(f"   ☠️  {agent_id} terminated: {self._max_offroad_steps} consecutive off-road steps!")
                else:
                    # Reset counter when back on road
                    if agent_id not in self._terminated_agents:
                        self._offroad_steps[agent_id] = 0

                # Crash
                if agent_info['crashed']:
                    reward -= 20.0

                # White line boundary violation
                white_continuous = bool(getattr(vehicle, 'on_white_continuous_line', False))
                if white_continuous:
                    reward -= 20.0

                # Store completion for lap detection
                self._last_completion[agent_id] = completion

                # DETAILED DEBUG LOGGING (every 50 steps OR when off-road/crashed)
                log_this_step = (self.episode_step % 50 == 0 or not on_road or agent_info['crashed'])

                if log_this_step:
                    # Get lane information
                    try:
                        lane_index = vehicle.lane_index[1] if hasattr(vehicle, 'lane_index') else "?"
                        lateral_pos = vehicle.lane.local_coordinates(vehicle.position)[0] if hasattr(vehicle, 'lane') else 0.0
                    except:
                        lane_index = "?"
                        lateral_pos = 0.0

                    # Segment type (simplified - no curve detection)
                    seg_type = "TRACK"

                    # Steering direction
                    if abs(steering_action) < 0.1:
                        steer_dir = "CENTER"
                    elif steering_action > 0.7:
                        steer_dir = "FULL_RIGHT"
                    elif steering_action < -0.7:
                        steer_dir = "FULL_LEFT"
                    elif steering_action > 0:
                        steer_dir = "RIGHT"
                    else:
                        steer_dir = "LEFT"

                    # Throttle status
                    if throttle_action > 0.7:
                        throttle_str = "FULL_THROTTLE"
                    elif throttle_action > 0.3:
                        throttle_str = "THROTTLE"
                    elif throttle_action > -0.3:
                        throttle_str = "COAST"
                    elif throttle_action > -0.7:
                        throttle_str = "BRAKE"
                    else:
                        throttle_str = "FULL_BRAKE"

                    # Calculate reward components for display
                    components = []
                    if progress > 0.01:
                        components.append(f"Progress(+{progress*20.0:.1f})")
                    if speed_normalized > 0.5:
                        components.append(f"Speed(+{speed_normalized:.2f})")
                    if completion > 0.5:
                        components.append(f"Completion(+{completion:.2f})")
                    if not on_road:
                        components.append("OffRoad(-20)")
                    if agent_info['crashed']:
                        components.append("Crash(-20)")
                    if white_continuous:
                        components.append("WhiteLine(-20)")

                    comp_str = ", ".join(components) if components else "Baseline"
                    road_status = "ON_ROAD" if on_road else "⚠️OFF_ROAD"

                    print(f"[DEBUG] Step {self.episode_step:4d} | {agent_id:8s} | "
                          f"Pos:{completion:5.1%} {seg_type:14s} | "
                          f"Speed:{speed_kmh:5.1f}km/h | "
                          f"Steer:{steer_dir:11s} Throttle:{throttle_str:14s} | "
                          f"{road_status:12s} | Components:[{comp_str}] | Reward:{reward:+8.1f}")

                # Store final reward
                rewards[agent_id] = reward

                # Logging info
                agent_info['vehicle_state'] = {
                    'speed': speed_kmh / max(1e-6, max_speed),
                    'position': list(getattr(vehicle, 'position', [0.0, 0.0])),
                    'on_road': on_road,
                    'crashed': agent_info['crashed'],
                }

                agent_info['reward_components'] = {
                    'progress': progress * 100.0,
                    'speed_reward': reward,  # Simplified for logging
                    'total': rewards[agent_id]
                }

                agent_info['metrics'] = {
                    'speed_kmh': speed_kmh,
                    'on_road': on_road,
                    'crashed': agent_info['crashed'],
                    'route_completion': completion,
                }

                # Store speed and action history for next step
                self._last_speed[agent_id] = speed_kmh
                self._action_history[agent_id] = current_action

                infos[agent_id] = agent_info

                # Track terminated agents
                if terminateds.get(agent_id, False) or truncateds.get(agent_id, False):
                    self._terminated_agents.add(agent_id)

        except Exception as e:
            print(f"⚠️  ERROR in curve-aware reward calculation: {e}")
            import traceback
            traceback.print_exc()

        # Ensure __all__ keys are set
        try:
            truncateds["__all__"] = all(v for k, v in truncateds.items() if k != "__all__")
            terminateds["__all__"] = all(v for k, v in terminateds.items() if k != "__all__")
        except Exception:
            pass

        return observations, rewards, terminateds, truncateds, infos

    def done_function(self, vehicle_id: str):
        """Override done logic: ONLY terminate on off-road, NOT on arrive_dest.

        This prevents the glitch where agents terminate when making forward progress.
        We want continuous laps, not termination on lap completion.
        """
        vehicle = self.agents[vehicle_id]
        done = False
        done_info = {
            "crash_vehicle": False,
            "crash_object": False,
            "out_of_road": False,
            "arrive_dest": False,  # NEVER terminate on destination
            "max_step": False,
        }

        # Check for crashes (if configured)
        if self.config.get('crash_vehicle_done', False):
            if getattr(vehicle, 'crash_vehicle', False):
                done = True
                done_info['crash_vehicle'] = True

        if self.config.get('crash_object_done', False):
            if getattr(vehicle, 'crash_object', False):
                done = True
                done_info['crash_object'] = True

        # Check for out of road (ONLY terminate if actually off-road)
        if self.config.get('out_of_road_done', False):
            if not getattr(vehicle, 'on_lane', True):
                done = True
                done_info['out_of_road'] = True

        # NEVER terminate on arrive_dest - we want continuous laps
        # done_info['arrive_dest'] stays False

        return done, done_info

    def reset(self, **kwargs):
        """Reset tracking state for new episode."""
        self._step_count = 0
        self._last_completion = {}
        self._action_history = {}
        self._steering_history = {}
        self._last_speed = {}
        self._terminated_agents = set()
        self._offroad_steps = {}  # Reset off-road step counter

        result = super().reset(**kwargs)

        # Initialize completion tracking
        for agent_id in self.agents.keys():
            self._last_completion[agent_id] = 0.0
            self._offroad_steps[agent_id] = 0  # Initialize off-road counter

        return result
