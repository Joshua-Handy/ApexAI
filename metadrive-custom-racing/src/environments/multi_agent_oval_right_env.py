"""
Multi-agent environment that builds a right-turn-only oval map at runtime using PGBlocks.

This extends the single-agent oval environment to support multiple agents racing simultaneously.
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


class MultiAgentOvalMap(PGMap):
    """A multi-agent PGMap that creates: Straight -> Right Curve -> Straight -> Right Curve -> Straight.

    The parameters (lane_num, lane_width) and lengths are modest defaults appropriate for multi-agent racing.
    """

    def _generate(self):
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "Map is not empty; create a fresh map to build the oval"

        lane_num = self.config.get("lane_num", 1)  # Default to single lane
        lane_width = self.config.get("lane_width", 20.0)  # Match config file lane width

        try:
            eng_cfg = getattr(self.engine, 'global_config', {})
            if bool(getattr(eng_cfg, 'get', lambda k, d=None: d)('debug', False)):
                print(f"[MapDebug] lane_num={lane_num}, lane_width={lane_width}")
        except Exception:
            pass

        # Start with spawn block
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
        # Build the oval: 4 right curves
        for _ in range(4):
            last_block = Curve(
                block_index,
                last_block.get_socket(0),
                self.road_network,
                lane_num,
                remove_negative_lanes=True,
                side_lane_line_type=PGLineType.CONTINUOUS,
                center_line_type=PGLineType.BROKEN,
            )
            last_block.construct_from_config(
                {
                    Parameter.length: 120,
                    Parameter.radius: 120,
                    Parameter.angle: 90,
                    Parameter.dir: 1
                },
                parent_node_path,
                physics_world
            )
            self.blocks.append(last_block)
            block_index += 1


class MultiAgentOvalMapManager(PGMapManager):
    """A map manager that always loads MultiAgentOvalMap once for multi-agent racing."""

    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            # Make a plain dict copy of map_config to avoid Config update restrictions
            map_config = dict(config.get("map_config", {}))
            _map = self.spawn_object(MultiAgentOvalMap, map_config=map_config, random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "Expected exactly one map in this manager"
            _map = list(self.spawned_objects.values())[0]
        self.load_map(_map)


class MultiAgentOvalEnv(MultiAgentMetaDrive):
    """Multi-agent MetaDrive environment that uses MultiAgentOvalMapManager to build a right-only oval."""

    def __init__(self, config=None):
        super().__init__(config)

    def setup_engine(self):
        """Ensure our custom map manager is registered with the engine so the custom
        MultiAgentOvalMap (which builds the oval in _generate) is used instead of
        the default PGMap/BIG generator.
        """
        # Let the base class register default managers first (traffic, etc.)
        super().setup_engine()
        # Overwrite the default map manager with our custom one
        self.engine.update_manager("map_manager", MultiAgentOvalMapManager())

    def step(self, actions):
        observations, rewards, terminateds, truncateds, infos = super().step(actions)

        try:
            for agent_id, vehicle in self.agents.items():
                agent_info = infos.get(agent_id, {})
                wl = getattr(vehicle, 'on_white_continuous_line', False)
                yl = getattr(vehicle, 'on_yellow_continuous_line', False)
                bl = getattr(vehicle, 'on_broken_line', False)
                agent_info['white_line_collision'] = bool(wl)
                agent_info['yellow_line_collision'] = bool(yl)
                agent_info['broken_line_collision'] = bool(bl)
                agent_info['lane_line_collision'] = bool(wl or yl or bl)
                agent_info['crashed'] = bool(getattr(vehicle, 'crash_vehicle', False) or getattr(vehicle, 'crash_object', False)
                                             or getattr(vehicle, 'crash_building', False) or getattr(vehicle, 'crash_sidewalk', False)
                                             or getattr(vehicle, 'crash_human', False))
                try:
                    heading = getattr(vehicle, 'heading', None)
                    if heading is None:
                        heading = getattr(vehicle, 'heading_theta', 0.0)
                    agent_info['heading'] = float(heading)
                except Exception:
                    pass
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
                agent_info['vehicle_state'] = {
                    'speed': getattr(vehicle, 'speed_km_h', 0.0) / max(1e-6, getattr(vehicle, 'max_speed_km_h', 1.0)),
                    'position': list(getattr(vehicle, 'position', [0.0, 0.0])),
                    'on_road': bool(getattr(vehicle, 'on_lane', True)),
                    'crashed': agent_info['crashed'],
                }
                infos[agent_id] = agent_info
        except Exception:
            pass

        # Use MetaDrive's built-in respawn manager and termination handling.

        try:
            truncateds["__all__"] = all(v for k, v in truncateds.items() if k != "__all__")
            terminateds["__all__"] = all(v for k, v in terminateds.items() if k != "__all__")
        except Exception:
            pass
        return observations, rewards, terminateds, truncateds, infos
    
    def done_function(self, vehicle_id: str):
        """Respect config-driven termination (keep detection but avoid forced termination)."""
        return super().done_function(vehicle_id)

    def reset(self, **kwargs):
        try:
            cfg = getattr(self, 'config', {}) or {}
            map_cfg = cfg.get('map_config', {}) or {}
            debug = bool(cfg.get('debug', False))
            if debug:
                lane_num = map_cfg.get('lane_num', None)
                lane_width = map_cfg.get('lane_width', None)
                agent_cfgs = cfg.get('agent_configs', {}) or {}
                print(f"[SpawnDebug] lane_num={lane_num}, lane_width={lane_width}, agents={len(agent_cfgs)}")
                for aid, ac in agent_cfgs.items():
                    sli = ac.get('spawn_lane_index', None)
                    slong = ac.get('spawn_longitude', None)
                    slat = ac.get('spawn_lateral', None)
                    lane_idx = None
                    if isinstance(sli, (list, tuple)) and len(sli) == 3:
                        lane_idx = sli[2]
                    print(f"[SpawnDebug] {aid}: lane_index={lane_idx}, tuple={sli}, longitude={slong}, lateral={slat}")
                    if lane_num is not None and lane_idx is not None:
                        try:
                            ln = int(lane_num)
                            li = int(lane_idx)
                            if not (0 <= li < ln):
                                print(f"[SpawnDebug] WARNING: {aid} lane_index {li} out of range [0,{ln-1}]")
                        except Exception:
                            pass
        except Exception:
            pass
        return super().reset(**kwargs)
