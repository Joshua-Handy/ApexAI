"""
Single-agent environment that builds a right-turn-only oval map at runtime using PGBlocks.

This avoids relying on ambiguous 'map' strings for curve direction and keeps MetaDrive core untouched.
"""
from __future__ import annotations

from typing import Dict, Any

try:
    from metadrive.envs import MetaDriveEnv  # type: ignore
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
    from metadrive.envs import MetaDriveEnv  # type: ignore
    from metadrive.component.map.pg_map import PGMap  # type: ignore
    from metadrive.component.pgblock.first_block import FirstPGBlock  # type: ignore
    from metadrive.component.pgblock.straight import Straight  # type: ignore
    from metadrive.component.pgblock.curve import Curve  # type: ignore
    from metadrive.component.pg_space import Parameter  # type: ignore
    from metadrive.constants import PGLineType  # type: ignore
    from metadrive.manager.pg_map_manager import PGMapManager  # type: ignore


class SingleAgentOvalMap(PGMap):
    """A minimal PGMap that creates: Straight -> Right Curve -> Straight -> Right Curve -> Straight.

    The parameters (lane_num, lane_width) and lengths are modest defaults appropriate for single-lane racing.
    """

    def _generate(self):
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "Map is not empty; create a fresh map to build the oval"

        lane_num = self.config.get("lane_num", 1)
        lane_width = self.config.get("lane_width", 20.0)  # Match config file lane width

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
                1,
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

class SingleAgentOvalMapManager(PGMapManager):
    """A map manager that always loads SingleAgentOvalMap once."""

    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(SingleAgentOvalMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "Expected exactly one map in this manager"
            _map = list(self.spawned_objects.values())[0]
        self.load_map(_map)


class SingleAgentOvalEnv(MetaDriveEnv):
    """Drop-in MetaDriveEnv that uses SingleAgentOvalMapManager to build a right-only oval."""


    def setup_engine(self):
        super().setup_engine()
        # Replace the default PGMapManager with our fixed oval builder
        self.engine.update_manager("map_manager", SingleAgentOvalMapManager())

    def _is_lap_done(self):
        # Ignore lap completion: always return False
        return False

    def _get_success_reward(self):
        # Ignore lap completion reward
        return 0.0
    
    def _is_arrive_destination(self, vehicle):
        # Ignore destination arrival - always return False
        return False
    
    def done_function(self, vehicle_id: str):
        # Override done function to check crashes and lane line collisions
        vehicle = self.vehicles[vehicle_id]
        
        # Check for solid crashes
        solid_crash = vehicle.crash_vehicle or vehicle.crash_object
        
        # Check for lane line collisions (both white continuous and yellow broken lines)
        white_line_collision = vehicle.on_white_continuous_line
        yellow_line_collision = vehicle.on_yellow_continuous_line or vehicle.on_broken_line
        lane_line_collision = white_line_collision or yellow_line_collision
        
        # Episode ends if there's a solid crash OR lane line collision
        done = solid_crash or lane_line_collision
        
        done_info = {
            "crash_vehicle": vehicle.crash_vehicle,
            "crash_object": vehicle.crash_object,
            "crash_sidewalk": False,
            "out_of_road": False,
            "arrive_dest": False,
            "max_step": False,
            "lane_line_collision": lane_line_collision,
            "white_line_collision": white_line_collision,
            "yellow_line_collision": yellow_line_collision,
        }
        return done, done_info
