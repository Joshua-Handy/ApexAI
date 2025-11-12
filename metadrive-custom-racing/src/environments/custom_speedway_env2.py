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


class SingleAgentRaceMap(PGMap):
    """A minimal PGMap that creates: Straight -> Right Curve -> Straight -> Right Curve -> Straight.

    The parameters (lane_num, lane_width) and lengths are modest defaults appropriate for single-lane racing.
    """

    def _generate(self):
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "Map not empty; create fresh map."

        lane_num = self.config.get("lane_num", 4)
        lane_width = self.config.get("lane_width", 40.0)
        self.random_seed=42
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

        # Define a custom layout: (type, params)
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
                block = Straight(block_index, last_block.get_socket(0), self.road_network,self.random_seed,remove_negative_lanes=True)
            else:
                block = Curve(block_index, last_block.get_socket(0), self.road_network, self.random_seed,remove_negative_lanes=True)

            block.construct_from_config(params, parent_node_path, physics_world)
            self.blocks.append(block)
            last_block = block
            block_index += 1

class SingleAgentRaceMapManager(PGMapManager):
    """A map manager that always loads SingleAgentOvalMap once."""
    def reset(self):
        # Clear any previously spawned map
        for obj_id in list(self.spawned_objects.keys()):
            self.destroy_object(obj_id)
        # Spawn new map instance
        config = self.engine.global_config
        new_map = self.spawn_object(SingleAgentRaceMap, map_config=config["map_config"], random_seed=None)
        self.load_map(new_map)


class SingleAgentRaceEnv(MetaDriveEnv):
    """Drop-in MetaDriveEnv that uses SingleAgentOvalMapManager to build a right-only oval."""


    def setup_engine(self):
        super().setup_engine()
        # Replace the default PGMapManager with our fixed oval builder
        self.engine.update_manager("map_manager", SingleAgentRaceMapManager())

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
