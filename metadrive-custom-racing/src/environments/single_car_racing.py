"""Factory to create a single-car MetaDrive environment for training and demos.

This file provides create_racing_environment(track_name, ...) which returns a
MetaDriveEnv configured for single-agent racing using a custom track JSON from
assets/track_configs.
"""
from typing import Optional, Dict, Any
import os

from utils.track_loader import load_track_config
from environments.oval_right_env import SingleAgentOvalEnv


def create_racing_environment(
    track_name: str = 'custom_speedway',
    use_render: bool = False,
    image_observation: bool = False,
    start_seed: int | None = None,
    manual_control: bool = False,
):
    """Create and return a MetaDrive environment for single-car racing.

    Notes:
    - Only pass top-level keys that MetaDrive expects. Avoid passing arbitrary
      `map_config` entries unless you know your MetaDrive version supports them.
    """
    cfg = load_track_config(track_name)

    # If this track specifies a special type, route to the corresponding env
    track_type = cfg.get("type")
    if track_type == "oval_right_pg":
        env = SingleAgentOvalEnv({
            'start_seed': start_seed if start_seed is not None else cfg.get('start_seed', 1000),
            'traffic_density': 0.0,
            'use_render': use_render,
            '_render_mode': 'headless' if not use_render else 'onscreen',
            'manual_control': manual_control,
            'vehicle_config': {
                'show_lidar': True,
                'show_lane_line_detector': True,
                'show_side_detector': True,
            },
            'image_observation': image_observation,
            'num_scenarios': 1,
            'out_of_road_penalty': 0,
            'crash_vehicle_penalty': 0,
            'crash_object_penalty': 0,
            'crash_sidewalk_penalty': 0,
            'out_of_road_done': False,
            'crash_vehicle_done': False,
            'crash_object_done': False,
            'on_continuous_line_done': True,
            'success_reward': 0,
            'map_config': {
                'lane_num': cfg.get('lane_num', 1),
                'lane_width': cfg.get('lane_width', 4.0),
                'exit_length': 30,
            }
        })
        return env

    env_config: Dict[str, Any] = {
        # MetaDrive accepts a 'map' string and 'start_seed'
        'map': cfg.get('map'),
        'start_seed': start_seed if start_seed is not None else cfg.get('start_seed', 1000),
        # Keep traffic off for single-car racing
        'traffic_density': 0.0,
        'use_render': use_render,
        # Use the private key to avoid older public API mismatches
        '_render_mode': 'headless' if not use_render else 'onscreen',
        'manual_control': manual_control,
        'vehicle_config': {
            'show_lidar': True,
            'show_lane_line_detector': False,
            'show_side_detector': False,
        },
        'image_observation': image_observation,
        'num_scenarios': 1,
    }

    # Create the MetaDrive environment (lazy import with local fallback)
    try:
        from metadrive.envs import MetaDriveEnv  # type: ignore
    except ImportError:
        import os, sys
        _here = os.path.dirname(__file__)
        _workspace_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
        _metadrive_src = os.path.join(_workspace_root, "metadrive")
        if _metadrive_src not in sys.path:
            sys.path.insert(0, _metadrive_src)
        from metadrive.envs import MetaDriveEnv  # type: ignore

    env = MetaDriveEnv(env_config)
    return env
