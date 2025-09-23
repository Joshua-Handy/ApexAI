"""
Example: run a single-car MetaDrive environment using a custom track config
Requires metadrive to be installed in the environment.
"""
import time
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import setup_logger
import os
import sys

# Ensure project src is on path
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from utils.track_loader import load_track_config
from agents.my_agent import MyAgent


def make_env_from_track(track_name: str = "custom_speedway") -> MetaDriveEnv:
    """Create a MetaDriveEnv using our custom track and no traffic."""
    track_cfg = load_track_config(track_name)

    # Build env_config with keys known to exist in MetaDrive default config
    env_config = {
        "map": track_cfg["map"],
        "start_seed": track_cfg.get("start_seed", 1000),

        # No traffic, single agent
        "num_scenarios": 1,
        "traffic_density": 0.0,
        "accident_prob": 0.0,

        # Vehicle and rendering settings go under existing keys
        "vehicle_config": {
            "show_lidar": True,
            "show_lane_line_detector": True,
            "show_side_detector": True,
        },

        # Rendering and observation - use keys present in MetaDrive defaults
        "use_render": True,
        "_render_mode": "human",
        "window_size": (1200, 800),
        "image_observation": False,
    }

    # Do not set map_config here to avoid MetaDrive's internal map parsing mismatch.
    # We pass the map string via `map` only. If you need per-block control, create
    # a full map config using MetaDrive's map creation utilities.
    env = MetaDriveEnv(env_config)
    return env


def run_demo(track_name: str = "custom_speedway", steps: int = 500):
    setup_logger(debug=False)
    env = make_env_from_track(track_name)

    obs, info = env.reset()
    print(f"Started env on track: {track_name}")

    try:
        agent = MyAgent(target_speed_kmh=80.0)

        for i in range(steps):
            # Get action from our custom agent
            action = agent.get_action(obs)

            out = env.step(action)

            # Support both gym v0.25 (obs, reward, done, info) and v0.26+ (obs, reward, terminated, truncated, info)
            if len(out) == 4:
                obs, reward, done, info = out
                terminated = done
                truncated = False
            else:
                obs, reward, terminated, truncated, info = out

            if i % 50 == 0:
                speed = env.vehicle.speed * 3.6 if hasattr(env, 'vehicle') else 0
                print(f"Step {i}: speed={speed:.1f} km/h, reward={reward:.2f}")

            if terminated or truncated:
                print("Episode done, resetting...")
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("Demo interrupted by user")
    finally:
        env.close()


if __name__ == "__main__":
    run_demo()
