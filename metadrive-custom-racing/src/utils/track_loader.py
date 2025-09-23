"""
Simple loader for custom track JSON files.
Returns a dict compatible with MetaDrive env construction.
"""
import json
import os
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TRACK_DIR = os.path.join(BASE_DIR, "assets", "track_configs")


def load_track_config(name: str) -> Dict[str, Any]:
    """Load track config by filename (without .json) from assets/track_configs."""
    path = os.path.join(TRACK_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Track config not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    # Convert to MetaDrive config keys expected by MetaDriveEnv
    config: Dict[str, Any] = {
        "map": data.get("map", "OOOO"),
        "lane_width": data.get("lane_width", 4.0),
        "lane_num": data.get("lane_num", 1),
        "start_seed": data.get("start_seed", 1000),
        # optional track type to select a special env/map builder
        "type": data.get("type"),
        # keep metadata for reference
        "_meta": {
            "name": data.get("name"),
            "description": data.get("description", ""),
            "target_lap_time": data.get("target_lap_time"),
            "lap_distance": data.get("lap_distance"),
        }
    }

    return config


if __name__ == "__main__":
    # Quick smoke test
    print(load_track_config("custom_speedway"))
