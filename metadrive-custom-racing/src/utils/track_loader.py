"""
Simple loader for custom track JSON files.
Returns a dict compatible with MetaDrive env construction.
"""
import json
import os
from typing import Dict, Any

# More robust path resolution that works from any working directory
def get_track_config_dir():
    """Get the track configs directory, searching upward from this file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Search upward for the assets/track_configs directory
    for _ in range(5):  # Search up to 5 levels
        parent = os.path.dirname(current_dir)
        if parent == current_dir:  # Reached root
            break
        current_dir = parent
        
        # Check if assets/track_configs exists in this directory
        track_dir = os.path.join(current_dir, "assets", "track_configs")
        if os.path.exists(track_dir):
            return track_dir
    
    # Fallback to original method
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base_dir, "assets", "track_configs")

TRACK_DIR = get_track_config_dir()


def load_track_config(name: str) -> Dict[str, Any]:
    """Load track config by filename (without .json) from assets/track_configs."""
    path = os.path.join(TRACK_DIR, f"{name}.json")
    if not os.path.exists(path):
        # Try alternative search paths
        possible_paths = [
            # Current working directory
            os.path.join(os.getcwd(), "assets", "track_configs", f"{name}.json"),
            # Relative to current working directory
            os.path.join(os.getcwd(), "..", "assets", "track_configs", f"{name}.json"),
            # Search from script location
            os.path.join(os.path.dirname(__file__), "..", "..", "assets", "track_configs", f"{name}.json"),
        ]
        
        for alt_path in possible_paths:
            alt_path = os.path.abspath(alt_path)
            if os.path.exists(alt_path):
                path = alt_path
                break
        else:
            raise FileNotFoundError(f"Track config not found: {path}\nSearched paths: {[path] + possible_paths}")

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
