"""
PGBlock map generator for MetaDrive.

This module provides a simple API to build a block-by-block map description and
export it as a MetaDrive-friendly config JSON. It generates a `map` string and a
`map_config` dict that MetaDrive can accept via `map_config` (some MetaDrive
versions expect specific formats; use with caution and adapt to your installed
MetaDrive version).

This generator is intentionally conservative: it creates the `map` string and a
lightweight `block_sequence` describing each block. For advanced usage, you can
instantiate real pgblock objects inside MetaDrive and append them to a road
network.
"""
import json
import os
from typing import List, Dict, Any


DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets", "track_configs"
)

"""
PGBlock map generator for MetaDrive.

This module provides a simple API to build a block-by-block map description and
export it as a MetaDrive-friendly config JSON. It generates a `map` string and a
`map_config` dict that MetaDrive can accept via `map_config` (some MetaDrive
versions expect specific formats; use with caution and adapt to your installed
MetaDrive version).

This generator is intentionally conservative: it creates the `map` string and a
lightweight `block_sequence` describing each block. For advanced usage, you can
instantiate real pgblock objects inside MetaDrive and append them to a road
network.
"""
import json
import os
from typing import List, Dict, Any


DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets", "track_configs"
)


class BlockSpec:
    """Specification for a single block."""

    def __init__(self, block_type: str, length: int = 50, radius: int = 30, angle: int = 90, direction: int = 1):
        self.block_type = block_type  # 'S' or 'C' or 'R' etc
        self.length = length
        self.radius = radius
        self.angle = angle
        self.direction = direction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.block_type,
            "length": self.length,
            "radius": self.radius,
            "angle": self.angle,
            "direction": self.direction,
        }


class PGBlockMapBuilder:
    """Build a simple sequence of blocks and export as config."""

    def __init__(self, lane_width: float = 4.0, lane_num: int = 1, start_seed: int = 1000):
        self.blocks: List[BlockSpec] = []
        self.lane_width = lane_width
        self.lane_num = lane_num
        self.start_seed = start_seed

    def add_straight(self, length: int = 80) -> 'PGBlockMapBuilder':
        self.blocks.append(BlockSpec('S', length=length))
        return self

    def add_curve(self, radius: int = 40, angle: int = 90, direction: int = 1) -> 'PGBlockMapBuilder':
        self.blocks.append(BlockSpec('C', radius=radius, angle=angle, direction=direction))
        return self

    def add_long_straight(self, length: int = 200) -> 'PGBlockMapBuilder':
        return self.add_straight(length=length)

    def build_map_string(self) -> str:
        """Return a simple map string from the block sequence."""
        # Map char mapping: use 'S' for straight and 'C' for curve; repeat
        return ''.join([b.block_type for b in self.blocks])

    def to_config(self, name: str) -> Dict[str, Any]:
        cfg = {
            "name": name,
            "map": self.build_map_string(),
            "map_config": {
                "blocks": [b.to_dict() for b in self.blocks],
                "lane_width": self.lane_width,
                "lane_num": self.lane_num,
            },
            "start_seed": self.start_seed,
            "description": f"PGBlock-generated map ({name})",
        }
        return cfg

    def save(self, name: str, out_dir: str = DEFAULT_OUTPUT_DIR) -> str:
        os.makedirs(out_dir, exist_ok=True)
        cfg = self.to_config(name)
        filename = os.path.join(out_dir, f"{name.lower().replace(' ', '_')}.json")
        with open(filename, 'w') as f:
            json.dump(cfg, f, indent=2)
        return filename


def example_builder():
    b = PGBlockMapBuilder(lane_width=4.0, lane_num=1)
    b.add_long_straight(200).add_curve(radius=50, angle=90, direction=1)
    b.add_long_straight(150).add_curve(radius=50, angle=90, direction=1)
    return b


if __name__ == '__main__':
    builder = example_builder()
    path = builder.save('pgblock_speedway')
    print('Saved map to:', path)
