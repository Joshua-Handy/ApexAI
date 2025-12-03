"""
Test ghost mode: verify that crash penalties are -5 instead of -50.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from environments.multi_agent_custom_speedway_env import MultiAgentCustomSpeedwayEnv
import numpy as np


def test_ghost_mode():
    print("="*70)
    print("TESTING GHOST MODE")
    print("="*70)

    # Test with ghost mode ON
    print("\n[Test] Ghost mode ON - crash penalty should be -5")
    env_config = {
        'num_agents': 2,
        'use_render': False,
        'map_config': {'lane_num': 3, 'lane_width': 8.0},
        'start_seed': 42,
        'crash_vehicle_done': False,
        'crash_object_done': False,
        'out_of_road_done': False,
        'boundary_training_mode': True,
        'random_spawn_lane_index': False,  # Spawn together to force crash
        'ghost_mode': True,  # Enable ghost mode!
        'horizon': 1500,
        'vehicle_config': {'max_speed_km_h': 120}
    }

    env = MultiAgentCustomSpeedwayEnv(env_config)
    obs_dict, _ = env.reset()

    # Drive forward to crash into each other
    for step in range(100):
        actions = {aid: [0.0, 1.0] for aid in obs_dict.keys()}  # Full throttle straight
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(actions)

        # Check if any agent crashed
        for aid in obs_dict.keys():
            if info_dict[aid].get('crashed', False):
                crash_penalty = info_dict[aid]['reward_components']['crash_penalty']
                print(f"  [Step {step}] {aid} crashed!")
                print(f"  Crash penalty: {crash_penalty}")
                if crash_penalty == -5.0:
                    print("  [PASS] Ghost mode crash penalty is -5")
                else:
                    print(f"  [FAIL] Expected -5, got {crash_penalty}")
                env.close()
                return

    print("  [WARNING] No crash detected in 100 steps")
    env.close()

    print("\n" + "="*70)
    print("GHOST MODE TEST COMPLETE")
    print("="*70)


if __name__ == '__main__':
    test_ghost_mode()
