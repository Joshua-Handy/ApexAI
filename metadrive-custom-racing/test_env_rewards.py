"""
Quick diagnostic script to test the environment and reward function.
"""
import sys
import os

# Add src to path
ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from environments.multi_agent_custom_speedway_env import MultiAgentCustomSpeedwayEnv
import numpy as np

def test_env():
    print("="*60)
    print("Testing MultiAgentCustomSpeedwayEnv")
    print("="*60)

    # Create environment
    env_config = {
        'num_agents': 2,
        'use_render': False,
        'map_config': {
            'lane_num': 3,
            'lane_width': 8.0,
        },
        'start_seed': 42,
        'crash_vehicle_done': False,
        'crash_object_done': False,
        'out_of_road_done': False,
        'boundary_training_mode': True,
        'random_spawn_lane_index': True,
        'horizon': 1500,
        'vehicle_config': {
            'max_speed_km_h': 120,
        }
    }

    env = MultiAgentCustomSpeedwayEnv(env_config)

    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Reset environment
    obs_dict, info_dict = env.reset()
    print(f"\nAgents: {list(obs_dict.keys())}")

    # Test several steps with different actions
    print("\n" + "="*60)
    print("Testing reward calculation with various actions")
    print("="*60)

    test_cases = [
        ("No action (0,0)", {"agent0": [0.0, 0.0], "agent1": [0.0, 0.0]}),
        ("Full throttle (0,1)", {"agent0": [0.0, 1.0], "agent1": [0.0, 0.0]}),
        ("Turn right + throttle (1,1)", {"agent0": [1.0, 1.0], "agent1": [0.0, 0.0]}),
        ("Turn left + throttle (-1,1)", {"agent0": [-1.0, 1.0], "agent1": [0.0, 0.0]}),
    ]

    for step_num, (description, actions) in enumerate(test_cases):
        print(f"\nStep {step_num+1}: {description}")
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(actions)

        for agent_id in ["agent0"]:
            vehicle = env.agents[agent_id]
            info = info_dict[agent_id]

            print(f"  {agent_id}:")
            print(f"    Speed: {getattr(vehicle, 'speed_km_h', 0):.2f} km/h")
            print(f"    On road: {getattr(vehicle, 'on_lane', False)}")
            print(f"    Position: {getattr(vehicle, 'position', [0,0])}")
            print(f"    Reward: {reward_dict[agent_id]:.2f}")

            if 'reward_components' in info:
                components = info['reward_components']
                print(f"    Reward breakdown:")
                for key, val in components.items():
                    if key != 'total':
                        print(f"      {key}: {val:.2f}")

            print(f"    Terminated: {terminated_dict[agent_id]}")
            print(f"    Truncated: {truncated_dict[agent_id]}")

    # Test a longer episode with constant throttle
    print("\n" + "="*60)
    print("Running 50 steps with constant throttle")
    print("="*60)

    obs_dict, info_dict = env.reset()

    rewards_history = []
    speeds_history = []

    for i in range(50):
        # Agent 0 goes straight, Agent 1 is stationary
        actions = {"agent0": [0.0, 1.0], "agent1": [0.0, 0.0]}
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(actions)

        vehicle = env.agents["agent0"]
        speed = getattr(vehicle, 'speed_km_h', 0)
        reward = reward_dict["agent0"]

        rewards_history.append(reward)
        speeds_history.append(speed)

        if i % 10 == 0 or i < 5:
            print(f"Step {i:3d}: Speed={speed:6.2f} km/h, Reward={reward:8.2f}")

        if terminated_dict["agent0"] or truncated_dict["agent0"]:
            print(f"Episode ended at step {i}")
            break

    print(f"\nSummary:")
    print(f"  Avg Reward: {np.mean(rewards_history):.2f}")
    print(f"  Total Reward: {np.sum(rewards_history):.2f}")
    print(f"  Min Reward: {np.min(rewards_history):.2f}")
    print(f"  Max Reward: {np.max(rewards_history):.2f}")
    print(f"  Final Speed: {speeds_history[-1]:.2f} km/h")

    env.close()
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)

if __name__ == "__main__":
    test_env()
