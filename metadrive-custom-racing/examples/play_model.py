"""Play back a saved PPO model in the MetaDrive environment with rendering.

Usage (PowerShell):
    python examples/play_model.py --model ../results/ppo_custom_speedway.zip --track custom_speedway --episodes 5
"""
import os
import sys
import argparse

# Ensure src on path
ROOT = os.path.dirname(__file__)
SRC = os.path.join(os.path.dirname(ROOT), 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from stable_baselines3 import PPO
from environments.multi_agent_custom_speedway_env import MultiAgentCustomSpeedwayEnv


def play(model_path: str, track: str = 'custom_speedway', episodes: int = 5):
    """Play back a trained PPO model with optional VecNormalize stats.

    If VecNormalize stats exist, we will wrap the env in a DummyVecEnv and load
    the normalization so observations are processed the same as during training.
    """
    # Create the SAME environment used during training (MultiAgentCustomSpeedwayEnv with 1 agent)
    base_env = MultiAgentCustomSpeedwayEnv({
        'num_agents': 1,
        'use_render': True,
        'map_config': {
            'lane_num': 3,
            'lane_width': 8.0,
        },
        'start_seed': 42,
        'crash_vehicle_done': False,
        'crash_object_done': False,
        'out_of_road_done': False,
        'boundary_training_mode': True,
        'random_spawn_lane_index': False,
        'horizon': 1500,
        'vehicle_config': {
            'max_speed_km_h': 120,
        },
        'show_terrain': True,
        'show_sidewalk': True,
    })
    # Load model
    model = PPO.load(model_path)

    import numpy as _np

    env = base_env  # Use base multi-agent env directly
    
    print(f"Playing model: {model_path}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    for ep in range(episodes):
        obs_dict, _ = env.reset()
        # Get the single agent's observation (environment creates 'agent0')
        agent_id = list(obs_dict.keys())[0]
        obs = obs_dict[agent_id]
        total_reward = 0.0
        done = False
        step_i = 0

        while not done:
            # Prepare observation for model.predict (add batch dimension)
            obs_in = _np.array(obs).reshape(1, -1)

            action, _ = model.predict(obs_in, deterministic=True)

            # Flatten action if needed
            if isinstance(action, _np.ndarray) and action.ndim > 1:
                action = action.flatten()

            # Step environment with single-agent action dict
            actions = {agent_id: action}
            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(actions)
            
            # Extract single agent's data
            obs = obs_dict[agent_id]
            reward = reward_dict[agent_id]
            terminated = terminated_dict[agent_id]
            truncated = truncated_dict[agent_id]
            done = terminated or truncated
            
            total_reward += float(reward)
            step_i += 1

            # Print some info every 100 steps
            if step_i % 100 == 0:
                info = info_dict.get(agent_id, {})
                vehicle_state = info.get('vehicle_state', {})
                speed_kmh = vehicle_state.get('speed', 0.0) * 120.0
                print(f"  Step {step_i}: Speed={speed_kmh:.1f} km/h, Reward={total_reward:.1f}")

        print(f'Episode {ep+1} reward: {total_reward:.1f}, steps: {step_i}')

    try:
        env.close()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--track', type=str, default='custom_speedway')
    parser.add_argument('--episodes', type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print('Model not found:', args.model)
        return
    play(args.model, track=args.track, episodes=args.episodes)


if __name__ == '__main__':
    main()