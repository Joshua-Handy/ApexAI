r"""
Test a single agent trained on custom speedway in the correct environment.

This script loads an agent trained on MultiAgentCustomSpeedwayEnv and tests it
in the same environment (wrapped for single-agent play).

Usage:
    python examples\test_single_agent.py --model results\results_agent_Alek\Alek_custom_speedway.zip --episodes 1
"""
import os
import sys
import argparse

# Add src to path
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym

from environments.multi_agent_custom_speedway_env import MultiAgentCustomSpeedwayEnv


class SingleAgentWrapper(gym.Wrapper):
    """Wrap multi-agent env to expose single agent for testing."""

    def __init__(self, env, agent_id: str = "agent0"):
        super().__init__(env)
        self.agent_id = agent_id

        # Get observation and action spaces
        sample_obs_space = list(env.observation_space.spaces.values())[0]
        sample_act_space = list(env.action_space.spaces.values())[0]

        self.observation_space = sample_obs_space
        self.action_space = sample_act_space

    def reset(self, **kwargs):
        obs_dict, info_dict = self.env.reset(**kwargs)
        return obs_dict[self.agent_id], info_dict.get(self.agent_id, {})

    def step(self, action):
        # Only our agent takes action, others take random actions
        actions = {}
        actual_agent_ids = list(self.env.action_space.spaces.keys())

        # Flatten action if needed (remove batch dimension)
        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = action.flatten()

        for aid in actual_agent_ids:
            if aid == self.agent_id:
                actions[aid] = action
            else:
                # Generate random action and ensure it's a 1D array
                random_action = self.env.action_space.spaces[aid].sample()
                if isinstance(random_action, np.ndarray) and random_action.ndim > 1:
                    random_action = random_action.flatten()
                actions[aid] = random_action

        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.env.step(actions)

        obs = obs_dict[self.agent_id]
        reward = reward_dict[self.agent_id]
        terminated = terminated_dict[self.agent_id]
        truncated = truncated_dict[self.agent_id]
        info = info_dict[self.agent_id]

        return obs, reward, terminated, truncated, info


class _ResetNoKwargs(gym.Wrapper):
    """Env wrapper that ignores seed/options kwargs in reset."""
    def __init__(self, env):
        super().__init__(env)

    def reset(self, *, seed=None, options=None):
        return self.env.reset()


def test_agent(model_path: str, vecnorm_path: str = None, episodes: int = 3):
    """Test a trained agent in the custom speedway environment."""

    print(f"\n{'='*70}")
    print(f"🧪 TESTING AGENT: {os.path.basename(model_path)}")
    print(f"{'='*70}\n")

    # Load model
    print(f"📦 Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    model = PPO.load(model_path)
    print(f"   ✅ Model loaded (expects {model.observation_space.shape[0]}D observations)")

    # Create environment matching training
    print(f"\n🏗️  Creating custom speedway environment...")
    env_config = {
        'num_agents': 2,  # Need at least 2 for multi-agent env
        'use_render': True,
        'map_config': {
            'lane_num': 3,
            'lane_width': 8.0,
        },
        'start_seed': 42,
        'crash_vehicle_done': False,
        'crash_object_done': False,
        'out_of_road_done': False,
        'horizon': 1500,
    }

    base_env = MultiAgentCustomSpeedwayEnv(env_config)
    env = SingleAgentWrapper(base_env, agent_id="agent0")
    env = _ResetNoKwargs(env)

    # Wrap in VecNormalize if stats available
    vecnorm_stats = None
    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"   Loading VecNormalize from: {os.path.basename(vecnorm_path)}")
        try:
            import pickle
            with open(vecnorm_path, 'rb') as f:
                vecnorm_stats = pickle.load(f)
            print(f"   ✅ VecNormalize loaded")
        except Exception as e:
            print(f"   ⚠️  Failed to load VecNormalize: {e}")

    print(f"   ✅ Environment ready\n")

    # Test episodes
    for episode in range(episodes):
        print(f"\n{'='*70}")
        print(f"📍 Episode {episode + 1}/{episodes}")
        print(f"{'='*70}")

        obs, _ = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        speeds = []
        done = False

        while not done and episode_steps < 1500:
            # Apply VecNormalize if available
            if vecnorm_stats is not None and hasattr(vecnorm_stats, 'obs_rms'):
                obs_rms = vecnorm_stats.obs_rms
                epsilon = 1e-8
                obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -10, 10)

            # Get action from model
            obs_batch = obs.reshape(1, -1)
            action, _ = model.predict(obs_batch, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_steps += 1

            # Track speed
            vehicle_state = info.get('vehicle_state', {})
            speed = vehicle_state.get('speed', 0.0) * 120.0  # Convert to km/h
            speeds.append(speed)

            # Print progress every 100 steps
            if episode_steps % 100 == 0:
                avg_speed = np.mean(speeds[-100:]) if speeds else 0.0
                print(f"   Step {episode_steps:4d} | Reward: {episode_reward:7.1f} | Avg Speed: {avg_speed:5.1f} km/h")

        # Episode summary
        avg_speed = np.mean(speeds) if speeds else 0.0
        max_speed = np.max(speeds) if speeds else 0.0

        print(f"\n{'─'*70}")
        print(f"📊 Episode {episode + 1} Results:")
        print(f"   Total Reward: {episode_reward:7.1f}")
        print(f"   Steps: {episode_steps}")
        print(f"   Avg Speed: {avg_speed:5.1f} km/h")
        print(f"   Max Speed: {max_speed:5.1f} km/h")
        print(f"   Status: {'✅ Completed' if done else '⏱️ Timeout'}")
        print(f"{'─'*70}")

    env.close()

    print(f"\n{'='*70}")
    print(f"🏁 Testing Complete!")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Test a single agent on custom speedway.')
    parser.add_argument('--model', type=str, required=True, help='Path to model .zip file')
    parser.add_argument('--vecnorm', type=str, default=None, help='Path to VecNormalize .pkl file (optional)')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to test')

    args = parser.parse_args()

    # Auto-detect vecnorm if not provided
    vecnorm_path = args.vecnorm
    if vecnorm_path is None:
        model_dir = os.path.dirname(args.model)
        agent_name = os.path.basename(args.model).replace('_custom_speedway.zip', '').replace('.zip', '')
        vecnorm_candidates = [
            os.path.join(model_dir, f'vecnorm_{agent_name}_custom_speedway.pkl'),
            os.path.join(model_dir, 'vecnorm_custom_speedway.pkl'),
        ]
        for candidate in vecnorm_candidates:
            if os.path.exists(candidate):
                vecnorm_path = candidate
                break

    test_agent(args.model, vecnorm_path, args.episodes)


if __name__ == '__main__':
    main()
