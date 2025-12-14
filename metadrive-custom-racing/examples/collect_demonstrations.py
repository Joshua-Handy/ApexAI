"""
Collect demonstration data from a trained agent for imitation learning.

This script runs a trained agent in the environment and records all
(observation, action) pairs for later behavioral cloning training.

Usage:
    python collect_demonstrations.py --agent agent_0v2 --episodes 50
    python collect_demonstrations.py --agent agent_0v2 --episodes 100 --output demos.pkl
"""
import os
import sys
import argparse
import pickle
from pathlib import Path

# Add src to path
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments.multi_agent_custom_speedway_curve_env import MultiAgentCustomSpeedwayCurveEnv


class SingleAgentWrapper(gym.Wrapper):
    """Wrap multi-agent env to expose single agent for training."""

    def __init__(self, env, agent_id: str, other_agent_models=None):
        super().__init__(env)
        self.agent_id = agent_id
        self._episode_step = 0
        self._horizon = env.config.get('horizon', 1500)
        self.other_agent_models = other_agent_models or {}
        self._other_obs_cache = {}

        # Get observation and action spaces for this specific agent
        sample_obs_space = list(env.observation_space.spaces.values())[0]
        sample_act_space = list(env.action_space.spaces.values())[0]

        self.observation_space = sample_obs_space
        self.action_space = sample_act_space

    def reset(self, **kwargs):
        self._episode_step = 0
        self._other_obs_cache = {}
        obs_dict, info_dict = self.env.reset(**kwargs)

        # Cache observations for other agents
        for aid in obs_dict.keys():
            if aid != self.agent_id:
                self._other_obs_cache[aid] = obs_dict[aid]

        return obs_dict[self.agent_id], info_dict.get(self.agent_id, {})

    def step(self, action):
        self._episode_step += 1

        # Create action dict
        actions = {}
        actual_agent_ids = list(self.env.action_space.spaces.keys())

        for aid in actual_agent_ids:
            if aid == self.agent_id:
                actions[aid] = action
            elif aid in self.other_agent_models:
                obs = self._other_obs_cache.get(aid)
                if obs is not None:
                    other_action, _ = self.other_agent_models[aid].predict(obs, deterministic=False)
                    actions[aid] = other_action
                else:
                    actions[aid] = [0.0, 0.0]
            else:
                # Make other agents stationary
                actions[aid] = [0.0, 0.0]

        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.env.step(actions)

        # Update observation cache for other agents
        for aid in obs_dict.keys():
            if aid != self.agent_id:
                self._other_obs_cache[aid] = obs_dict[aid]

        obs = obs_dict[self.agent_id]
        reward = reward_dict[self.agent_id]
        terminated = terminated_dict[self.agent_id]
        truncated = truncated_dict[self.agent_id]
        info = info_dict[self.agent_id]

        # Manually enforce horizon
        if self._episode_step >= self._horizon:
            truncated = True
            info['TimeLimit.truncated'] = True

        return obs, reward, terminated, truncated, info


class _ResetNoKwargs(gym.Wrapper):
    """Env wrapper that ignores seed/options kwargs in reset (MetaDrive compat)."""
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset()


def load_agent_model(agent_name, results_dir):
    """Load trained agent model."""

    # Find agent directory
    agent_dir = os.path.join(results_dir, f'results_agent_{agent_name}')
    if not os.path.exists(agent_dir):
        raise FileNotFoundError(f"Agent directory not found: {agent_dir}")

    # Find model file
    model_patterns = [
        f'{agent_name}_custom_speedway.zip',
        f'ppo_custom_speedway.zip',
        f'best_model.zip',
    ]

    model_path = None
    for pattern in model_patterns:
        candidate = os.path.join(agent_dir, pattern)
        if os.path.exists(candidate):
            model_path = candidate
            break

    if not model_path:
        raise FileNotFoundError(f"No model found in {agent_dir}")

    print(f"[OK] Loading model: {model_path}")
    model = PPO.load(model_path)

    # Try to load VecNormalize stats
    vecnorm_path = os.path.join(agent_dir, f'vecnorm_custom_speedway.pkl')
    vecnorm = None
    if os.path.exists(vecnorm_path):
        try:
            def make_dummy():
                import gymnasium as gym
                return gym.make('CartPole-v1')
            dummy_venv = DummyVecEnv([make_dummy])
            vecnorm = VecNormalize.load(vecnorm_path, dummy_venv)
            dummy_venv.close()
            print(f"[OK] Loaded VecNormalize stats")
        except:
            print(f"[WARN] VecNormalize not loaded, continuing without normalization")

    return model, vecnorm


def create_environment():
    """Create the training environment."""

    env_config = {
        'num_agents': 2,  # MetaDrive requires at least 2
        'use_render': False,
        'map_config': {
            'lane_num': 2,
            'lane_width': 8.0,
        },
        'start_seed': 42,

        # STRICT MODE: Match training
        'crash_vehicle_done': True,
        'crash_object_done': True,
        'out_of_road_done': True,

        'random_spawn_lane_index': False,
        'ghost_mode': False,

        'horizon': 1500,
        'vehicle_config': {
            'max_speed_km_h': 120,
        },

        'show_terrain': True,
        'show_sidewalk': True,
    }

    base_env = MultiAgentCustomSpeedwayCurveEnv(env_config)
    env = SingleAgentWrapper(base_env, 'agent0')  # Wrap for single agent
    env = _ResetNoKwargs(env)

    return env


def collect_demonstrations(model, vecnorm, env, num_episodes, verbose=True):
    """Collect demonstrations from a trained agent.

    Returns:
        demonstrations: dict with 'observations' and 'actions' arrays
    """

    observations = []
    actions = []

    print(f"\n{'='*70}")
    print(f"COLLECTING DEMONSTRATIONS")
    print(f"{'='*70}")
    print(f"Episodes: {num_episodes}")
    print(f"VecNorm: {'Yes' if vecnorm else 'No'}")
    print(f"{'='*70}\n")

    successful_episodes = 0
    total_steps = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_steps = 0
        episode_reward = 0.0

        while not done:
            # Store raw observation
            observations.append(obs.copy())

            # Apply VecNormalize if available
            obs_normalized = obs.copy()
            if vecnorm is not None and hasattr(vecnorm, 'obs_rms'):
                obs_rms = vecnorm.obs_rms
                epsilon = 1e-8
                obs_normalized = np.clip(
                    (obs_normalized - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon),
                    -10, 10
                )

            # Get action from model
            action, _ = model.predict(obs_normalized, deterministic=True)

            # Store action
            actions.append(action.copy())

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_steps += 1
            total_steps += 1

        # Episode complete
        if episode_steps > 50:  # Count as successful if lasted more than 50 steps
            successful_episodes += 1

        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Steps: {episode_steps} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Total demonstrations: {len(observations)}")

    # Convert to numpy arrays
    demonstrations = {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'metadata': {
            'num_episodes': num_episodes,
            'total_steps': total_steps,
            'successful_episodes': successful_episodes,
            'obs_shape': observations[0].shape,
            'action_shape': actions[0].shape,
        }
    }

    print(f"\n{'='*70}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total demonstrations: {len(observations)}")
    print(f"Successful episodes: {successful_episodes}/{num_episodes}")
    print(f"Observation shape: {demonstrations['metadata']['obs_shape']}")
    print(f"Action shape: {demonstrations['metadata']['action_shape']}")
    print(f"{'='*70}\n")

    return demonstrations


def main():
    parser = argparse.ArgumentParser(description='Collect demonstrations from trained agent')
    parser.add_argument('--agent', type=str, required=True, help='Agent name (e.g., agent_0v2)')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes to collect')
    parser.add_argument('--results-dir', type=str, default=None, help='Results directory')
    parser.add_argument('--output', type=str, default=None, help='Output file (default: demonstrations_{agent}.pkl)')

    args = parser.parse_args()

    # Setup paths
    results_dir = args.results_dir or os.path.join(os.path.dirname(__file__), '..', 'results')
    output_file = args.output or f'demonstrations_{args.agent}.pkl'
    output_path = os.path.join(os.path.dirname(__file__), output_file)

    # Load model
    print(f"Loading agent: {args.agent}")
    model, vecnorm = load_agent_model(args.agent, results_dir)

    # Create environment
    print(f"Creating environment...")
    env = create_environment()

    # Collect demonstrations
    demonstrations = collect_demonstrations(model, vecnorm, env, args.episodes)

    # Save demonstrations
    print(f"Saving demonstrations to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(demonstrations, f)

    env.close()

    print(f"\n[OK] Demonstrations saved successfully!")
    print(f"   Use this file for imitation learning training:")
    print(f"   python train_imitation.py --demonstrations {output_file} --agent F16")


if __name__ == '__main__':
    main()
