"""
Quick local training test with detailed logging - run for 70k steps to debug.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import numpy as np

from environments.multi_agent_custom_speedway_env import MultiAgentCustomSpeedwayEnv

try:
    import wandb
    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False


class DetailedMetricsCallback(BaseCallback):
    """Log detailed metrics every episode."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_speeds = []
        self.episode_boundaries = []

    def _on_step(self) -> bool:
        # Collect info from all environments
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                # Episode finished
                ep_reward = info['episode']['r']
                ep_length = info['episode']['l']

                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                # Log detailed stats every 10 episodes
                if len(self.episode_rewards) % 10 == 0:
                    recent_rewards = self.episode_rewards[-10:]
                    recent_lengths = self.episode_lengths[-10:]

                    stats = {
                        'episode/reward_mean': np.mean(recent_rewards),
                        'episode/reward_std': np.std(recent_rewards),
                        'episode/reward_min': np.min(recent_rewards),
                        'episode/reward_max': np.max(recent_rewards),
                        'episode/reward_median': np.median(recent_rewards),
                        'episode/length_mean': np.mean(recent_lengths),
                        'episode/length_std': np.std(recent_lengths),
                        'episode/num_episodes': len(self.episode_rewards),
                    }

                    print(f"\n[STATS] Episode {len(self.episode_rewards)} Stats (last 10):")
                    print(f"   Reward: {stats['episode/reward_mean']:.1f} ± {stats['episode/reward_std']:.1f}")
                    print(f"           [{stats['episode/reward_min']:.1f} to {stats['episode/reward_max']:.1f}]")
                    print(f"   Length: {stats['episode/length_mean']:.1f} ± {stats['episode/length_std']:.1f}")

                    if WANDB_AVAILABLE and wandb.run:
                        wandb.log(stats, step=self.num_timesteps)

        return True


class SingleAgentWrapper(gym.Wrapper):
    """Wrap multi-agent env to expose single agent for training."""

    def __init__(self, env, agent_id: str = "agent0"):
        super().__init__(env)
        self.agent_id = agent_id

        sample_obs_space = list(env.observation_space.spaces.values())[0]
        sample_act_space = list(env.action_space.spaces.values())[0]

        self.observation_space = sample_obs_space
        self.action_space = sample_act_space

    def reset(self, **kwargs):
        obs_dict = self.env.reset(**kwargs)
        if isinstance(obs_dict, tuple):
            obs_dict, info = obs_dict
            return obs_dict[self.agent_id], info.get(self.agent_id, {})
        return obs_dict[self.agent_id], {}

    def step(self, action):
        actions = {}
        actual_agent_ids = list(self.env.action_space.spaces.keys())

        for aid in actual_agent_ids:
            if aid == self.agent_id:
                actions[aid] = action
            else:
                # Make other agents stationary
                actions[aid] = [0.0, 0.0]

        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.env.step(actions)

        obs = obs_dict[self.agent_id]
        reward = reward_dict[self.agent_id]
        terminated = terminated_dict[self.agent_id]
        truncated = truncated_dict[self.agent_id]
        info = info_dict[self.agent_id]

        return obs, reward, terminated, truncated, info


def make_env():
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
    base = MultiAgentCustomSpeedwayEnv(env_config)
    env = SingleAgentWrapper(base, "agent0")
    env = Monitor(env)
    return env


def main():
    print("\n" + "="*70)
    print("QUICK TRAINING TEST - 70K STEPS")
    print("="*70)

    # Initialize wandb
    if WANDB_AVAILABLE:
        wandb.init(
            project="metadrive-speedway",
            name="quick_test_70k",
            config={
                "timesteps": 70000,
                "test": True,
            }
        )
        print("[OK] Wandb initialized")

    # Create environment
    env = DummyVecEnv([make_env])
    print("[OK] Environment created")

    # Create model
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        ent_coef=0.05,  # Increased exploration
        tensorboard_log="./logs/",
    )
    print("[OK] Model created")

    # Train with detailed callback
    callback = DetailedMetricsCallback()

    print("\n[TRAIN] Starting training...")
    print("   Watch for:")
    print("   - Rewards should improve from -15k toward positive")
    print("   - If stuck at -150k+, agent is hitting boundaries constantly")
    print("\n")

    model.learn(
        total_timesteps=70000,
        callback=callback,
        progress_bar=True,
    )

    print("\n" + "="*70)
    print("[DONE] Training complete!")
    print("="*70)

    # Final stats
    if callback.episode_rewards:
        print(f"\nFinal stats ({len(callback.episode_rewards)} episodes):")
        print(f"   Reward mean: {np.mean(callback.episode_rewards):.1f}")
        print(f"   Reward std:  {np.std(callback.episode_rewards):.1f}")
        print(f"   Reward min:  {np.min(callback.episode_rewards):.1f}")
        print(f"   Reward max:  {np.max(callback.episode_rewards):.1f}")

    # Save model
    model.save("quick_test_70k")
    print("\n[SAVE] Model saved to quick_test_70k.zip")

    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    main()
