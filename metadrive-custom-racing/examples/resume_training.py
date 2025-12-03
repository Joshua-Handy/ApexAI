"""
Resume training from a checkpoint.

This script loads a saved model and continues training for additional timesteps.

Usage:
    # Resume from final model
    python examples\resume_training.py --model results\results_agent_Alek_v2\Alek_v2_custom_speedway.zip --timesteps 500000

    # Resume from checkpoint
    python examples\resume_training.py --model results\results_agent_Alek_v2\checkpoint_Alek_v2_500000_steps.zip --timesteps 500000
"""
import os
import sys
import argparse
from typing import Callable

# Add src to path
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from environments.multi_agent_custom_speedway_env import MultiAgentCustomSpeedwayEnv

# Import wandb if available
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SingleAgentWrapper(gym.Wrapper):
    """Wrap multi-agent env to expose single agent for training."""

    def __init__(self, env, agent_id: str):
        super().__init__(env)
        self.agent_id = agent_id

        sample_obs_space = list(env.observation_space.spaces.values())[0]
        sample_act_space = list(env.action_space.spaces.values())[0]

        self.observation_space = sample_obs_space
        self.action_space = sample_act_space

    def reset(self, **kwargs):
        obs_dict, info_dict = self.env.reset(**kwargs)
        return obs_dict[self.agent_id], info_dict.get(self.agent_id, {})

    def step(self, action):
        actions = {}
        actual_agent_ids = list(self.env.action_space.spaces.keys())

        for aid in actual_agent_ids:
            if aid == self.agent_id:
                actions[aid] = action
            else:
                actions[aid] = self.env.action_space.spaces[aid].sample()

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


def make_env(agent_id: int = 0, seed: int = 42) -> Callable[[], object]:
    def _init():
        env_config = {
            'num_agents': 2,
            'use_render': False,
            'map_config': {
                'lane_num': 3,
                'lane_width': 8.0,
            },
            'start_seed': seed,
            'crash_vehicle_done': False,
            'crash_object_done': False,
            'out_of_road_done': False,
            'horizon': 1500,
            'vehicle_config': {
                'max_speed_km_h': 120,
            }
        }
        base = MultiAgentCustomSpeedwayEnv(env_config)
        env = SingleAgentWrapper(base, f"agent{agent_id}")
        env = _ResetNoKwargs(env)
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description='Resume training from a checkpoint.')

    # Required
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--timesteps', type=int, required=True, help='Additional timesteps to train')

    # Optional
    parser.add_argument('--vecnorm', type=str, default=None, help='Path to VecNormalize file (optional, auto-detected)')
    parser.add_argument('--checkpoint-freq', type=int, default=100000, help='Checkpoint save frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb', action='store_true', default=True, help='Enable wandb logging')
    parser.add_argument('--no-wandb', action='store_false', dest='wandb', help='Disable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='metadrive-speedway-resume', help='Wandb project')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return

    # Extract agent name from path
    model_dir = os.path.dirname(args.model)
    model_filename = os.path.basename(args.model)

    # Try to extract agent name
    if 'results_agent_' in model_dir:
        agent_name = model_dir.split('results_agent_')[-1].split(os.sep)[0]
    else:
        agent_name = model_filename.replace('.zip', '').replace('checkpoint_', '')

    print(f"\n{'='*70}")
    print(f"📦 RESUMING TRAINING: {agent_name}")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Additional timesteps: {args.timesteps:,}")
    print(f"{'='*70}\n")

    # Auto-detect VecNormalize if not provided
    vecnorm_path = args.vecnorm
    if vecnorm_path is None:
        vecnorm_candidates = [
            os.path.join(model_dir, f'vecnorm_{agent_name}_custom_speedway.pkl'),
            os.path.join(model_dir, 'vecnorm_custom_speedway.pkl'),
        ]
        for candidate in vecnorm_candidates:
            if os.path.exists(candidate):
                vecnorm_path = candidate
                print(f"✅ Auto-detected VecNormalize: {os.path.basename(vecnorm_path)}")
                break

    # Create environment
    vec_env = DummyVecEnv([make_env(agent_id=0, seed=args.seed)])

    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"📊 Loading VecNormalize stats...")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = True  # Enable training mode
        vec_env.norm_reward = False

    # Load model
    print(f"🔄 Loading model...")
    model = PPO.load(args.model, env=vec_env)
    print(f"✅ Model loaded successfully!")

    # Setup wandb
    if args.wandb and WANDB_AVAILABLE:
        try:
            wandb.init(
                project=args.wandb_project,
                name=f"{agent_name}_resume",
                config={
                    "resumed_from": args.model,
                    "additional_timesteps": args.timesteps,
                },
                sync_tensorboard=True,
                resume="allow",
            )
            print(f"✅ Wandb initialized")
        except Exception as e:
            print(f"⚠️  Wandb failed: {e}")

    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq),
        save_path=model_dir,
        name_prefix=f'checkpoint_{agent_name}',
    )

    # Continue training
    print(f"\n🚀 Resuming training for {args.timesteps:,} additional steps...")
    print(f"{'='*70}\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback],
        reset_num_timesteps=False,  # Continue from current timestep count
    )

    # Save final model
    final_model_path = os.path.join(model_dir, f'{agent_name}_custom_speedway.zip')
    model.save(final_model_path)
    print(f"\n💾 Model saved to: {final_model_path}")

    # Save VecNormalize stats
    if isinstance(vec_env, VecNormalize):
        vecnorm_save_path = os.path.join(model_dir, f'vecnorm_{agent_name}_custom_speedway.pkl')
        vec_env.save(vecnorm_save_path)
        print(f"💾 VecNormalize saved to: {vecnorm_save_path}")

    vec_env.close()

    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()

    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
