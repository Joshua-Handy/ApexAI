"""
Transfer learning script that explicitly uses PPO.load() and learn() for MetaDrive env.
"""
import os
import sys
import argparse
from typing import Callable
import gymnasium as gym

# Add src to path
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

# Weights & Biases integration
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not installed. Install with: pip install wandb")
    WANDB_AVAILABLE = False

from environments.single_car_racing import create_racing_environment


class RacingMetricsCallback(BaseCallback):
    """Custom callback to log racing-specific metrics to wandb."""
    
    def __init__(self, results_dir, track_name, verbose=0):
        super().__init__(verbose)
        self.results_dir = results_dir
        self.track_name = track_name
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_speeds = []
        self.last_vecnorm_save = 0
        self.vecnorm_save_freq = 100000  # Save vecnorm stats every 100k steps
    
    def _on_step(self) -> bool:
        # Log step-level metrics
        if WANDB_AVAILABLE and wandb.run is not None:
            # Get info from the last step
            infos = self.locals.get('infos', [])
            if infos:
                # Extract metrics from environment info
                speeds = [info.get('speed', 0) for info in infos if 'speed' in info]
                if speeds:
                    # Calculate global step accounting for parallel environments
                    env = self.training_env
                    n_envs = env.num_envs if hasattr(env, 'num_envs') else 1
                    global_step = self.num_timesteps * n_envs
                    
                    wandb.log({
                        "racing/step_avg_speed": np.mean(speeds),
                        "racing/step_max_speed": np.max(speeds),
                        "racing/global_step": global_step,
                    }, step=global_step)
                    
                    # Periodically save vecnorm stats
                    if global_step - self.last_vecnorm_save >= self.vecnorm_save_freq:
                        if isinstance(env, VecNormalize):
                            try:
                                vec_path = os.path.join(self.results_dir, f'vecnorm_{self.track_name}.pkl')
                                env.save(vec_path)
                                print(f'Saved intermediate VecNormalize stats to {vec_path} at step {global_step}')
                                self.last_vecnorm_save = global_step
                            except Exception as e:
                                print(f'Warning: Failed to save intermediate VecNormalize stats at step {global_step}:', e)
        
        return True
    
    def _on_rollout_end(self) -> None:
        # Log episode-level metrics
        if WANDB_AVAILABLE and wandb.run is not None:
            # Get episode statistics
            ep_info_buffer = getattr(self.model, 'ep_info_buffer', None)
            if ep_info_buffer and len(ep_info_buffer) > 0:
                ep_rewards = [ep_info['r'] for ep_info in ep_info_buffer]
                ep_lengths = [ep_info['l'] for ep_info in ep_info_buffer]
                
                # Calculate global step accounting for parallel environments
                env = self.training_env
                n_envs = env.num_envs if hasattr(env, 'num_envs') else 1
                global_step = self.num_timesteps * n_envs
                
                wandb.log({
                    "racing/episode_reward_mean": np.mean(ep_rewards),
                    "racing/episode_reward_std": np.std(ep_rewards),
                    "racing/episode_length_mean": np.mean(ep_lengths),
                    "racing/episode_count": len(ep_rewards),
                }, step=global_step)


def make_env(track_name: str = 'custom_speedway', seed: int | None = None) -> Callable[[], object]:
    def _init():
        base = create_racing_environment(track_name, use_render=False, start_seed=seed)
        env = Monitor(base)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description='Transfer learning with PPO on MetaDrive')
    parser.add_argument('--load-model', type=str, required=True, help='Path to the model to load (.zip)')
    parser.add_argument('--track', type=str, required=True, help='Track to train on')
    parser.add_argument('--timesteps', type=int, default=500_000, help='Training timesteps')
    parser.add_argument('--num-envs', type=int, default=4, help='Number of parallel envs')
    parser.add_argument('--results-dir', type=str, default='results_v2', help='Directory to save models')
    parser.add_argument('--checkpoint-freq', type=int, default=25000, help='Checkpoint frequency')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--wandb-project', type=str, default='metadrive-racingv2')
    parser.add_argument('--wandb-name', type=str, help='W&B run name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    # Initialize W&B
    if args.wandb and WANDB_AVAILABLE:
        wandb_name = args.wandb_name or f"transfer_{args.track}_{args.timesteps}steps"
        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config=vars(args),
            sync_tensorboard=True,
            monitor_gym=True,
        )

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Create vectorized environment
    if args.num_envs > 1:
        env = SubprocVecEnv([make_env(args.track, seed=args.seed + i) for i in range(args.num_envs)])
    else:
        env = DummyVecEnv([make_env(args.track, seed=args.seed)])

    # Load the model
    print(f"Loading model from: {args.load_model}")
    model = PPO.load(args.load_model, env=env)
    print(f"Model loaded successfully. Policy network shape: {[layer.shape for layer in model.policy.parameters()]}")

    # Try to load vecnorm stats from source model's directory
    source_vecnorm = os.path.join(os.path.dirname(args.load_model), f'vecnorm_{args.track}.pkl')
    if not os.path.exists(source_vecnorm):
        # Try alternate naming scheme
        source_vecnorm = os.path.join(os.path.dirname(args.load_model), f'vecnorm_{os.path.splitext(os.path.basename(args.load_model))[0].split("_", 1)[1]}.pkl')
    print(f"Looking for vecnorm at: {source_vecnorm}")
    if os.path.exists(source_vecnorm):
        print(f"Loading VecNormalize stats from: {source_vecnorm}")
        try:
            env = VecNormalize.load(source_vecnorm, env)
            print("VecNormalize stats loaded successfully")
            print(f"Observation scaling - mean: {env.obs_rms.mean}, var: {env.obs_rms.var}")
            env.training = True  # Enable training mode
            model.set_env(env)
        except Exception as e:
            print(f"Error loading vecnorm stats: {e}")
            print("Creating new VecNormalize wrapper")
    else:
        print("No VecNormalize stats found, creating new normalization")
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
        model.set_env(env)

    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback with version suffix if v3
    version_suffix = "_v3" if "v3" in (args.wandb_name or "") else ""
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // args.num_envs),
        save_path=args.results_dir,
        name_prefix=f'checkpoint_{args.track}{version_suffix}',
    )
    callbacks.append(checkpoint_callback)

    # WandB callback
    if args.wandb and WANDB_AVAILABLE:
        wandb_callback = WandbCallback(
            gradient_save_freq=max(1000 // args.num_envs, 1),
            model_save_path=None,
        )
        racing_callback = RacingMetricsCallback(
            results_dir=args.results_dir,
            track_name=args.track,
            verbose=1
        )
        callbacks.extend([wandb_callback, racing_callback])

    # Continue training
    print(f"Starting transfer learning on {args.track} for {args.timesteps} timesteps")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=False  # Continue counting timesteps
    )

    # Save final model and stats
    version_suffix = "_v3" if "v3" in (args.wandb_name or "") else ""
    final_model_path = os.path.join(args.results_dir, f'ppo_{args.track}{version_suffix}.zip')
    model.save(final_model_path)
    print(f"Saved final model to: {final_model_path}")

    if isinstance(env, VecNormalize):
        vecnorm_path = os.path.join(args.results_dir, f'vecnorm_{args.track}{version_suffix}.pkl')
        env.save(vecnorm_path)
        print(f"Saved VecNormalize stats to: {vecnorm_path}")

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()