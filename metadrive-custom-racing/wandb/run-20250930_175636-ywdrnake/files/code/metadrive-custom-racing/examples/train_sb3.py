"""
Training scaffold using Stable-Baselines3 (PPO) for MetaDrive env.
This script checks for SB3 and runs a very short training loop as a scaffold.
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
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

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
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_speeds = []
    
    def _on_step(self) -> bool:
        # Log step-level metrics
        if WANDB_AVAILABLE and wandb.run is not None:
            # Get info from the last step
            infos = self.locals.get('infos', [])
            if infos:
                # Extract metrics from environment info
                speeds = [info.get('speed', 0) for info in infos if 'speed' in info]
                if speeds:
                    wandb.log({
                        "racing/step_avg_speed": np.mean(speeds),
                        "racing/step_max_speed": np.max(speeds),
                    }, step=self.num_timesteps)
        
        return True
    
    def _on_rollout_end(self) -> None:
        # Log episode-level metrics
        if WANDB_AVAILABLE and wandb.run is not None:
            # Get episode statistics
            ep_info_buffer = getattr(self.model, 'ep_info_buffer', None)
            if ep_info_buffer and len(ep_info_buffer) > 0:
                ep_rewards = [ep_info['r'] for ep_info in ep_info_buffer]
                ep_lengths = [ep_info['l'] for ep_info in ep_info_buffer]
                
                wandb.log({
                    "racing/episode_reward_mean": np.mean(ep_rewards),
                    "racing/episode_reward_std": np.std(ep_rewards),
                    "racing/episode_length_mean": np.mean(ep_lengths),
                    "racing/episode_count": len(ep_rewards),
                }, step=self.num_timesteps)


class _ResetNoKwargs(gym.Wrapper):
    """Env wrapper that ignores seed/options kwargs in reset (MetaDrive compat)."""
    def __init__(self, env):
        super().__init__(env)

    def reset(self, *, seed=None, options=None):
        # Ignore seed/options to avoid MetaDrive assertion path; delegate to base reset
        return self.env.reset()


def make_env(track_name: str = 'custom_speedway', seed: int | None = None) -> Callable[[], object]:
    def _init():
        base = create_racing_environment(track_name, use_render=False, start_seed=seed)
        env = _ResetNoKwargs(base)
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description='Train PPO on MetaDrive custom racing.')
    # Paths and track
    parser.add_argument('--track', type=str, default='custom_speedway', help='Track name (from assets/track_configs)')
    parser.add_argument('--results-dir', type=str, default=None, help='Directory to save models/logs (default: ../results)')
    parser.add_argument('--tb-subdir', type=str, default='tensorboard', help='TensorBoard subdirectory name')

    # Training budget and seeds
    parser.add_argument('--timesteps', type=int, default=200_000, help='Total training timesteps')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num-envs', type=int, default=1, help='Number of parallel envs (>=2 uses SubprocVecEnv)')

    # PPO hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-steps', type=int, default=2048)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-range', type=float, default=0.2)

    # Normalization and eval/checkpoint cadence
    parser.add_argument('--vecnorm', action='store_true', help='Enable VecNormalize on observations')
    parser.add_argument('--eval-freq', type=int, default=5000, help='Eval frequency (timesteps)')
    parser.add_argument('--eval-episodes', type=int, default=5, help='Episodes per eval')
    parser.add_argument('--no-eval', action='store_true', help='Disable evaluation during training')
    parser.add_argument('--checkpoint-freq', type=int, default=5000, help='Model checkpoint frequency (timesteps)')

    # Resume training options
    parser.add_argument('--resume-from', type=str, default=None, help='Path to a saved SB3 model (.zip) to resume from')
    parser.add_argument('--resume-if-exists', action='store_true', help='Auto-resume from results/ppo_<track>.zip if present')
    
    # Weights & Biases options
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='metadrive-racing', help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='Wandb entity/team name')
    parser.add_argument('--wandb-name', type=str, default=None, help='Wandb run name (auto-generated if not provided)')

    args = parser.parse_args()
    
    # Initialize Weights & Biases if requested
    wandb_run = None
    if args.wandb and WANDB_AVAILABLE:
        run_name = args.wandb_name or f"ppo_{args.track}_{args.timesteps}steps"
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "algorithm": "PPO",
                "track": args.track,
                "timesteps": args.timesteps,
                "num_envs": args.num_envs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "n_steps": args.n_steps,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "clip_range": args.clip_range,
                "vecnorm": args.vecnorm,
                "seed": args.seed,
                "eval_freq": args.eval_freq,
                "checkpoint_freq": args.checkpoint_freq,
            },
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        print(f"Weights & Biases run: {wandb_run.name}")
    elif args.wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Install with: pip install wandb")

    # Paths
    results_dir = args.results_dir or os.path.join(os.path.dirname(__file__), '..', 'results')
    tb_log = os.path.join(results_dir, args.tb_subdir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(tb_log, exist_ok=True)

    # Vectorized env
    if args.num_envs > 1:
        env_fns = [make_env(args.track, seed=args.seed + i) for i in range(args.num_envs)]
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv([make_env(args.track, seed=args.seed)])

    # Optional normalization
    if args.vecnorm:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    # Optional evaluation env (single)
    eval_callback = None
    if not args.no_eval:
        eval_env = DummyVecEnv([make_env(args.track, seed=args.seed + 10_000)])
        if args.vecnorm:
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
            # sync running stats so evaluation uses same normalization
            try:
                if isinstance(vec_env, VecNormalize):
                    eval_env.obs_rms = vec_env.obs_rms
            except Exception:
                pass

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=results_dir,
            log_path=results_dir,
            eval_freq=max(1, args.eval_freq),
            n_eval_episodes=max(1, args.eval_episodes),
            deterministic=True,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // max(1, args.num_envs)),
        save_path=results_dir,
        name_prefix=f'checkpoint_{args.track}',
    )

    # PPO model (new or resumed)
    resume_path = args.resume_from
    if resume_path is None and args.resume_if_exists:
        candidate = os.path.join(results_dir, f'ppo_{args.track}.zip')
        if os.path.exists(candidate):
            resume_path = candidate

    if resume_path and os.path.exists(resume_path):
        print(f'Resuming from model: {resume_path}')
        model = PPO.load(resume_path, env=vec_env)
        # If normalization was requested, also try to load stats
        if args.vecnorm:
            vn_path = os.path.join(results_dir, f'vecnorm_{args.track}.pkl')
            if os.path.exists(vn_path) and isinstance(vec_env, VecNormalize):
                try:
                    print(f'Loading VecNormalize stats from: {vn_path}')
                    loaded = VecNormalize.load(vn_path, vec_env)
                    # Replace reference so model uses loaded stats
                    vec_env = loaded
                    model.set_env(vec_env)
                except Exception as e:
                    print('Warning: failed to load VecNormalize stats:', e)
    else:
        model = PPO(
            'MlpPolicy',
            vec_env,
            verbose=1,
            tensorboard_log=tb_log,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            seed=args.seed,
        )

    callbacks = [checkpoint_callback]
    if eval_callback is not None:
        callbacks.insert(0, eval_callback)
    
    # Add wandb callback if enabled
    if wandb_run and WANDB_AVAILABLE:
        wandb_callback = WandbCallback(
            gradient_save_freq=1000,
            model_save_path=f"models/{wandb_run.id}",
            verbose=2,
        )
        racing_callback = RacingMetricsCallback(verbose=1)
        callbacks.extend([wandb_callback, racing_callback])
    
    # When resuming, continue counting timesteps
    reset_steps = False if resume_path else True
    model.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=reset_steps)

    # Save model
    save_path = os.path.join(results_dir, f'ppo_{args.track}.zip')
    model.save(save_path)
    print('Model saved to', save_path)

    # Save VecNormalize stats if enabled
    if args.vecnorm:
        try:
            env_for_saving = model.get_env()
            if isinstance(env_for_saving, VecNormalize):
                vec_path = os.path.join(results_dir, f'vecnorm_{args.track}.pkl')
                env_for_saving.save(vec_path)
                print('Saved VecNormalize stats to', vec_path)
        except Exception:
            pass
    
    # Finish wandb run
    if wandb_run:
        # Log final model as artifact
        artifact = wandb.Artifact(f"ppo_{args.track}_model", type="model")
        artifact.add_file(save_path)
        wandb_run.log_artifact(artifact)
        wandb.finish()
        print("Weights & Biases run completed")


if __name__ == '__main__':
    main()
