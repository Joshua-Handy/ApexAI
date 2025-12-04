"""
Train multiple agents on the custom speedway track.

This script allows you to train 2 or more agents (like Alek and Saegan) 
on the same custom speedway track with complex layout.

Usage:
    python train_multi_custom_speedway.py --num-agents 2 --timesteps 500000 --vecnorm
    python train_multi_custom_speedway.py --num-agents 2 --agent-names Alek Saegan --timesteps 500000
"""
import os
import sys
import argparse
from typing import Callable
import gymnasium as gym
import torch as th
import time
from multiprocessing import Process, Queue
import traceback

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

# Weights & Biases integration
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not installed. Install with: pip install wandb")
    WANDB_AVAILABLE = False

from environments.multi_agent_custom_speedway_env import MultiAgentCustomSpeedwayEnv


class SingleAgentWrapper(gym.Wrapper):
    """Wrap multi-agent env to expose single agent for training.

    Supports ghost mode where other agents use loaded models instead of staying stationary.
    """

    def __init__(self, env, agent_id: str, other_agent_models=None):
        """
        Args:
            env: Multi-agent environment
            agent_id: ID of the agent being trained
            other_agent_models: Dict of {agent_id: model} for other agents (ghost mode)
                               If None, other agents stay stationary
        """
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

        # Create action dict - only need action for agents that exist
        actions = {}

        # Get actual agent IDs from the current spaces (use action_space as source of truth)
        actual_agent_ids = list(self.env.action_space.spaces.keys())

        for aid in actual_agent_ids:
            if aid == self.agent_id:
                # Our agent uses the provided action
                actions[aid] = action
            elif aid in self.other_agent_models:
                # Other agent uses its loaded model (ghost mode)
                obs = self._other_obs_cache.get(aid)
                if obs is not None:
                    other_action, _ = self.other_agent_models[aid].predict(obs, deterministic=False)
                    actions[aid] = other_action
                else:
                    actions[aid] = [0.0, 0.0]
            else:
                # Make other agents STATIONARY (default behavior)
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

        # CRITICAL FIX: Manually enforce horizon since MetaDrive doesn't do it properly
        if self._episode_step >= self._horizon:
            truncated = True
            info['TimeLimit.truncated'] = True

        return obs, reward, terminated, truncated, info


class _ResetNoKwargs(gym.Wrapper):
    """Env wrapper that ignores seed/options kwargs in reset (MetaDrive compat)."""
    def __init__(self, env):
        super().__init__(env)

    def reset(self, *, seed=None, options=None):
        return self.env.reset()


def make_env(track_name: str = 'custom_speedway', seed: int = None, agent_id: int = 0, num_agents: int = 2,
             ghost_mode: bool = False, other_agent_models=None) -> Callable[[], object]:
    def _init():
        # Create multi-agent environment with REAL RACETRACK settings
        # Always create at least 2 agents for multi-agent env (MetaDrive requirement)
        actual_num_agents = max(num_agents, 2)

        env_config = {
            'num_agents': actual_num_agents,
            'use_render': False,
            'map_config': {
                'lane_num': 3,
                'lane_width': 8.0,
            },
            'start_seed': seed if seed else 42,

            # TRAINING: Allow crashes so agents learn to avoid them
            'crash_vehicle_done': False,  # Don't end episode - let them learn
            'crash_object_done': False,
            'out_of_road_done': False,
            'boundary_training_mode': True,  # Don't terminate on boundaries during training - big penalty but let agent recover

            # IMPORTANT: Spawn agents far apart so stationary agent doesn't block learning agent
            'random_spawn_lane_index': True,  # Each agent spawns in different lane

            # Ghost mode for multi-agent racing
            'ghost_mode': ghost_mode,  # Small crash penalty if True

            'horizon': 1500,  # Racing duration

            # Max speed configuration for racing
            'vehicle_config': {
                'max_speed_km_h': 120,
            }
        }
        base = MultiAgentCustomSpeedwayEnv(env_config)
        # Wrap to expose single agent for PPO training
        # Pass other agent models if ghost mode is enabled
        env = SingleAgentWrapper(base, f"agent{agent_id}", other_agent_models=other_agent_models)
        env = _ResetNoKwargs(env)
        env = Monitor(env)
        return env
    return _init


def train_single_agent(
    agent_id: int,
    agent_name: str,
    track: str,
    timesteps: int,
    results_dir: str,
    vecnorm: bool,
    learning_rate: float,
    batch_size: int,
    n_steps: int,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    seed: int,
    eval_freq: int,
    checkpoint_freq: int,
    no_eval: bool,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str,
    ghost_mode: bool = False,
    other_agent_model_path: str = None,
    result_queue: Queue = None
):
    """Train a single agent. Can be run in parallel."""
    try:
        print(f"\n{'='*60}")
        print(f"🏁 Starting training for {agent_name} (Agent {agent_id})")
        print(f"{'='*60}\n")
        
        # Initialize wandb for this agent
        wandb_run = None
        if wandb_enabled and WANDB_AVAILABLE:
            try:
                run_name = f"{agent_name}_{track}"
                wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=run_name,
                    config={
                        "algorithm": "PPO",
                        "agent_name": agent_name,
                        "agent_id": agent_id,
                        "track": track,
                        "timesteps": timesteps,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "n_steps": n_steps,
                        "gamma": gamma,
                        "gae_lambda": gae_lambda,
                        "clip_range": clip_range,
                        "vecnorm": vecnorm,
                        "seed": seed + agent_id,
                    },
                    sync_tensorboard=True,
                    monitor_gym=True,
                    save_code=True,
                )
                print(f"✅ Weights & Biases initialized for {agent_name}")
            except Exception as e:
                print(f"⚠️  Wandb initialization failed: {e}")
                print(f"   Continuing training without wandb...")
                wandb_run = None
                wandb_enabled = False

        # Create agent-specific directories
        agent_results_dir = os.path.join(results_dir, f'results_agent_{agent_name}')
        agent_tb_log = os.path.join(agent_results_dir, 'tensorboard')
        os.makedirs(agent_results_dir, exist_ok=True)
        os.makedirs(agent_tb_log, exist_ok=True)

        # Load other agent model if ghost mode is enabled
        other_agent_models = None
        if ghost_mode and other_agent_model_path:
            try:
                print(f"[Ghost Mode] Loading other agent model from {other_agent_model_path}")
                other_model = PPO.load(other_agent_model_path)
                # Assuming 2 agents: agent0 and agent1
                other_agent_id = 1 if agent_id == 0 else 0
                other_agent_models = {f"agent{other_agent_id}": other_model}
                print(f"[Ghost Mode] Loaded model for agent{other_agent_id}")
            except Exception as e:
                print(f"[Warning] Failed to load other agent model: {e}")
                print("[Warning] Continuing with stationary other agent")

        # Create vectorized environment
        vec_env = DummyVecEnv([make_env(track, seed=seed + agent_id, agent_id=agent_id, num_agents=2,
                                         ghost_mode=ghost_mode, other_agent_models=other_agent_models)])

        if vecnorm:
            # IMPORTANT: Disable obs normalization - MetaDrive already normalizes to [0,1]
            # Normalizing pre-normalized data destroys the signal!
            vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

        # Disable evaluation for multi-agent training (causes issues)
        eval_callback = None

        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, checkpoint_freq),
            save_path=agent_results_dir,
            name_prefix=f'checkpoint_{agent_name}',
        )

        # Create PPO model with improved hyperparameters
        model = PPO(
            'MlpPolicy',
            vec_env,
            verbose=1,
            tensorboard_log=agent_tb_log,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            seed=seed + agent_id,
            ent_coef=0.3,  # VERY HIGH: Force extreme exploration so agent discovers movement!
            vf_coef=0.5,    # Value function coefficient
            max_grad_norm=0.5,
            n_epochs=10,     # Reduced from 15 to prevent overfitting
            target_kl=0.03,  # Increased from 0.01 to allow more learning per update
            policy_kwargs=dict(
                net_arch=[256, 256, 128],
                activation_fn=th.nn.ReLU,
                ortho_init=True,
            )
        )

        # Setup callbacks
        callbacks = [checkpoint_callback]
        if eval_callback is not None:
            callbacks.insert(0, eval_callback)
        
        if wandb_run and WANDB_AVAILABLE and wandb_enabled:
            try:
                wandb_callback = WandbCallback(
                    gradient_save_freq=1000,
                    model_save_path=None,
                    verbose=2,
                )
                callbacks.append(wandb_callback)
            except Exception as e:
                print(f"⚠️  Wandb callback failed: {e}")
                print(f"   Continuing training without wandb callback...")

        # Train the model
        print(f"🚀 Training {agent_name} for {timesteps:,} timesteps...")
        model.learn(total_timesteps=timesteps, callback=callbacks)

        # Save model
        save_path = os.path.join(agent_results_dir, f'{agent_name}_{track}.zip')
        model.save(save_path)
        print(f"💾 Model saved to {save_path}")

        # Save VecNormalize stats if enabled
        if vecnorm:
            try:
                env_for_saving = model.get_env()
                if isinstance(env_for_saving, VecNormalize):
                    vec_path = os.path.join(agent_results_dir, f'vecnorm_{agent_name}_{track}.pkl')
                    env_for_saving.save(vec_path)
                    print(f"💾 VecNormalize stats saved to {vec_path}")
            except Exception as e:
                print(f"⚠️  Failed to save VecNormalize stats: {e}")

        # Close environments properly
        try:
            vec_env.close()
            if eval_callback and hasattr(eval_callback, 'eval_env'):
                eval_callback.eval_env.close()
        except Exception as e:
            print(f"⚠️  Error closing environments: {e}")

        # Finish wandb run
        if wandb_run:
            try:
                artifact = wandb.Artifact(f"{agent_name}_model", type="model")
                artifact.add_file(save_path, name="model.zip")
                wandb_run.log_artifact(artifact)
                print(f"☁️  Model uploaded to wandb for {agent_name}")
            except Exception as e:
                print(f"⚠️  Failed to upload to wandb: {e}")
            try:
                wandb.finish()
            except Exception as e:
                print(f"⚠️  Wandb finish failed: {e}")
                # Force cleanup
                try:
                    import atexit
                    atexit._run_exitfuncs()
                except:
                    pass

        print(f"\n✅ Training complete for {agent_name}!\n")
        
        if result_queue:
            result_queue.put({'agent_name': agent_name, 'success': True, 'save_path': save_path})
        
        return save_path

    except Exception as e:
        print(f"\n❌ Error training {agent_name}: {e}")
        traceback.print_exc()
        if result_queue:
            result_queue.put({'agent_name': agent_name, 'success': False, 'error': str(e)})
        raise


def main():
    parser = argparse.ArgumentParser(description='Train multiple agents on custom speedway track.')
    
    # Agent configuration
    parser.add_argument('--num-agents', type=int, default=2, help='Number of agents to train')
    parser.add_argument('--agent-names', type=str, nargs='+', default=None, 
                       help='Names for agents (e.g., --agent-names Alek Saegan). If not provided, uses agent_0, agent_1, etc.')
    
    # Track and paths
    parser.add_argument('--track', type=str, default='custom_speedway', help='Track name')
    parser.add_argument('--results-dir', type=str, default=None, 
                       help='Base results directory (default: ../results)')
    
    # Training configuration
    parser.add_argument('--timesteps', type=int, default=500_000, help='Total training timesteps per agent')
    parser.add_argument('--seed', type=int, default=0, help='Random seed base (each agent gets seed+agent_id)')
    parser.add_argument('--parallel', action='store_true', help='Train agents in parallel (default: sequential)')
    
    # PPO hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-steps', type=int, default=4096)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-range', type=float, default=0.2)
    
    # Training options
    parser.add_argument('--vecnorm', action='store_true', help='Enable VecNormalize')
    parser.add_argument('--eval-freq', type=int, default=5000, help='Eval frequency')
    parser.add_argument('--checkpoint-freq', type=int, default=50000, help='Checkpoint frequency')
    parser.add_argument('--no-eval', action='store_true', help='Disable evaluation')

    # Ghost mode for multi-agent training
    parser.add_argument('--ghost-mode', action='store_true',
                       help='Enable ghost mode: small crash penalty (-5) so agents can race together during training')
    parser.add_argument('--other-agent-model', type=str, default=None,
                       help='Path to other agent model for ghost racing (optional)')
    
    # Weights & Biases (enabled by default)
    parser.add_argument('--wandb', action='store_true', default=True, help='Enable wandb logging (default: True)')
    parser.add_argument('--no-wandb', action='store_false', dest='wandb', help='Disable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='metadrive-speedway', help='Wandb project')
    parser.add_argument('--wandb-entity', type=str, default=None, help='Wandb entity')
    
    args = parser.parse_args()
    
    # Setup results directory
    results_dir = args.results_dir or os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup agent names - if names provided, use their count as num_agents
    if args.agent_names:
        agent_names = args.agent_names
        args.num_agents = len(agent_names)  # Override num_agents with actual count
    else:
        agent_names = [f'agent_{i}' for i in range(args.num_agents)]
    
    print("\n" + "="*70)
    print("🏎️  MULTI-AGENT CUSTOM SPEEDWAY TRAINING")
    print("="*70)
    print(f"Track: {args.track}")
    print(f"Number of agents: {args.num_agents}")
    print(f"Agent names: {', '.join(agent_names)}")
    print(f"Timesteps per agent: {args.timesteps:,}")
    print(f"Mode: {'Parallel' if args.parallel else 'Sequential'}")
    print(f"VecNormalize: {args.vecnorm}")
    print(f"Wandb: {args.wandb}")
    print("="*70 + "\n")
    
    # Confirm if training many timesteps
    if args.timesteps > 100000:
        response = input(f"⚠️  Train {args.num_agents} agents for {args.timesteps:,} timesteps each? [y/N]: ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
    
    start_time = time.time()
    
    if args.parallel and args.num_agents > 1:
        print(f"\n🔀 Training {args.num_agents} agents in PARALLEL...\n")
        
        # Create a queue to collect results
        result_queue = Queue()
        
        # Create processes for each agent
        processes = []
        for i, agent_name in enumerate(agent_names):
            p = Process(
                target=train_single_agent,
                args=(
                    i, agent_name, args.track, args.timesteps, results_dir,
                    args.vecnorm, args.learning_rate, args.batch_size, args.n_steps,
                    args.gamma, args.gae_lambda, args.clip_range, args.seed,
                    args.eval_freq, args.checkpoint_freq, args.no_eval,
                    args.wandb, args.wandb_project, args.wandb_entity,
                    args.ghost_mode, args.other_agent_model, result_queue
                )
            )
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        print("\n" + "="*70)
        print("🏁 PARALLEL TRAINING COMPLETE")
        print("="*70)
        for result in results:
            if result['success']:
                print(f"✅ {result['agent_name']}: {result['save_path']}")
            else:
                print(f"❌ {result['agent_name']}: {result.get('error', 'Unknown error')}")
        
    else:
        print(f"\n➡️  Training {args.num_agents} agents SEQUENTIALLY...\n")
        
        trained_models = []
        for i, agent_name in enumerate(agent_names):
            try:
                save_path = train_single_agent(
                    i, agent_name, args.track, args.timesteps, results_dir,
                    args.vecnorm, args.learning_rate, args.batch_size, args.n_steps,
                    args.gamma, args.gae_lambda, args.clip_range, args.seed,
                    args.eval_freq, args.checkpoint_freq, args.no_eval,
                    args.wandb, args.wandb_project, args.wandb_entity,
                    args.ghost_mode, args.other_agent_model, None
                )
                trained_models.append((agent_name, save_path))
            except KeyboardInterrupt:
                print(f"\n⚠️  Training interrupted by user after {agent_name}")
                break
            except Exception as e:
                print(f"\n❌ Failed to train {agent_name}: {e}")
                continue
        
        print("\n" + "="*70)
        print("🏁 SEQUENTIAL TRAINING COMPLETE")
        print("="*70)
        for agent_name, save_path in trained_models:
            print(f"✅ {agent_name}: {save_path}")
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print(f"\n⏱️  Total training time: {hours}h {minutes}m {seconds}s")
    print("\n" + "="*70)
    print("🎉 All agents trained! You can now:")
    print(f"   1. Watch agents: python examples/play_model.py --model results/results_agent_<name>/<name>_{args.track}.zip")
    print(f"   2. Race them in multi-agent env (create custom script)")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
