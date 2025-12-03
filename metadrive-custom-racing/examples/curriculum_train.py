"""
Curriculum Learning for Multi-Agent Racing

Trains agents progressively through increasing difficulty:
1. Straight line basics (50k)
2. Gentle curves (100k)
3. Complex track (150k)
4. Multi-agent racing (200k)

Usage:
    python curriculum_train.py --agent_name Alek
"""
import os
import sys
import argparse
from pathlib import Path

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

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False
    WandbCallback = None


class SingleAgentWrapper(gym.Wrapper):
    """Wrap multi-agent env for single-agent training."""

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
                # Other agents stationary
                actions[aid] = [0.0, 0.0]

        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.env.step(actions)

        obs = obs_dict[self.agent_id]
        reward = reward_dict[self.agent_id]
        terminated = terminated_dict[self.agent_id]
        truncated = truncated_dict[self.agent_id]
        info = info_dict[self.agent_id]

        return obs, reward, terminated, truncated, info


# ==================== CURRICULUM PHASES ====================

def make_phase1_env():
    """Phase 1: Straight line - learn basic throttle and lane keeping."""
    env_config = {
        'num_agents': 2,
        'use_render': False,
        'map_config': {
            'lane_num': 3,
            'lane_width': 8.0,
        },
        'start_seed': 100,  # Different seed for straight track
        'crash_vehicle_done': False,
        'crash_object_done': False,
        'out_of_road_done': False,
        'boundary_training_mode': True,
        'random_spawn_lane_index': True,
        'horizon': 500,  # Shorter episodes for Phase 1
        'vehicle_config': {
            'max_speed_km_h': 120,
        }
    }
    base = MultiAgentCustomSpeedwayEnv(env_config)
    env = SingleAgentWrapper(base, "agent0")
    env = Monitor(env)
    return env


def make_phase2_env():
    """Phase 2: Gentle curves - learn steering."""
    env_config = {
        'num_agents': 2,
        'use_render': False,
        'map_config': {
            'lane_num': 3,
            'lane_width': 8.0,
        },
        'start_seed': 200,  # Different seed for gentle curves
        'crash_vehicle_done': False,
        'crash_object_done': False,
        'out_of_road_done': False,
        'boundary_training_mode': True,
        'random_spawn_lane_index': True,
        'horizon': 1000,  # Medium episodes
        'vehicle_config': {
            'max_speed_km_h': 120,
        }
    }
    base = MultiAgentCustomSpeedwayEnv(env_config)
    env = SingleAgentWrapper(base, "agent0")
    env = Monitor(env)
    return env


def make_phase3_env():
    """Phase 3: Complex track - master the full speedway."""
    env_config = {
        'num_agents': 2,
        'use_render': False,
        'map_config': {
            'lane_num': 3,
            'lane_width': 8.0,
        },
        'start_seed': 42,  # Original complex track
        'crash_vehicle_done': False,
        'crash_object_done': False,
        'out_of_road_done': False,
        'boundary_training_mode': True,
        'random_spawn_lane_index': True,
        'horizon': 1500,  # Full episodes
        'vehicle_config': {
            'max_speed_km_h': 120,
        }
    }
    base = MultiAgentCustomSpeedwayEnv(env_config)
    env = SingleAgentWrapper(base, "agent0")
    env = Monitor(env)
    return env


def make_phase4_env(opponent_model_path):
    """Phase 4: Multi-agent racing - compete against trained opponent."""
    # TODO: Implement multi-agent training with opponent
    return make_phase3_env()  # For now, continue solo


# ==================== TRAINING PHASES ====================

def train_phase(phase_num, phase_name, make_env_fn, timesteps,
                agent_name, checkpoint_dir, previous_model=None, wandb_run=None):
    """Train a single curriculum phase."""

    print("\n" + "="*70)
    print(f"PHASE {phase_num}: {phase_name}")
    print("="*70)
    print(f"Timesteps: {timesteps:,}")
    print(f"Previous model: {previous_model if previous_model else 'None (starting fresh)'}")
    print("="*70 + "\n")

    # Create environment
    env = DummyVecEnv([make_env_fn])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    print("[OK] Environment created")

    # Create or load model
    if previous_model and os.path.exists(previous_model):
        print(f"[LOAD] Loading from previous phase: {previous_model}")
        model = PPO.load(previous_model, env=env)

        # Load VecNormalize stats
        vecnorm_path = previous_model.replace('.zip', '_vecnorm.pkl')
        if os.path.exists(vecnorm_path):
            print(f"[LOAD] Loading VecNormalize stats: {vecnorm_path}")
            env = VecNormalize.load(vecnorm_path, env)
            env.training = True
            env.norm_reward = False
    else:
        print("[NEW] Creating new model")
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            ent_coef=0.05,  # Higher exploration
            policy_kwargs=dict(
                net_arch=[256, 256, 128],
                activation_fn=th.nn.ReLU,
            )
        )

    # Setup checkpoint callback
    phase_checkpoint_dir = os.path.join(checkpoint_dir, f"phase{phase_num}")
    os.makedirs(phase_checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000, timesteps // 5),  # 5 checkpoints per phase
        save_path=phase_checkpoint_dir,
        name_prefix=f"{agent_name}_phase{phase_num}",
    )

    # Setup wandb callback
    callbacks = [checkpoint_callback]
    if WANDB_AVAILABLE and WandbCallback:
        wandb_callback = WandbCallback(
            model_save_path=phase_checkpoint_dir,
            verbose=2,
        )
        callbacks.append(wandb_callback)
        print("[OK] Wandb logging enabled")

    print(f"[TRAIN] Starting Phase {phase_num} training...")

    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        reset_num_timesteps=(previous_model is None),  # Reset if starting fresh
        progress_bar=True,
    )

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, f"{agent_name}_phase{phase_num}_complete.zip")
    model.save(final_model_path)
    print(f"[SAVE] Phase {phase_num} complete: {final_model_path}")

    # Save VecNormalize
    vecnorm_path = final_model_path.replace('.zip', '_vecnorm.pkl')
    env.save(vecnorm_path)
    print(f"[SAVE] VecNormalize stats: {vecnorm_path}")

    env.close()

    return final_model_path


def main():
    parser = argparse.ArgumentParser(description='Curriculum learning for racing agents')
    parser.add_argument('--agent_name', type=str, required=True, help='Agent name (e.g., Alek, Saegan)')
    parser.add_argument('--start_phase', type=int, default=1, help='Start from phase (1-4)')
    parser.add_argument('--wandb', action='store_true', default=True, help='Use wandb logging')

    args = parser.parse_args()

    # Setup paths
    checkpoint_dir = os.path.join(ROOT, 'curriculum_checkpoints', args.agent_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\n" + "="*70)
    print(f"CURRICULUM LEARNING: {args.agent_name}")
    print("="*70)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Starting from phase: {args.start_phase}")
    print("="*70)

    # Initialize wandb
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project="metadrive-speedway",
            name=f"{args.agent_name}_curriculum",
            config={
                "agent": args.agent_name,
                "curriculum": True,
            }
        )
        print("[OK] Wandb initialized\n")

    previous_model = None

    # Phase 1: Straight line basics
    if args.start_phase <= 1:
        previous_model = train_phase(
            phase_num=1,
            phase_name="Straight Line Basics",
            make_env_fn=make_phase1_env,
            timesteps=50000,
            agent_name=args.agent_name,
            checkpoint_dir=checkpoint_dir,
            previous_model=None,
        )

    # Phase 2: Gentle curves
    if args.start_phase <= 2:
        if args.start_phase == 2 and previous_model is None:
            # Resume from Phase 1
            previous_model = os.path.join(checkpoint_dir, f"{args.agent_name}_phase1_complete.zip")

        previous_model = train_phase(
            phase_num=2,
            phase_name="Gentle Curves",
            make_env_fn=make_phase2_env,
            timesteps=100000,
            agent_name=args.agent_name,
            checkpoint_dir=checkpoint_dir,
            previous_model=previous_model,
        )

    # Phase 3: Complex track
    if args.start_phase <= 3:
        if args.start_phase == 3 and previous_model is None:
            # Resume from Phase 2
            previous_model = os.path.join(checkpoint_dir, f"{args.agent_name}_phase2_complete.zip")

        previous_model = train_phase(
            phase_num=3,
            phase_name="Complex Track Master",
            make_env_fn=make_phase3_env,
            timesteps=150000,
            agent_name=args.agent_name,
            checkpoint_dir=checkpoint_dir,
            previous_model=previous_model,
        )

    # Phase 4: Multi-agent racing
    if args.start_phase <= 4:
        if args.start_phase == 4 and previous_model is None:
            # Resume from Phase 3
            previous_model = os.path.join(checkpoint_dir, f"{args.agent_name}_phase3_complete.zip")

        train_phase(
            phase_num=4,
            phase_name="Multi-Agent Racing",
            make_env_fn=lambda: make_phase4_env(previous_model),
            timesteps=200000,
            agent_name=args.agent_name,
            checkpoint_dir=checkpoint_dir,
            previous_model=previous_model,
        )

    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()

    print("\n" + "="*70)
    print("[DONE] CURRICULUM TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinal trained model: {checkpoint_dir}/{args.agent_name}_phase4_complete.zip")
    print("\nNext steps:")
    print("1. Test the trained agent: python test_single_agent.py")
    print("2. Race against other agents: python race_side_by_side.py")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
