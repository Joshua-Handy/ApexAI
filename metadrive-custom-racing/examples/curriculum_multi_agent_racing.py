"""
MetaDrive Multi-Agent Racing with MAPPO/PPO

This script implements a comprehensive multi-agent racing environment where 4 agents
learn to race competitively on a right-turn oval track using curriculum learning.

Key Features:
- 4 agents with proper racing grid formation (evenly spaced)
- Curriculum learning: Safety → Control → Speed → Racing
- Per-agent done logic (crashed agents marked done, others continue)
- Custom reward function emphasizing progress, safety, and lap completion
- Shared policy learning for coordinated behavior
- No automatic respawning - realistic racing consequences
- Lidar perception with smooth action updates
- Stable learning with gradual speed progression

Training Phases:
1. SAFETY: Low speed, high safety rewards (learn track boundaries)
2. CONTROL: Medium speed, smooth steering rewards (learn vehicle control)
3. SPEED: Higher speed rewards (learn efficient racing lines)
4. RACING: Full competition with overtaking and positioning rewards
"""
import os
import sys
import argparse
import numpy as np
import pickle
import time
import gymnasium as gym
from typing import Dict, List, Tuple, Optional
from collections import deque

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch
from metadrive.component.pgblock.first_block import FirstPGBlock

try:
    from environments.multi_agent_oval_right_env import MultiAgentOvalEnv
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    print("Warning: MultiAgentOvalEnv not available, using standard MetaDrive")
    from metadrive.envs import MultiAgentMetaDrive
    MULTI_AGENT_AVAILABLE = False

# Weights & Biases integration
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not installed")
    WANDB_AVAILABLE = False


class CurriculumProgressCallback(BaseCallback):
    """Callback to track curriculum learning progress and phase transitions."""
    
    def __init__(self, agent_id: str, curriculum_manager, verbose: int = 1):
        super().__init__(verbose)
        self.agent_id = agent_id
        self.curriculum_manager = curriculum_manager
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_crashes = []
        self.episode_completions = []
        self.phase_performance = []
        
    def _on_step(self) -> bool:
        return True
        
    def _on_rollout_end(self) -> None:
        """Track performance and check for curriculum phase transitions."""
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if 'r' in ep_info and 'l' in ep_info:
                episode_reward = ep_info['r']
                episode_length = ep_info['l']

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Keep only recent episodes for phase evaluation
                if len(self.episode_rewards) > 50:
                    self.episode_rewards = self.episode_rewards[-50:]
                    self.episode_lengths = self.episode_lengths[-50:]
                
                # Calculate performance metrics
                recent_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward
                recent_length = np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) >= 10 else episode_length
                
                for agent_id in self.curriculum_manager.agent_ids:
                    self.curriculum_manager.update_performance(agent_id, recent_reward, recent_length)
                
                # Log progress
                current_phase = self.curriculum_manager.get_current_phase()
                phase_progress = self.curriculum_manager.get_phase_progress()

                print(
                    f"timesteps={self.num_timesteps:,} "
                    f"agent={self.agent_id} "
                    f"phase={current_phase} "
                    f"phase_progress={phase_progress:.1%} "
                    f"episode_reward={episode_reward:.2f} "
                    f"episode_length={int(episode_length)} "
                    f"recent10_reward={recent_reward:.2f} "
                    f"recent10_length={recent_length:.1f}"
                )
                
                # Log to wandb if available
                if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'record'):
                    self.model.logger.record(f"{self.agent_id}/episode_reward", episode_reward)
                    self.model.logger.record(f"{self.agent_id}/episode_length", episode_length)
                    self.model.logger.record(f"{self.agent_id}/recent_reward", recent_reward)
                    self.model.logger.record(f"{self.agent_id}/curriculum_phase", self.curriculum_manager.phase_index)
                    self.model.logger.record(f"{self.agent_id}/phase_progress", phase_progress)
                    self.model.logger.dump(self.num_timesteps)


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, agent_id: str, patience: int = 5, min_improvement: float = 1.0, verbose: int = 1):
        super().__init__(verbose)
        self.agent_id = agent_id
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_mean_reward = -np.inf
        self.wait = 0
        self.stopped_epoch = 0

    def _on_step(self) -> bool:
        return True


class TrainingStatsCallback(BaseCallback):
    """Print concise training stats: timesteps, entropy, recent rewards."""
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.last_print_ts = 0
        self.last_logged_update = -1

    def _on_step(self) -> bool:
        try:
            lv = getattr(self.model.logger, 'name_to_value', {})
            upd = lv.get('train/n_updates', None)
            if upd is not None:
                try:
                    upd_i = int(upd)
                    if upd_i != self.last_logged_update:
                        self._print_train_metrics(lv)
                        self.last_logged_update = upd_i
                except Exception:
                    pass
        except Exception:
            pass
        return True

    def _on_rollout_end(self) -> None:
        entropy_mean = float('nan')
        try:
            import torch
            buf = getattr(self.model, 'rollout_buffer', None)
            if buf is not None and hasattr(buf, 'observations'):
                obs = buf.observations
                if isinstance(obs, np.ndarray):
                    flat = obs.reshape(-1, obs.shape[-1])
                    obs_t = torch.as_tensor(flat, device=self.model.device)
                else:
                    obs_t = obs
                dist = self.model.policy.get_distribution(obs_t)
                ent = dist.entropy()
                if hasattr(ent, 'mean'):
                    entropy_mean = float(ent.mean().detach().cpu().numpy())
        except Exception:
            pass

        ep_r = None
        ep_l = None
        recent_mean_r = None
        try:
            if len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                ep_r = ep_info.get('r', None)
                ep_l = ep_info.get('l', None)
                recent = [ep.get('r', 0.0) for ep in list(self.model.ep_info_buffer)[-10:] if isinstance(ep, dict)]
                if recent:
                    recent_mean_r = float(np.mean(recent))
        except Exception:
            pass

        lr = None
        clip = None
        try:
            lr_sched = getattr(self.model, 'lr_schedule', None)
            if callable(lr_sched):
                lr = float(lr_sched(self.num_timesteps))
            else:
                lr = float(getattr(self.model, 'learning_rate', np.nan))
        except Exception:
            pass
        try:
            cr = getattr(self.model, 'clip_range', None)
            if callable(cr):
                clip = float(cr(self.num_timesteps))
            elif cr is not None:
                clip = float(cr)
        except Exception:
            pass

        train_vals = {}
        try:
            lv = getattr(self.model.logger, 'name_to_value', {})
            def g(k):
                v = lv.get(k, None)
                try:
                    return float(v)
                except Exception:
                    return None
            train_vals = {
                'train_entropy_loss': g('train/entropy_loss'),
                'train_policy_gradient_loss': g('train/policy_gradient_loss'),
                'train_value_loss': g('train/value_loss'),
                'approx_kl': g('train/approx_kl'),
                'clip_fraction': g('train/clip_fraction'),
                'train_loss': g('train/loss'),
                'explained_variance': g('train/explained_variance'),
                'policy_std': g('train/std'),
                'n_updates': g('train/n_updates'),
                'clip_range_train': g('train/clip_range'),
            }
        except Exception:
            pass

        try:
            parts = [
                f"timesteps={self.num_timesteps:,}",
                f"policy_entropy={entropy_mean:.3f}",
            ]
            if lr is not None and not np.isnan(lr):
                parts.append(f"learning_rate={lr:.6f}")
            if clip is not None and not np.isnan(clip):
                parts.append(f"clip_range={clip:.3f}")
            if ep_r is not None:
                parts.append(f"episode_reward={float(ep_r):.2f}")
            if ep_l is not None:
                parts.append(f"episode_length={int(ep_l)}")
            if recent_mean_r is not None:
                parts.append(f"recent10_reward={recent_mean_r:.2f}")
            for k, v in train_vals.items():
                if v is not None and not np.isnan(v):
                    parts.append(f"{k}={v:.4f}")
            print(" ".join(parts))
        except Exception:
            pass

    def _print_train_metrics(self, lv: dict) -> None:
        try:
            keys = [
                'train/entropy_loss',
                'train/policy_gradient_loss',
                'train/value_loss',
                'train/approx_kl',
                'train/clip_fraction',
                'train/loss',
                'train/explained_variance',
                'train/std',
                'train/n_updates',
                'train/clip_range',
            ]
            parts = [f"timesteps={self.num_timesteps:,}"]
            for k in keys:
                v = lv.get(k, None)
                if v is None:
                    continue
                try:
                    fv = float(v)
                    parts.append(f"{k.replace('train/', '')}={fv:.4f}")
                except Exception:
                    parts.append(f"{k.replace('train/', '')}={v}")
            print(" ".join(parts))
        except Exception:
            pass

    def _on_rollout_end(self) -> bool:
        try:
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 10:
                ep_info_list = list(self.model.ep_info_buffer)
                recent_rewards = [ep['r'] for ep in ep_info_list[-10:] if isinstance(ep, dict) and 'r' in ep]
                if recent_rewards:
                    current_mean_reward = np.mean(recent_rewards)
                    if current_mean_reward > self.best_mean_reward + self.min_improvement:
                        self.best_mean_reward = current_mean_reward
                        self.wait = 0
                        if self.verbose > 0:
                            print(f"         {self.agent_id} new best performance: {current_mean_reward:.1f}")
                    else:
                        self.wait += 1
                        if self.verbose > 0:
                            print(f"         {self.agent_id} no improvement for {self.wait}/{self.patience} checks")
                        if self.wait >= self.patience:
                            if self.verbose > 0:
                                print(f"         {self.agent_id} early stopping - performance not improving")
                            return False
        except Exception as e:
            if self.verbose > 0:
                print(f"         {self.agent_id} early stopping check failed: {e}")
        return True


class CurriculumManager:
    """Manages curriculum learning phases for progressive skill development."""
    
    def __init__(self, agent_ids: List[str]):
        self.agent_ids = agent_ids
        self.phase_index = 0
        self.agent_performance = {agent_id: [] for agent_id in agent_ids}
        
        # Define curriculum phases with EASIER thresholds for faster progression
        self.phases = [
            {
                'name': 'safety',
                'description': 'Learn track boundaries and basic control',
                'max_speed_reward': 0.3,
                'safety_weight': 3.0,
                'progress_weight': 1.0,
                'completion_threshold': 15.0,
                'min_episodes': 10,
                'horizon': 2000,
                'termination_flags': {
                    'crash_done': False,
                    'out_of_road_done': False,
                    'on_continuous_line_done': False,
                    'on_broken_line_done': False,
                },
                'line_touch_streak_limit': None,
            },
            {
                'name': 'control',
                'description': 'Improve steering precision and consistency',
                'max_speed_reward': 0.6,
                'safety_weight': 2.0,
                'progress_weight': 1.5,
                'completion_threshold': 25.0,
                'min_episodes': 15,
                'horizon': 2500,
                'termination_flags': {
                    'crash_done': False,
                    'out_of_road_done': False,
                    'on_continuous_line_done': False,
                    'on_broken_line_done': False,
                },
                'line_touch_streak_limit': None,
            },
            {
                'name': 'speed',
                'description': 'Learn efficient racing lines and speed management',
                'max_speed_reward': 1.0,
                'safety_weight': 1.0,
                'progress_weight': 2.0,
                'completion_threshold': 40.0,
                'min_episodes': 20,
                'horizon': 3000,
                'termination_flags': {
                    'crash_done': True,
                    'out_of_road_done': False,
                    'on_continuous_line_done': False,
                    'on_broken_line_done': False,
                },
                'line_touch_streak_limit': 3,
            },
            {
                'name': 'racing',
                'description': 'Full competition with overtaking and positioning',
                'max_speed_reward': 3.0,
                'safety_weight': 0.2,
                'progress_weight': 3.0,
                'completion_threshold': float('inf'),
                'min_episodes': 0,
                'horizon': 4000,
                'termination_flags': {
                    'crash_done': True,
                    'out_of_road_done': True,
                    'on_continuous_line_done': False,
                    'on_broken_line_done': False,
                },
                'line_touch_streak_limit': None,
            }
        ]
    
    def get_current_phase(self) -> str:
        """Get current curriculum phase name."""
        return self.phases[self.phase_index]['name']
    
    def get_phase_config(self) -> Dict:
        """Get current phase configuration."""
        return self.phases[self.phase_index].copy()
    
    def update_performance(self, agent_id: str, reward: float, episode_length: int):
        """Update agent performance and check for phase transitions."""
        self.agent_performance[agent_id].append({
            'reward': reward,
            'episode_length': episode_length,
            'timestamp': time.time()
        })
        
        # Keep only recent performance
        if len(self.agent_performance[agent_id]) > 100:
            self.agent_performance[agent_id] = self.agent_performance[agent_id][-100:]
    
    def should_advance_phase(self) -> bool:
        """Check if all agents are ready to advance to next phase."""
        if self.phase_index >= len(self.phases) - 1:
            return False  # Already at final phase
        
        current_phase = self.phases[self.phase_index]
        
        # Check if all agents meet the advancement criteria
        for agent_id in self.agent_ids:
            performance = self.agent_performance[agent_id]
            
            # Need minimum episodes
            if len(performance) < current_phase['min_episodes']:
                return False
            
            # Check recent performance against threshold
            recent_rewards = [p['reward'] for p in performance[-10:]]
            if len(recent_rewards) > 0:
                avg_reward = np.mean(recent_rewards)
                if avg_reward < current_phase['completion_threshold']:
                    return False
        
        return True
    
    def advance_phase(self):
        """Advance to the next curriculum phase."""
        if self.phase_index < len(self.phases) - 1:
            self.phase_index += 1
            print(f"\nCURRICULUM PHASE ADVANCEMENT!")
            print(f"   Advanced to Phase {self.phase_index + 1}: {self.get_current_phase().upper()}")
            print(f"   Goal: {self.phases[self.phase_index]['description']}")
            return True
        return False
    
    def get_phase_progress(self) -> float:
        """Get progress within current phase (0.0 to 1.0)."""
        if self.phase_index >= len(self.phases) - 1:
            return 1.0  # Final phase
        
        current_phase = self.phases[self.phase_index]
        
        # Calculate based on minimum performance across all agents
        min_episodes = current_phase['min_episodes']
        threshold = current_phase['completion_threshold']
        
        total_progress = 0.0
        for agent_id in self.agent_ids:
            performance = self.agent_performance[agent_id]
            
            # Episode progress
            episode_progress = min(1.0, len(performance) / max(1, min_episodes))
            
            # Performance progress
            if len(performance) > 0:
                recent_rewards = [p['reward'] for p in performance[-10:]]
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                performance_progress = min(1.0, max(0.0, avg_reward / threshold))
            else:
                performance_progress = 0.0
            
            # Combined progress (both criteria must be met)
            agent_progress = min(episode_progress, performance_progress)
            total_progress += agent_progress
        
        return total_progress / len(self.agent_ids)


class MultiAgentRacingEnvironment(gym.Env):
    """Multi-agent racing environment with curriculum learning and per-agent done logic."""
    
    def __init__(self, num_agents: int = 4, curriculum_manager: CurriculumManager = None, debug_steps: bool = False):
        super().__init__()
        self.num_agents = num_agents
        self.agent_ids = [f"agent{i}" for i in range(num_agents)]
        self.curriculum_manager = curriculum_manager or CurriculumManager(self.agent_ids)
        self.debug_steps = debug_steps
        self.throttle_cap = 0.4
        self.episodes_completed = 0
        self.action_noise_scale = 0.02
        self.agent_reward_profiles = {
            "agent0": {"speed_scale": 0.6, "progress_scale": 0.8, "safety_scale": 1.5, "race_position_scale": 0.0},
            "agent1": {"speed_scale": 0.8, "progress_scale": 1.0, "safety_scale": 1.2, "race_position_scale": 0.0},
            "agent2": {"speed_scale": 1.4, "progress_scale": 1.2, "safety_scale": 0.8, "race_position_scale": 0.2},
            "agent3": {"speed_scale": 2.0, "progress_scale": 1.5, "safety_scale": 0.5, "race_position_scale": 0.5},
        }
        self.agent_steer_bias_count = {agent_id: 0 for agent_id in self.agent_ids}
        self.steering_bias_threshold = 0.08
        self.steering_bias_window = 20
        
        self.agent_prev_lane_offset = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.agent_prev_heading = {agent_id: None for agent_id in self.agent_ids}
        
        # Create base environment
        self.base_env = self._create_base_environment()
        
        # Get observation and action spaces from base environment
        reset_result = self.base_env.reset()
        if isinstance(reset_result, tuple):
            obs_dict = reset_result[0]
        else:
            obs_dict = reset_result
        
        # Set spaces for shared policy (using first agent's spaces)
        sample_obs = obs_dict[self.agent_ids[0]]
        self.observation_space = gym.spaces.Box(
            low=-float('inf'), 
            high=float('inf'), 
            shape=sample_obs.shape, 
            dtype=sample_obs.dtype
        )
        self.action_space = list(self.base_env.action_space.spaces.values())[0]
        
        # Agent state tracking
        self.agent_done = {agent_id: False for agent_id in self.agent_ids}
        self.agent_line_touch_streak = {agent_id: 0 for agent_id in self.agent_ids}
        self.agent_positions = {agent_id: np.array([0.0, 0.0]) for agent_id in self.agent_ids}
        self.agent_prev_positions = {agent_id: np.array([0.0, 0.0]) for agent_id in self.agent_ids}
        self.agent_laps = {agent_id: 0 for agent_id in self.agent_ids}
        self.agent_lap_times = {agent_id: [] for agent_id in self.agent_ids}
        self.agent_episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.agent_prev_steering = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.steering_smooth = 0.7
        
        # Episode tracking
        self.episode_step = 0
        self.current_agent_idx = 0  # For round-robin shared policy training
        
        print(f"Multi-Agent Racing Environment Created!")
        print(f"   Agents: {len(self.agent_ids)}")
        print(f"   Curriculum: {self.curriculum_manager.get_current_phase().upper()}")
        print(f"   Observation space: {self.observation_space.shape}")
        print(f"   Action space: {self.action_space}")
        
    def _create_base_environment(self):
        """Create the base MetaDrive environment with proper racing grid setup."""
        phase_config = self.curriculum_manager.get_phase_config()
        
        # Generate racing grid configurations: unique lane per agent, safe center positions
        racing_grid_configs = {}
        lane_width = 12.0
        for i in range(self.num_agents):
            racing_grid_configs[f"agent{i}"] = {
                "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, i % self.num_agents),
                "spawn_longitude": 4.0,
                "spawn_lateral": 0.0,
                "spawn_velocity": [2.0, 0.0],
            }
        
        # Environment configuration
        term = phase_config.get('termination_flags', {})
        config = {
            "num_agents": self.num_agents,
            "traffic_density": 0.0,
            "use_render": False,
            
            # CRITICAL: Per-agent done logic, no auto-respawn
            "crash_done": bool(term.get('crash_done', False)),
            "out_of_road_done": bool(term.get('out_of_road_done', False)),
            "on_continuous_line_done": bool(term.get('on_continuous_line_done', False)),
            "on_broken_line_done": bool(term.get('on_broken_line_done', False)),
            "allow_respawn": False,  # No automatic respawning
            "crash_vehicle_done": False,
            "crash_object_done": False,
            "horizon": phase_config['horizon'],
            
            # Curriculum-based rewards
            "success_reward": phase_config['progress_weight'] * 10.0,
            "driving_reward": phase_config['progress_weight'],
            "speed_reward": phase_config['max_speed_reward'],
            "use_lateral_reward": True,
            "out_of_road_penalty": phase_config['safety_weight'] * 2.0,
            "crash_vehicle_penalty": phase_config['safety_weight'] * 5.0,
            "crash_object_penalty": phase_config['safety_weight'] * 4.0,
            
            # Track configuration
            "map_config": {
                "lane_num": self.num_agents,
                "lane_width": lane_width,
            },
            
            # Vehicle configuration with lidar
            "vehicle_config": {
                "show_lidar": True,
                "show_lane_line_detector": True,
                "show_side_detector": True,
                "enable_reverse": False,
                "lidar": {
                    "num_others": 4,  # Detect other vehicles
                    "distance": 50,  # Detection range
                    "num_lasers": 72,  # High resolution
                },
            },
            "agent_configs": racing_grid_configs
        }
        
        # Use custom environment
        env = MultiAgentOvalEnv(config)
        print("Using custom right-turn oval track!")
        return env
    
    def _calculate_curriculum_reward(self, agent_id: str, obs, action, info) -> float:
        reward = 0.0
        phase_cfg = self.curriculum_manager.get_phase_config()
        profile = self.agent_reward_profiles.get(agent_id, {"speed_scale": 1.0, "progress_scale": 1.0, "safety_scale": 1.0, "race_position_scale": 0.0})

        if 'vehicle_state' in info:
            vehicle_state = info['vehicle_state']
            speed = vehicle_state.get('speed', 0.0)
            position = np.array(vehicle_state.get('position', [0, 0]))
            on_road = vehicle_state.get('on_road', True)
            crashed = vehicle_state.get('crashed', False)
        else:
            speed = obs[0] if len(obs) > 0 else 0.0
            position = np.array([obs[1], obs[2]]) if len(obs) > 2 else np.array([0.0, 0.0])
            on_road = True
            crashed = False

        self.agent_prev_positions[agent_id] = self.agent_positions[agent_id].copy()
        self.agent_positions[agent_id] = position

        reward += min(speed * 25.0, 25.0) * phase_cfg['max_speed_reward'] * profile['speed_scale']

        if not np.array_equal(self.agent_prev_positions[agent_id], np.array([0.0, 0.0])):
            forward_progress = position[0] - self.agent_prev_positions[agent_id][0]
            if forward_progress > 0.01:
                reward += min(forward_progress * 5.0, 5.0) * phase_cfg['progress_weight'] * profile['progress_scale']
            if position[0] > 100 and self.agent_prev_positions[agent_id][0] < -100:
                self.agent_laps[agent_id] += 1
                reward += 10.0 * phase_cfg['progress_weight'] * profile['progress_scale']
                print(f"   LAP COMPLETE: {agent_id} lap {self.agent_laps[agent_id]}")

        if info.get('lane_line_collision', False) or info.get('white_line_collision', False) \
           or info.get('yellow_line_collision', False) or info.get('broken_line_collision', False):
            reward -= phase_cfg['safety_weight'] * profile['safety_scale'] * 0.1

        if isinstance(action, (list, tuple)) and len(action) >= 2:
            steer = float(action[0])
            reward -= phase_cfg['safety_weight'] * profile['safety_scale'] * 0.10 * abs(steer)
            prev = float(self.agent_prev_steering.get(agent_id, 0.0))
            same_dir = (steer * prev) > 0 and abs(steer) > self.steering_bias_threshold and abs(prev) > self.steering_bias_threshold
            if same_dir:
                self.agent_steer_bias_count[agent_id] += 1
            else:
                self.agent_steer_bias_count[agent_id] = 0
            if self.agent_steer_bias_count[agent_id] > self.steering_bias_window:
                bias_pen = 0.02 * (self.agent_steer_bias_count[agent_id] - self.steering_bias_window)
                bias_pen = min(bias_pen, 0.5)
                reward -= bias_pen * phase_cfg['safety_weight'] * profile['safety_scale']

        if speed < 0.02:
            reward -= 0.1 * phase_cfg['safety_weight'] * profile['safety_scale']

        if crashed:
            reward -= 6.0 * phase_cfg['safety_weight'] * profile['safety_scale']
            self.agent_done[agent_id] = True

        if not on_road:
            reward -= 1.0 * phase_cfg['safety_weight'] * profile['safety_scale']
        else:
            reward += 0.05 * phase_cfg['progress_weight'] * profile['progress_scale']

        if profile.get('race_position_scale', 0.0) > 0.0:
            others = [aid for aid in self.agent_ids if aid != agent_id]
            if others:
                ahead = sum(self.agent_positions[aid][0] < position[0] for aid in others)
                frac = ahead / len(others)
                reward += 0.5 * frac * profile['race_position_scale']

        lane_width = info.get('lane_width', None)
        lane_offset = info.get('lane_offset', None)
        if lane_width is not None and lane_offset is not None:
            center_closeness = 1.0 - min(1.0, abs(lane_offset) / max(1e-6, lane_width * 0.5))
            reward += 0.3 * center_closeness * phase_cfg['progress_weight'] * profile['progress_scale']
            prev_off = self.agent_prev_lane_offset.get(agent_id, lane_offset)
            delta_off = abs(lane_offset - prev_off)
            lateral_pen = min(0.2, delta_off / max(1e-6, lane_width * 0.5))
            reward -= lateral_pen * 0.1 * phase_cfg['safety_weight'] * profile['safety_scale']
            self.agent_prev_lane_offset[agent_id] = lane_offset
        heading = info.get('heading', None)
        if heading is not None:
            prev_heading = self.agent_prev_heading.get(agent_id, heading)
            try:
                dh = abs(float(heading) - float(prev_heading))
                dh = dh % (2.0 * np.pi)
                dh = min(dh, (2.0 * np.pi) - dh)
                reward -= min(0.2, dh) * 0.05 * phase_cfg['safety_weight'] * profile['safety_scale']
            except Exception:
                pass
            self.agent_prev_heading[agent_id] = heading

        reward = max(reward, -5.0)
        return np.clip(reward, -5.0, 30.0)
    
    def reset(self, **kwargs):
        """Reset environment and return observation for shared policy."""
        # Check for curriculum advancement
        if self.curriculum_manager.should_advance_phase():
            if self.curriculum_manager.advance_phase():
                # Recreate environment with new phase config
                self.base_env.close()
                self.base_env = self._create_base_environment()
        
        # Reset base environment
        obs_dict = self.base_env.reset()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]
        
        # Reset agent states
        self.agent_done = {agent_id: False for agent_id in self.agent_ids}
        self.agent_line_touch_streak = {agent_id: 0 for agent_id in self.agent_ids}
        self.agent_positions = {agent_id: np.array([0.0, 0.0]) for agent_id in self.agent_ids}
        self.agent_prev_positions = {agent_id: np.array([0.0, 0.0]) for agent_id in self.agent_ids}
        self.agent_laps = {agent_id: 0 for agent_id in self.agent_ids}
        self.agent_episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.episode_step = 0
        self.current_agent_idx = 0
        self.agent_prev_lane_offset = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.agent_prev_heading = {agent_id: None for agent_id in self.agent_ids}
        
        # Return observation for current agent (shared policy approach)
        current_agent = self.agent_ids[self.current_agent_idx]
        return obs_dict[current_agent], {}
    
    def step(self, action):
        """Step environment with shared policy approach."""
        current_agent = self.agent_ids[self.current_agent_idx]
        
        # Create actions for all agents (shared policy with variation)
        actions = {}
        for i, agent_id in enumerate(self.agent_ids):
            if self.agent_done[agent_id]:
                actions[agent_id] = [0.0, 0.0]
            elif agent_id == current_agent:
                actions[agent_id] = action
            else:
                if len(action) >= 2:
                    mod_steering = action[0] + self.action_noise_scale * ((i % 5) - 2)
                    mod_throttle = action[1] + 0.05 * ((i % 3) - 1)
                    mod_steering = np.clip(mod_steering, -1.0, 1.0)
                    mod_throttle = np.clip(mod_throttle, 0.0, 1.0)
                    actions[agent_id] = [mod_steering, mod_throttle]
                else:
                    actions[agent_id] = action
        def _norm_val(v):
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if isinstance(v, (list, tuple)):
                return [float(x) for x in v]
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            try:
                return float(v)
            except Exception:
                return v
        normalized_actions = {k: _norm_val(v) for k, v in actions.items()}
        for aid, v in normalized_actions.items():
            if isinstance(v, list) and len(v) >= 2:
                s = float(v[0])
                t = float(v[1])
                s = float(np.clip(s, -0.3, 0.3))
                t = float(np.clip(t, 0.0, self.throttle_cap))
                normalized_actions[aid] = [s, t]
        for aid, v in normalized_actions.items():
            if isinstance(v, list) and len(v) >= 2:
                prev = float(self.agent_prev_steering.get(aid, 0.0))
                s = float(v[0])
                smoothed = self.steering_smooth * prev + (1.0 - self.steering_smooth) * s
                self.agent_prev_steering[aid] = smoothed
                normalized_actions[aid][0] = smoothed
        if self.debug_steps:
            print(f"Step {self.episode_step}: Actions sent to env: {normalized_actions}")

        # Step base environment
        info_dict = {}  # Always initialize
        step_result = self.base_env.step(normalized_actions)
        if len(step_result) == 4:
            obs_dict, reward_dict, done_dict, info_dict = step_result
        elif len(step_result) == 5:
            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = step_result
            done_dict = {k: terminated_dict.get(k, False) or truncated_dict.get(k, False) for k in terminated_dict.keys()}
        else:
            obs_dict, reward_dict, done_dict = None, None, None
            if len(step_result) > 0 and isinstance(step_result[-1], dict):
                info_dict = step_result[-1]

        if self.debug_steps:
            for agent_id in self.agent_ids:
                agent_info = info_dict.get(agent_id, {})
                heading = agent_info.get('heading', None)
                print(f"[STEP] Agent {agent_id}: reward={self.agent_episode_rewards[agent_id]:.2f}, heading={heading}")
        
        # Calculate curriculum rewards for all agents
        total_reward = 0.0
        active_agents = 0
        current_agent_reward = 0.0
        
        for agent_id in self.agent_ids:
            if not self.agent_done[agent_id]:
                agent_obs = obs_dict[agent_id]
                agent_info = info_dict.get(agent_id, {})
                agent_action = normalized_actions[agent_id]
                
                # Calculate custom reward
                agent_reward = self._calculate_curriculum_reward(agent_id, agent_obs, agent_action, agent_info)
                total_reward += agent_reward
                self.agent_episode_rewards[agent_id] += agent_reward
                active_agents += 1

                if agent_id == current_agent:
                    current_agent_reward = agent_reward
                
                # Check if agent should be marked as done
                phase_cfg = self.curriculum_manager.get_phase_config()
                streak_limit = phase_cfg.get('line_touch_streak_limit')
                if agent_info.get('lane_line_collision', False):
                    self.agent_line_touch_streak[agent_id] += 1
                else:
                    self.agent_line_touch_streak[agent_id] = 0

                lane_streak_done = streak_limit is not None and self.agent_line_touch_streak[agent_id] >= streak_limit
                term_flags = phase_cfg.get('termination_flags', {})
                crash_done_flag = bool(term_flags.get('crash_done', False))
                mark_done = bool(done_dict.get(agent_id, False)) or lane_streak_done or (bool(agent_info.get('crashed', False)) and crash_done_flag)
                if mark_done:
                    self.agent_done[agent_id] = True
                    if self.debug_steps:
                        reason = "lane_line_streak" if lane_streak_done else ("crash" if agent_info.get('crashed', False) and crash_done_flag else "done")
                        print(f"   DONE: {agent_id} ({reason}) at step {self.episode_step}")
        
        # Average reward across active agents (kept for logging), but return current agent's reward
        if active_agents > 0:
            total_reward = total_reward / active_agents
        
        # Episode termination - only when all agents done or horizon reached
        phase_config = self.curriculum_manager.get_phase_config()
        all_done = all(self.agent_done.values())
        horizon_reached = self.episode_step >= (phase_config['horizon'] - 1)
        episode_done = all_done or horizon_reached
        
        if episode_done:
            if self.debug_steps:
                active_count = sum(1 for done in self.agent_done.values() if not done)
                print(f"    Episode complete: {active_count} agents still active, step {self.episode_step}")
            self.episodes_completed += 1
            self.throttle_cap = min(1.0, 0.4 + 0.01 * self.episodes_completed)
            self.action_noise_scale = max(0.005, 0.02 - 0.0002 * self.episodes_completed)
        
        self.episode_step += 1
        
        # Move to next agent for next step (round-robin)
        self.current_agent_idx = (self.current_agent_idx + 1) % len(self.agent_ids)
        
        # Return observation for next agent, including heading and position
        next_agent = self.agent_ids[self.current_agent_idx]
        obs = obs_dict.get(next_agent, None)
        if self.agent_done.get(next_agent, False) or obs is None:
            zero_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            return zero_obs, current_agent_reward, episode_done, episode_done, {}
        # Return original observation to match observation_space
        return obs, current_agent_reward, episode_done, episode_done, {}
    
    def close(self):
        """Clean up environment."""
        if hasattr(self, 'base_env'):
            self.base_env.close()


def train_multi_agent_racing(
    num_agents: int = 4,
    total_timesteps: int = 200_000,
    results_dir: str = 'multi_agent_racing_results',
    use_wandb: bool = False
):
    """Train multi-agent racing with curriculum learning."""
    
    print("MULTI-AGENT RACING WITH CURRICULUM LEARNING")
    print("=" * 60)
    print(f"Agents: {num_agents}")
    print(f"Algorithm: PPO with Shared Policy")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Results directory: {results_dir}")
    print()
    print("TRAINING MODE:")
    print("   Final phase only: maximize speed while obeying constraints")
    print()
    print("FEATURES:")
    print("   - Per-agent done logic (crashed agents marked done)")
    print("   - No automatic respawning")
    print("   - Curriculum-based reward progression")
    print("   - Shared policy learning")
    print("   - Lidar perception with smooth actions")
    print("   - Racing grid formation")
    print()
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.abspath(os.path.join(base_dir, results_dir))
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize Weights & Biases
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="metadrive-multi-agent-racing-curriculum",
            name=f"curriculum_racing_{num_agents}agents",
            config={
                "num_agents": num_agents,
                "total_timesteps": total_timesteps,
                "algorithm": "PPO_Shared_Policy_Curriculum",
                "curriculum_phases": 1,
                "racing_track": "right_turn_oval",
                "perception": "lidar",
                "done_logic": "per_agent"
            }
        )
    
    # Create curriculum manager and environment
    agent_ids = [f"agent{i}" for i in range(num_agents)]
    curriculum_manager = CurriculumManager(agent_ids)
    racing_env = MultiAgentRacingEnvironment(num_agents, curriculum_manager)
    
    print(f"Environment created successfully!")
    print(f"   Current phase: {curriculum_manager.get_current_phase().upper()}")
    
    # Create PPO model with shared policy - ENHANCED FOR EXPLORATION
    import torch.nn as nn
    
    model = PPO(
        policy="MlpPolicy",
        env=racing_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log=os.path.join(results_dir, "tensorboard"),
        policy_kwargs={
            "net_arch": [64, 64],  # Smaller network for faster learning
            "activation_fn": nn.Tanh,  # FIXED: Use actual function not string
            "ortho_init": False,  # Don't use orthogonal initialization
        }
    )
    
    print(f"PPO model created with shared policy.")
    
    # Create callbacks - use first agent for tracking since it's shared policy
    progress_callback = CurriculumProgressCallback(agent_ids[0], curriculum_manager, verbose=0)
    stats_callback = TrainingStatsCallback(verbose=0)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(results_dir, "checkpoints"),
        name_prefix="curriculum_racing",
        verbose=0,
    )
    
    callbacks = [progress_callback, stats_callback, checkpoint_callback]

    if use_wandb and WANDB_AVAILABLE:
        wandb_callback = WandbCallback(
            gradient_save_freq=10000,
            model_save_path=None,
            model_save_freq=0,
            verbose=1,
        )
        callbacks.append(wandb_callback)

    early_stopping_callback = EarlyStoppingCallback(agent_ids[0], patience=15, min_improvement=1.0, verbose=1)
    callbacks.append(early_stopping_callback)
    
    # Training loop
    print(f"\nStarting fast-safe training in final phase...")
    print()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
        )
        
        print(f"Training completed.")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save final models
        final_dir = os.path.join(results_dir, "final_models")
        os.makedirs(final_dir, exist_ok=True)
        
        # Save shared policy
        shared_model_path = os.path.join(final_dir, "shared_policy.zip")
        model.save(shared_model_path)
        print(f"Shared policy saved: {shared_model_path}")
        
        # Create individual agent models (copies of shared policy)
        for agent_id in agent_ids:
            agent_model_path = os.path.join(final_dir, f"{agent_id}_final.zip")
            model.save(agent_model_path)
            print(f"Saved {agent_id} model: {agent_model_path}")
        
        # Save curriculum progress
        curriculum_data = {
            'final_phase': curriculum_manager.get_current_phase(),
            'phase_index': curriculum_manager.phase_index,
            'agent_performance': curriculum_manager.agent_performance,
            'phases': curriculum_manager.phases
        }
        
        with open(os.path.join(results_dir, "curriculum_progress.pkl"), 'wb') as f:
            pickle.dump(curriculum_data, f)
        
        # Cleanup
        racing_env.close()
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()
    
    print(f"\nMulti-Agent Racing Training Complete!")
    print(f"Models saved in: {results_dir}")
    print()
    print("Training Benefits:")
    print("1. Curriculum learning for stable progression")
    print("2. Per-agent done logic for realistic racing")
    print("3. Shared policy for coordinated behavior")
    print("4. Progressive reward shaping")
    print("5. No respawn loops - proper racing consequences")


def main():
    """Main training function with command line arguments."""
    parser = argparse.ArgumentParser(description='Multi-Agent Racing with Curriculum Learning')
    parser.add_argument('--agents', type=int, default=4,
                       help='Number of racing agents (default: 4)')
    parser.add_argument('--timesteps', type=int, default=200_000,
                       help='Total training timesteps (default: 200k)')
    parser.add_argument('--results-dir', type=str, default='multi_agent_racing_results',
                       help='Results directory (default: multi_agent_racing_results)')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    
    args = parser.parse_args()
    
    try:
        train_multi_agent_racing(
            num_agents=args.agents,
            total_timesteps=args.timesteps,
            results_dir=args.results_dir,
            use_wandb=args.wandb
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
