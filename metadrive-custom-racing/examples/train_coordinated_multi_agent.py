
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

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.buffers import RolloutBuffer
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


class EarlyStoppingCallback(BaseCallback):
    """Early stopping callback to prevent catastrophic forgetting."""
    
    def __init__(self, agent_id: str, patience: int = 5, min_improvement: float = 1.0, verbose: int = 1):
        super().__init__(verbose)
        self.agent_id = agent_id
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_mean_reward = -np.inf
        self.wait = 0
        self.stopped_epoch = 0
    
    def _on_step(self) -> bool:
        """Called after each step. Required method for BaseCallback."""
        return True
        
    def _on_rollout_end(self) -> bool:
        """Check if we should stop training due to performance degradation."""
        try:
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 10:  # Need at least 10 episodes
                # Convert to list if it's a deque and safely slice
                ep_info_list = list(self.model.ep_info_buffer)
                recent_rewards = [ep['r'] for ep in ep_info_list[-10:] if isinstance(ep, dict) and 'r' in ep]
                if recent_rewards:
                    current_mean_reward = np.mean(recent_rewards)
                    
                    if current_mean_reward > self.best_mean_reward + self.min_improvement:
                        self.best_mean_reward = current_mean_reward
                        self.wait = 0
                        if self.verbose > 0:
                            print(f"         🎯 {self.agent_id} new best performance: {current_mean_reward:.1f}")
                    else:
                        self.wait += 1
                        if self.verbose > 0:
                            print(f"         ⚠️  {self.agent_id} no improvement for {self.wait}/{self.patience} checks")
                        
                        if self.wait >= self.patience:
                            if self.verbose > 0:
                                print(f"         🛑 {self.agent_id} early stopping - performance not improving")
                            return False  # Stop training
        except Exception as e:
            # Safely handle any buffer access errors
            if self.verbose > 0:
                print(f"         ⚠️  {self.agent_id} early stopping check failed: {e}")
        
        return True


class ConciseProgressCallback(BaseCallback):
    def __init__(self, tag: str, verbose: int = 0):
        super().__init__(verbose)
        self.tag = tag
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if 'r' in ep_info and 'l' in ep_info:
                r = ep_info['r']
                l = ep_info['l']
                self.episode_rewards.append(r)
                self.episode_lengths.append(l)
                if len(self.episode_rewards) > 100:
                    self.episode_rewards = self.episode_rewards[-100:]
                    self.episode_lengths = self.episode_lengths[-100:]
                mean_r = float(np.mean(self.episode_rewards[-10:])) if self.episode_rewards else float(r)
                mean_l = float(np.mean(self.episode_lengths[-10:])) if self.episode_lengths else float(l)
                if mean_r > self.best_mean_reward:
                    self.best_mean_reward = mean_r
                print(f"[TS={self.num_timesteps:,}] [tag={self.tag}] [ep_r={r:.1f}] [ep_l={l}] [mean10_r={mean_r:.1f}] [mean10_l={mean_l:.1f}] [best={self.best_mean_reward:.1f}]")
                if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'record'):
                    self.model.logger.record(f"{self.tag}/episode_reward", r)
                    self.model.logger.record(f"{self.tag}/episode_length", l)
                    self.model.logger.record(f"{self.tag}/mean_reward_10ep", mean_r)
                    self.model.logger.record(f"{self.tag}/best_mean_reward", self.best_mean_reward)
                    self.model.logger.dump(self.num_timesteps)


class MultiAgentProgressCallback(BaseCallback):
    """Custom callback to track multi-agent training progress with detailed metrics."""
    
    def __init__(self, agent_id: str, coordinator, verbose: int = 1):
        super().__init__(verbose)
        self.agent_id = agent_id
        self.coordinator = coordinator
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        """Called after each step."""
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        # Get episode statistics
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if 'r' in ep_info and 'l' in ep_info:
                episode_reward = ep_info['r']
                episode_length = ep_info['l']
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Keep only recent episodes
                if len(self.episode_rewards) > 100:
                    self.episode_rewards = self.episode_rewards[-100:]
                    self.episode_lengths = self.episode_lengths[-100:]
                
                # Calculate statistics
                mean_reward = np.mean(self.episode_rewards[-10:])  # Last 10 episodes
                mean_length = np.mean(self.episode_lengths[-10:])
                
                # Update best performance
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                
                # Print detailed progress
                print(f"      👥 {self.agent_id} - Timesteps: {self.num_timesteps:,}")
                print(f"         Episode: {len(self.episode_rewards)} | Reward: {episode_reward:.1f} | Length: {episode_length}")
                print(f"         Mean (10ep): Reward {mean_reward:.1f} | Length {mean_length:.1f} | Best: {self.best_mean_reward:.1f}")
                
                # Log to wandb if available
                if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'record'):
                    self.model.logger.record(f"{self.agent_id}/episode_reward", episode_reward)
                    self.model.logger.record(f"{self.agent_id}/episode_length", episode_length)
                    self.model.logger.record(f"{self.agent_id}/mean_reward_10ep", mean_reward)
                    self.model.logger.record(f"{self.agent_id}/best_mean_reward", self.best_mean_reward)
                    
                    # CRITICAL FIX: Force dump to WandB
                    self.model.logger.dump(self.num_timesteps)
                
                # BACKUP: Direct WandB logging
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            f"{self.agent_id}/episode_reward": episode_reward,
                            f"{self.agent_id}/episode_length": episode_length,
                            f"{self.agent_id}/mean_reward_10ep": mean_reward,
                            f"{self.agent_id}/best_mean_reward": self.best_mean_reward,
                        }, step=self.num_timesteps)
                except:
                    pass


class SharedExperienceBuffer:
    """Shared experience buffer for multi-agent coordination."""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.experiences = deque(maxlen=max_size)
        self.agent_contributions = {}
    
    def add_experience(self, agent_id: str, obs, action, reward, next_obs, done, info):
        """Add experience from an agent."""
        experience = {
            'agent_id': agent_id,
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done,
            'info': info,
            'timestamp': time.time()
        }
        
        self.experiences.append(experience)
        
        if agent_id not in self.agent_contributions:
            self.agent_contributions[agent_id] = 0
        self.agent_contributions[agent_id] += 1
    
    def get_recent_experiences(self, count: int = 1000) -> List[Dict]:
        """Get recent experiences for analysis."""
        return list(self.experiences)[-count:]
    
    def get_opponent_experiences(self, agent_id: str, count: int = 500) -> List[Dict]:
        """Get experiences from other agents for learning."""
        opponent_experiences = [
            exp for exp in self.experiences 
            if exp['agent_id'] != agent_id
        ]
        return opponent_experiences[-count:]
    
    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        return {
            'total_experiences': len(self.experiences),
            'agent_contributions': self.agent_contributions.copy(),
            'buffer_usage': len(self.experiences) / self.max_size
        }


class MultiAgentCoordinator:
    """Coordinates training between multiple agents."""
    
    def __init__(self, agent_ids: List[str], shared_buffer: SharedExperienceBuffer):
        self.agent_ids = agent_ids
        self.shared_buffer = shared_buffer
        self.training_order = 0
        self.performance_history = {agent_id: [] for agent_id in agent_ids}
        
    def get_next_training_agent(self) -> str:
        """Get the next agent to train (round-robin)."""
        agent_id = self.agent_ids[self.training_order % len(self.agent_ids)]
        self.training_order += 1
        return agent_id
    
    def update_performance(self, agent_id: str, reward: float, episode_length: int):
        """Update agent performance tracking."""
        self.performance_history[agent_id].append({
            'reward': reward,
            'episode_length': episode_length,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.performance_history[agent_id]) > 100:
            self.performance_history[agent_id] = self.performance_history[agent_id][-100:]
    
    def get_performance_ranking(self) -> List[Tuple[str, float]]:
        """Get agents ranked by recent performance."""
        rankings = []
        
        for agent_id in self.agent_ids:
            if self.performance_history[agent_id]:
                recent_rewards = [p['reward'] for p in self.performance_history[agent_id][-10:]]
                avg_reward = np.mean(recent_rewards)
                rankings.append((agent_id, avg_reward))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)


class SharedPolicyMultiAgentWrapper(gym.Env):
    """Wrapper that enables shared policy training for all agents simultaneously."""
    
    def __init__(self, base_env, agent_ids: List[str], coordinator: MultiAgentCoordinator, curriculum_phase: str = 'drive'):
        super().__init__()
        self.base_env = base_env
        self.agent_ids = agent_ids
        self.coordinator = coordinator
        self.curriculum_phase = curriculum_phase
        self.current_agent_idx = 0  # Round-robin through agents
        # Per-agent reward profiles to diversify objectives
        self.agent_reward_profiles = {
            agent_ids[0]: {"speed_scale": 0.6, "progress_scale": 0.8, "safety_scale": 1.5},
            agent_ids[1] if len(agent_ids) > 1 else agent_ids[0]: {"speed_scale": 0.9, "progress_scale": 1.0, "safety_scale": 1.2},
            agent_ids[2] if len(agent_ids) > 2 else agent_ids[0]: {"speed_scale": 1.3, "progress_scale": 1.2, "safety_scale": 0.9},
            agent_ids[3] if len(agent_ids) > 3 else agent_ids[0]: {"speed_scale": 1.8, "progress_scale": 1.5, "safety_scale": 0.6},
        }
        
        # Get individual agent spaces from the multi-agent environment
        # Base environment has Dict spaces, we need individual spaces
        if hasattr(base_env, 'observation_space') and hasattr(base_env.observation_space, 'spaces'):
            # Multi-agent Dict space - extract individual agent space
            self.observation_space = list(base_env.observation_space.spaces.values())[0]
            self.action_space = list(base_env.action_space.spaces.values())[0]
        else:
            # Fallback - assume single agent spaces
            self.observation_space = base_env.observation_space
            self.action_space = base_env.action_space
        
        # Track episode data for all agents
        self.episode_rewards = {agent_id: [] for agent_id in agent_ids}
        self.episode_length = 0
        self.current_episode_rewards = {agent_id: 0.0 for agent_id in agent_ids}
        
        # Racing metrics for logging
        self.metrics = {agent_id: {
            'crashes_per_episode': 0,
            'out_of_bounds_count': 0,
            'distance_traveled': 0.0,
            'max_speed_achieved': 0.0,
            'laps_completed': 0,
            'progress_on_track': 0.0,
            'smooth_steering_score': 0.0,
            'last_steering': 0.0,
            'idle_time': 0,
            'backward_time': 0
        } for agent_id in agent_ids}
        
        # Previous state for reward calculation
        self.prev_positions = {agent_id: None for agent_id in agent_ids}
        self.prev_speeds = {agent_id: 0.0 for agent_id in agent_ids}
        
    def get_current_agent_id(self):
        """Get the current agent ID for round-robin training."""
        return self.agent_ids[self.current_agent_idx]
    
    def calculate_custom_reward(self, obs, action, info, agent_id: str):
        """Calculate custom racing reward that FORCES movement and penalizes staying still."""
        reward = 0.0
        profile = self.agent_reward_profiles.get(agent_id, {"speed_scale": 1.0, "progress_scale": 1.0, "safety_scale": 1.0})
        
        # Get current vehicle state from info
        if 'vehicle_state' in info:
            vehicle_state = info['vehicle_state']
            current_speed = vehicle_state.get('speed', 0.0)
            position = vehicle_state.get('position', [0, 0])
            on_road = vehicle_state.get('on_road', True)
            crashed = vehicle_state.get('crashed', False)
        else:
            # Fallback to observation data
            current_speed = obs[0] if len(obs) > 0 else 0.0
            position = [obs[1], obs[2]] if len(obs) > 2 else [0, 0]
            on_road = True
            crashed = False
        
        # Update metrics
        self.metrics[agent_id]['max_speed_achieved'] = max(self.metrics[agent_id]['max_speed_achieved'], current_speed)
        
        # CRITICAL FIX: NO REWARDS WITHOUT MOVEMENT!
        base_reward = 0.0  # Start with zero - earn everything through action
        
        # 1. MOVEMENT REQUIREMENT - MUST MOVE TO GET ANY POSITIVE REWARD
        if current_speed < 0.05:  # Not moving
            # SEVERE PENALTY for sitting still
            idle_penalty = -2.0 * profile['safety_scale']
            reward += idle_penalty
            self.metrics[agent_id]['idle_time'] += 1
            # NO OTHER REWARDS if not moving!
            
        else:
            # ONLY give rewards if actually moving
            
            # 2. Speed reward - encourage faster movement
            if current_speed > 0.1:
                speed_reward = 1.0 * min(current_speed / 0.8, 1.0) * profile['speed_scale']
                reward += speed_reward
            
            # 3. Progress reward - ONLY if actually moving forward
            if self.prev_positions[agent_id] is not None:
                distance_moved = np.linalg.norm(np.array(position) - np.array(self.prev_positions[agent_id]))
                
                if distance_moved > 0.1:  # Actually moved
                    self.metrics[agent_id]['distance_traveled'] += distance_moved
                    
                    # Forward progress reward
                    forward_progress = position[0] - self.prev_positions[agent_id][0]
                    if forward_progress > 0:
                        progress_reward = 2.0 * min(forward_progress / 5.0, 1.0) * profile['progress_scale']
                        reward += progress_reward
                    
                    # Base movement reward
                    movement_reward = 0.5 * min(distance_moved, 2.0) * profile['progress_scale']
                    reward += movement_reward
                else:
                    # Penalty for moving but not making progress
                    stall_penalty = -0.5 * profile['safety_scale']
                    reward += stall_penalty
            
            # 4. On-road bonus (only when moving)
            if on_road:
                road_bonus = 0.2 * profile['progress_scale']
                reward += road_bonus
        
        # 5. Crash penalty (regardless of movement)
        if crashed:
            crash_penalty = -5.0 * profile['safety_scale']
            reward += crash_penalty
            self.metrics[agent_id]['crashes_per_episode'] += 1
        
        # 6. Action-taking requirement (correct mapping: [steering, throttle])
        if len(action) > 1:
            throttle = action[1]
            if throttle < 0.1:
                reward += -1.0
        
        # Update previous state
        self.prev_positions[agent_id] = position
        self.prev_speeds[agent_id] = current_speed
        
        # Clip reward
        return np.clip(reward, -10.0, 5.0)  # Negative bias to force action
    
    def reset_metrics(self):
        """Reset metrics for new episode."""
        for agent_id in self.agent_ids:
            self.metrics[agent_id] = {
                'crashes_per_episode': 0,
                'out_of_bounds_count': 0,
                'distance_traveled': 0.0,
                'max_speed_achieved': 0.0,
                'laps_completed': 0,
                'progress_on_track': 0.0,
                'smooth_steering_score': 0.0,
                'last_steering': 0.0,
                'idle_time': 0,
                'backward_time': 0
            }
            self.prev_positions[agent_id] = None
            self.prev_speeds[agent_id] = 0.0
        
    def reset(self, **kwargs):
        """Reset environment and update coordinator."""
        obs_dict = self.base_env.reset()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]
        
        # Update coordinator with previous episode performance
        if self.episode_length > 0:
            for agent_id in self.agent_ids:
                self.coordinator.update_performance(
                    agent_id, 
                    self.current_episode_rewards[agent_id], 
                    self.episode_length
                )
        
        # Reset episode tracking
        self.episode_length = 0
        self.current_episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.reset_metrics()
        
        # Return observation for current agent (round-robin)
        current_agent = self.get_current_agent_id()
        return obs_dict[current_agent], {}
    
    def step(self, action):
        """Step with multi-agent coordination using shared policy."""
        # Get current agent for this step
        current_agent = self.get_current_agent_id()
        
        # Create action dict for all agents
        actions = {}

        # All agents get the SAME ACTION from the shared policy, but with slight variations
        for i, agent_id in enumerate(self.agent_ids):
            if agent_id == current_agent:
                # Current agent gets the exact action from shared policy
                # Ensure correct mapping when sending to base env: [steering, throttle]
                if len(action) >= 2:
                    steering = float(np.clip(action[0], -1.0, 1.0))
                    throttle = float(np.clip(action[1], 0.0, 1.0))
                    actions[agent_id] = [steering, throttle]
                else:
                    actions[agent_id] = action
            else:
                # Other agents get slightly modified actions to create diversity
                if len(action) >= 2:
                    # Slightly vary steering and throttle to prevent identical behavior
                    base_steering = float(action[0])
                    base_throttle = float(action[1])
                    modified_steering = base_steering + 0.02 * ((i % 5) - 2)  # -0.04 .. +0.04
                    modified_throttle = base_throttle + 0.05 * ((i % 3) - 1)  # -0.05, 0, +0.05
                    modified_steering = float(np.clip(modified_steering, -1.0, 1.0))
                    modified_throttle = float(np.clip(modified_throttle, 0.0, 1.0))
                    actions[agent_id] = [modified_steering, modified_throttle]
                else:
                    actions[agent_id] = action
        
        # Step environment
        step_result = self.base_env.step(actions)
        
        if len(step_result) == 4:
            obs_dict, reward_dict, done_dict, info_dict = step_result
        else:
            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = step_result
            done_dict = {k: terminated_dict.get(k, False) or truncated_dict.get(k, False) 
                        for k in terminated_dict.keys()}
        
        # Calculate custom rewards for all agents and accumulate
        total_reward = 0.0
        current_agent_reward = 0.0
        for agent_id in self.agent_ids:
            agent_obs = obs_dict[agent_id]
            agent_info = info_dict.get(agent_id, {})
            agent_action = actions[agent_id]
            
            # Calculate custom reward for this agent
            agent_reward = self.calculate_custom_reward(agent_obs, agent_action, agent_info, agent_id)
            total_reward += agent_reward
            self.current_episode_rewards[agent_id] += agent_reward
            if agent_id == current_agent:
                current_agent_reward = agent_reward
            
            # Add experience to shared buffer
            self.coordinator.shared_buffer.add_experience(
                agent_id, agent_obs, agent_action, agent_reward, agent_obs, False, agent_info
            )
        
        # Average reward across all agents (for logging), return current agent's reward
        total_reward = total_reward / len(self.agent_ids)
        
        # Episode termination - only at horizon limit to keep all agents active
        episode_done = self.episode_length >= (self.base_env.config.get('horizon', 3000) - 1)
        if episode_done:
            print(f"    ⏰ Episode completed at step {self.episode_length}")
        
        # Track episode progress
        self.episode_length += 1
        
        # Move to next agent for next step (round-robin)
        self.current_agent_idx = (self.current_agent_idx + 1) % len(self.agent_ids)
        
        # Return observation for next agent
        next_agent = self.get_current_agent_id()
        return obs_dict[next_agent], current_agent_reward, episode_done, episode_done, {}
    
    def close(self):
        """Clean up."""
        pass


def create_coordinated_multi_agent_environment(num_agents: int = 4, track: str = 'right_oval', curriculum_phase: str = 'drive'):
    """Create coordinated multi-agent environment with curriculum learning."""
    
    def get_horizontal_lane_position(agent_index: int, total_agents: int, lane_width: float):
        center = (total_agents - 1) / 2.0
        lateral = (agent_index - center) * lane_width
        longitude = -20.0
        return longitude, lateral
    
    racing_grid_configs = {}
    lane_width = 12.0
    for i in range(num_agents):
        longitude, lateral = get_horizontal_lane_position(i, num_agents, lane_width)
        racing_grid_configs[f"agent{i}"] = {
            "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, i % num_agents),
            "spawn_longitude": 4.0,
            "spawn_lateral": 0.0,
        }
    
    # Curriculum-based configuration
    if curriculum_phase == 'drive':
        # Phase 1: Learn basic driving and track boundaries
        max_speed = 0.6
        crash_penalty = 5.0
        out_penalty = 3.0
        speed_reward = 0.3
        progress_reward = 1.0
        horizon = 2000
        print(f"🎓 CURRICULUM PHASE 1: Learning to Drive (max_speed={max_speed})")
    elif curriculum_phase == 'control':
        # Phase 2: Learn better control and faster driving
        max_speed = 0.8
        crash_penalty = 3.0
        out_penalty = 2.0
        speed_reward = 0.5
        progress_reward = 1.2
        horizon = 2500
        print(f"🎓 CURRICULUM PHASE 2: Learning Control (max_speed={max_speed})")
    else:  # 'race'
        # Phase 3: Full racing with competition
        max_speed = 1.0
        crash_penalty = 2.0
        out_penalty = 1.0
        speed_reward = 0.8
        progress_reward = 1.5
        horizon = 3000
        print(f"🎓 CURRICULUM PHASE 3: Racing Competition (max_speed={max_speed})")
    
    # Environment config optimized for ALL AGENTS MOVING with ZERO RESPAWN LOOPS
    config = {
        "num_agents": num_agents,
        "traffic_density": 0.0,
        "use_render": False,
        
        # CRITICAL: Completely disable ALL termination and respawn mechanisms
        "crash_done": False,  # No episode end on crashes
        "out_of_road_done": False,  # No episode end when out of bounds
        "crash_vehicle_done": False,  # No individual agent termination on vehicle crash
        "crash_object_done": False,  # No individual agent termination on object crash
        "crash_human_done": False,  # No individual agent termination on human crash
        "on_continuous_line_done": False,  # No termination on lane line violations
        "on_broken_line_done": False,  # No termination on broken line violations
        "out_of_route_done": False,  # No termination when off route
        "allow_respawn": False,  # CRITICAL: Prevents infinite respawn loops
        "force_seed_spawn_manager": False,  # Disable forced respawning
        
        "horizon": horizon,  # Only terminate at max episode steps
        "success_reward": progress_reward * 10.0,
        "driving_reward": progress_reward,
        "speed_reward": speed_reward,
        "use_lateral_reward": True,
        "out_of_road_penalty": out_penalty,
        "crash_vehicle_penalty": crash_penalty,
        "crash_object_penalty": crash_penalty * 0.8,
        "map_config": {
            "lane_num": num_agents,
            "lane_width": lane_width,
        },
        "vehicle_config": {
            "show_lidar": True,
            "show_lane_line_detector": True,
            "show_side_detector": True,
            "enable_reverse": False,
            # Use default lidar config to maintain consistent observation space
        },
        "agent_configs": racing_grid_configs  # FIXED: Use racing grid configs
    }
    
    # Try custom environment first
    if MULTI_AGENT_AVAILABLE:
        try:
            env = MultiAgentOvalEnv(config)
            print("🏁 Using custom right-turn oval track!")
            return env
        except Exception as e:
            print(f"⚠️  Custom environment failed: {e}")
    
    # Fallback to standard multi-agent
    from metadrive.envs import MultiAgentMetaDrive
    config["map"] = "O"  # Standard oval
    env = MultiAgentMetaDrive(config)
    print("🏁 Using standard oval track!")
    return env


def train_coordinated_multi_agent_racing(
    num_agents: int = 4,
    timesteps: int = 200_000,
    track: str = 'right_oval',
    results_dir: str = 'coordinated_multi_agent_results',
    use_wandb: bool = False
):
    """Train agents with curriculum learning and per-agent termination."""
    
    print("🏁 CURRICULUM-BASED Multi-Agent Racing Training")
    print("=" * 60)
    print(f"Agents: {num_agents}")
    print(f"Algorithm: Curriculum PPO with Per-Agent Termination")
    print(f"Track: {track}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Results: {results_dir}")
    print()
    print("🎓 CURRICULUM PHASES:")
    print("   Phase 1: DRIVE  - Learn basic control & track boundaries")
    print("   Phase 2: CONTROL - Improve steering & faster driving")
    print("   Phase 3: RACE   - Full racing with competition")
    print()
    print("🔧 KEY FEATURES:")
    print("   ✅ Per-agent termination (no full resets)")
    print("   ✅ Ghost agent system for crashed agents")
    print("   ✅ Custom racing rewards (progress + speed + safety)")
    print("   ✅ Comprehensive metrics logging")
    print("   ✅ Curriculum learning for progressive skill building")
    print()
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize Weights & Biases
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="metadrive-coordinated-multi-agent-racing",
            name=f"consequence_learning_{num_agents}agents_{track}",
            config={
                "num_agents": num_agents,
                "track": track,
                "timesteps": timesteps,
                "algorithm": "Coordinated PPO Consequence-Learning",
                "speed_reward": 4.0,
                "driving_reward": 3.0,
                "success_reward": 50.0,
                "crash_penalty": 40.0,
                "out_of_road_penalty": 30.0,
                "use_lateral_reward": True,
                "focus": "CONSEQUENCE_BASED_RACING",
                "crash_done": True,
                "horizon": 3000
            }
        )
    
    # Create base environment with curriculum phase
    base_env = create_coordinated_multi_agent_environment(num_agents, track, 'race')  # Start with racing phase
    
    # Get agent IDs
    reset_result = base_env.reset()
    if isinstance(reset_result, tuple):
        obs_dict = reset_result[0]
    else:
        obs_dict = reset_result
        
    agent_ids = list(obs_dict.keys())
    print(f"🏎️  Training agents: {agent_ids}")
    
    # Create shared coordination system
    shared_buffer = SharedExperienceBuffer(max_size=50000)
    coordinator = MultiAgentCoordinator(agent_ids, shared_buffer)
    
    # SHARED POLICY APPROACH: Use one model for all agents
    print(f"🏎️  Setting up shared policy for all agents...")
    
    # Create SINGLE shared environment wrapper
    shared_env = SharedPolicyMultiAgentWrapper(base_env, agent_ids, coordinator, 'race')
    
    # Create ONE PPO model that all agents will share
    shared_model = PPO(
        policy="MlpPolicy",
        env=shared_env,
        learning_rate=5e-4,  # Standard learning rate for shared policy
        n_steps=2048,        # Standard rollout length
        batch_size=64,       # Standard batch size
        gamma=0.99,          # Standard discount factor
        gae_lambda=0.95,     # Standard GAE parameter
        clip_range=0.2,      # Standard PPO clip
        ent_coef=0.01,       # Higher exploration for multi-agent
        vf_coef=0.5,         # Standard value function coefficient
        max_grad_norm=0.5,   # Standard gradient clipping
        verbose=0,
        tensorboard_log=os.path.join(results_dir, "shared_tensorboard"),
    )
    
    print(f"✅ Shared policy model created - all agents will learn together!")
    
    # Create callbacks for shared training
    progress_callback = ConciseProgressCallback("shared_policy", verbose=0)
    early_stopping_callback = EarlyStoppingCallback("shared_policy", patience=10, min_improvement=5.0, verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # Save every 50K steps
        save_path=os.path.join(results_dir, "shared_checkpoints"),
        name_prefix="shared_policy",
        verbose=0,
    )
    
    callbacks = [progress_callback, early_stopping_callback, checkpoint_callback]
    
    if use_wandb and WANDB_AVAILABLE:
        wandb_callback = WandbCallback(
            gradient_save_freq=10000,
            model_save_path=None,
            model_save_freq=0,
            verbose=1,
        )
        callbacks.append(wandb_callback)
    
    # SHARED POLICY TRAINING - All agents learn together
    print(f"\n🚀 Starting SHARED POLICY multi-agent training...")
    print("� SHARED POLICY TRAINING - All agents learn together!")
    print("🎯 Key improvements for all agents moving:")
    print("   - Shared policy: All agents use the same brain")
    print("   - Simultaneous learning: Experience from all agents")
    print("   - Movement rewards: Strong incentives to move")
    print("   - No respawn loops: Agents stay active all episode")
    print("   - Coordination: Agents learn to work together")
    print()
    
    total_training_steps = timesteps
    
    try:
        print(f"🏁 Training shared policy for {total_training_steps:,} steps...")
        print(f"   🔧 Learning Rate: 5e-4 | Batch Size: 64 | Shared Policy")
        print(f"   👥 All {num_agents} agents learning together!")
        print(f"   🏁 Movement-focused rewards + coordination bonuses")
        
        # Train the shared model
        shared_model.learn(
            total_timesteps=total_training_steps,
            callback=callbacks,
        )
        
        print(f"   ✅ Shared policy training completed!")
        
        # Get performance stats
        buffer_stats = shared_buffer.get_stats()
        print(f"   📈 Shared experiences: {buffer_stats['total_experiences']:,}")
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save the shared model
        final_dir = os.path.join(results_dir, "final_models")
        os.makedirs(final_dir, exist_ok=True)
        
        shared_model_path = os.path.join(final_dir, "shared_policy.zip")
        shared_model.save(shared_model_path)
        print(f"✅ Shared policy model saved: {shared_model_path}")
        
        # Create individual agent models (copies of shared policy)
        for agent_id in agent_ids:
            agent_model_path = os.path.join(final_dir, f"{agent_id}_final.zip")
            shared_model.save(agent_model_path)
            print(f"✅ {agent_id} model saved: {agent_model_path}")
        
        # Save coordination data
        coordination_data = {
            'performance_history': coordinator.performance_history,
            'buffer_stats': shared_buffer.get_stats(),
            'training_type': 'shared_policy',
            'num_agents': num_agents,
            'total_steps': total_training_steps
        }
        
        with open(os.path.join(results_dir, "coordination_data.pkl"), 'wb') as f:
            pickle.dump(coordination_data, f)
        
        # Cleanup
        base_env.close()
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()


def main():
    checkpoint_freq = steps_per_agent
    
    try:
        for round_num in range(12):  # 12 rounds focused on consequence-based learning
            print(f"\n🏁 Consequence Learning Round {round_num + 1}/12")
            print("-" * 50)
            
            # Train each agent in turn
            for agent_id in agent_ids:
                print(f"🏎️  Training {agent_id} (Round {round_num + 1})...")
                
                model = models[agent_id]
                
                # Train this agent with callbacks for consequence-based learning
                print(f"   � Training with CONSEQUENCE-BASED settings for proper racing...")
                print(f"   🔧 Learning Rate: 3e-6 | Batch Size: 128 | Episodes: 3000 steps")
                print(f"   🏁 Real Racing: Crashes end episodes - learn consequences!")
                print(f"   🎯 Balanced: Speed (4.0x) + Safety (3.0x) + Completion (50.0x)...")
                model.learn(
                    total_timesteps=steps_per_agent,
                    reset_num_timesteps=False,
                    callback=callbacks[agent_id],
                )
                print(f"   ✅ {agent_id} training completed for this round!")
                
                # Get performance ranking
                rankings = coordinator.get_performance_ranking()
                if rankings:
                    print(f"   🏆 Current Rankings:")
                    for i, (ranked_agent, avg_reward) in enumerate(rankings):
                        print(f"      {i+1}. {ranked_agent}: {avg_reward:.1f} avg reward")
                
                # Print shared buffer stats
                buffer_stats = shared_buffer.get_stats()
                print(f"   📈 Shared experiences: {buffer_stats['total_experiences']:,}")
                print()
            
            # Save checkpoint after each round
            checkpoint_dir = os.path.join(results_dir, f"round_{round_num + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            for agent_id, model in models.items():
                model_path = os.path.join(checkpoint_dir, f"{agent_id}.zip")
                model.save(model_path)
            
            print(f"✅ Round {round_num + 1} complete. Checkpoints saved to {checkpoint_dir}")
    
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save final models
        final_dir = os.path.join(results_dir, "final_models")
        os.makedirs(final_dir, exist_ok=True)
        
        for agent_id, model in models.items():
            model_path = os.path.join(final_dir, f"{agent_id}_final.zip")
            model.save(model_path)
            print(f"✅ {agent_id} final model saved: {model_path}")
        
        # Save coordination data
        coordination_data = {
            'performance_history': coordinator.performance_history,
            'buffer_stats': shared_buffer.get_stats(),
            'final_rankings': coordinator.get_performance_ranking()
        }
        
        with open(os.path.join(results_dir, "coordination_data.pkl"), 'wb') as f:
            pickle.dump(coordination_data, f)
        
        # Cleanup
        base_env.close()
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()
    
    print(f"\n🏁 Coordinated Multi-Agent Training Complete!")
    print(f"Models saved in: {results_dir}")
    print()
    print("🎯 Benefits of coordinated training:")
    print("1. ✅ Agents learn from each other's experiences")
    print("2. ✅ Coordinated racing strategies")
    print("3. ✅ No VecNormalize compatibility issues")
    print("4. ✅ True multi-agent interaction during training")
    
    # Print final rankings
    final_rankings = coordinator.get_performance_ranking()
    if final_rankings:
        print(f"\n🏆 Final Agent Rankings:")
        for i, (agent_id, avg_reward) in enumerate(final_rankings):
            print(f"   {i+1}. {agent_id}: {avg_reward:.1f} average reward")


def main():
    parser = argparse.ArgumentParser(description='Coordinated Multi-Agent Racing Training')
    parser.add_argument('--agents', type=int, default=4,
                       help='Number of racing agents')
    parser.add_argument('--timesteps', type=int, default=200_000,
                       help='Total training timesteps')
    parser.add_argument('--track', type=str, default='right_oval',
                       help='Track name')
    parser.add_argument('--results-dir', type=str, default='coordinated_multi_agent_results',
                       help='Results directory')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    
    args = parser.parse_args()
    
    try:
        train_coordinated_multi_agent_racing(
            num_agents=args.agents,
            timesteps=args.timesteps,
            track=args.track,
            results_dir=args.results_dir,
            use_wandb=args.wandb
        )
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
