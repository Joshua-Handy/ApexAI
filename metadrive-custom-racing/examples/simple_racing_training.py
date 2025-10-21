#!/usr/bin/env python3
"""
SIMPLE Multi-Agent Racing Training - No Curriculum, Just Basic Racing
"""
import sys
import os
import numpy as np
from typing import Dict, Any
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from environments.multi_agent_oval_right_env import MultiAgentOvalEnv
from metadrive.component.pgblock.first_block import FirstPGBlock
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym

class SimpleRacingWrapper(gym.Wrapper):
    """Simple wrapper that ensures positive throttle and proper rewards"""
    
    def __init__(self, env):
        super().__init__(env)
        # Get agent IDs from the environment - use different methods based on what's available
        if hasattr(env, 'agent_ids'):
            self.agent_ids = env.agent_ids
        elif hasattr(env, 'agents'):
            self.agent_ids = list(env.agents.keys()) if env.agents else []
        else:
            # Fallback - create agent IDs based on num_agents
            num_agents = getattr(env, 'num_agents', 4)
            self.agent_ids = [f"agent{i}" for i in range(num_agents)]
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs
    
    def step(self, actions):
        """Step with action transformation and simple rewards"""
        # Force strong throttle for all agents to guarantee movement
        transformed_actions = {}
        for agent_id, action in actions.items():
            if len(action) >= 2:
                steering = np.clip(action[0], -0.5, 0.5)
                throttle = 0.8
                transformed_actions[agent_id] = [steering, throttle]
            else:
                transformed_actions[agent_id] = [0.0, 0.8]
        obs, rewards, terminated, truncated, infos = self.env.step(transformed_actions)
        
        # Simple reward shaping: reward for movement and staying on track
        shaped_rewards = {}
        for agent_id in self.agent_ids:
            if agent_id in rewards:
                base_reward = rewards[agent_id]
                
                # Bonus for speed (encourage movement)
                speed_bonus = 0.0
                if agent_id in self.env.agents:
                    vehicle = self.env.agents[agent_id]
                    speed_kmh = vehicle.speed_km_h
                    speed_bonus = min(speed_kmh * 0.01, 0.5)  # Up to 0.5 bonus for speed
                
                # Simple total reward
                shaped_rewards[agent_id] = base_reward + speed_bonus + 0.1  # Base survival bonus
            else:
                shaped_rewards[agent_id] = 0.0
        
        return obs, shaped_rewards, terminated, truncated, infos

def create_simple_racing_env():
    """Create a simple racing environment"""
    
    agent_ids = ["agent0", "agent1", "agent2", "agent3"]
    
    def get_horizontal_lane_position(agent_index: int, total_agents: int, lane_width: float):
        center = (total_agents - 1) / 2.0
        lateral = (agent_index - center) * lane_width
        longitude = -20.0
        return longitude, lateral
    
    racing_grid_configs = {}
    lane_width = 12.0
    for i, agent_id in enumerate(agent_ids):
        longitude, lateral = get_horizontal_lane_position(i, len(agent_ids), lane_width)
        racing_grid_configs[agent_id] = {
            "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, i % len(agent_ids)),
            "spawn_longitude": 4.0,
            "spawn_lateral": 0.0,
        }
    
    # SIMPLE Environment configuration
    env_config = {
        "num_agents": len(agent_ids),
        "traffic_density": 0.0,
        "use_render": False,  # No rendering during training
        
        # Simple done conditions
        "crash_done": False,  # Don't end on crash, just give penalty
        "out_of_road_done": False,  # Don't end on off-road
        "allow_respawn": True,  # Allow respawn to continue learning
        "horizon": 2000,  # Reasonable episode length
        
        "agent_configs": racing_grid_configs,
        
        # Action space: make sure throttle can be positive
        "discrete_action": False,
        
        "map_config": {
            "lane_num": len(agent_ids),
            "lane_width": lane_width,
        }
    }
    
    # Create environment
    base_env = MultiAgentOvalEnv(env_config)
    wrapped_env = SimpleRacingWrapper(base_env)
    
    return wrapped_env

class SimpleProgressCallback(BaseCallback):
    """Simple progress logging"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout"""
        self.episode_count += 1
        
        if self.episode_count % 10 == 0:
            print(f"📊 Completed {self.episode_count} training episodes")
            
            # Try to get some metrics
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                mean_length = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
                print(f"   Recent: reward={mean_reward:.2f}, length={mean_length:.0f}")
        
        return True

def train_simple_racing():
    """Train with simple, direct approach"""
    
    print("🏎️  SIMPLE Multi-Agent Racing Training")
    print("=" * 50)
    
    # Create environment
    print("🏁 Creating simple racing environment...")
    env = create_simple_racing_env()
    
    print(f"   Agents: {env.agent_ids}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Vectorize environment
    def make_env():
        return create_simple_racing_env()
    
    vec_env = DummyVecEnv([make_env])
    
    # Create model with simple configuration
    print("\n🤖 Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        policy_kwargs={
            "net_arch": [64, 64],  # Simple network
            "activation_fn": "tanh"
        }
    )
    
    # Train
    print("\n🏁 Starting training...")
    callback = SimpleProgressCallback()
    
    total_timesteps = 50000  # Shorter training to start
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Save models
    print("\n💾 Saving trained models...")
    os.makedirs("simple_racing_results", exist_ok=True)
    
    # Save the shared model for all agents
    model.save("simple_racing_results/simple_racing_model")
    print("✅ Saved simple_racing_model.zip")
    
    print("\n🎉 Simple training completed!")
    print(f"   Total timesteps: {total_timesteps}")
    print(f"   Model saved to: simple_racing_results/simple_racing_model.zip")
    
    # Close environment
    vec_env.close()

if __name__ == "__main__":
    train_simple_racing()
