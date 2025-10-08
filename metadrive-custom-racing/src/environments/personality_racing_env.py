"""
Personality-aware racing environment for training agents with different racing strategies.
This environment implements personality-based reward shaping for competitive racing training.
"""
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from typing import Dict, Any, Tuple
from environments.single_car_racing import create_racing_environment


class PersonalityRacingEnv(Wrapper):
    """Wrapper environment that adds personality-based rewards to base racing environment."""
    
    def __init__(self, track_name: str = 'right_oval', personality: str = 'balanced', **kwargs):
        self.base_env = create_racing_environment(track_name=track_name, **kwargs)
        super().__init__(self.base_env)
        self.personality = personality
        self.step_count = 0
        self.last_speed = 0.0
        self.aggressive_moves = 0
        self.safe_drives = 0
        self.speed_records = []
        
        # Personality-specific parameters
        self.setup_personality_parameters()
        
    def setup_personality_parameters(self):
        """Configure hybrid personality-specific reward weights and behavior parameters."""
        if self.personality == "conservative_cruiser":
            self.reward_weights = {
                "speed_bonus": 0.3,
                "safety_bonus": 5.0,
                "crash_penalty": -5.0,
                "consistency_bonus": 4.0,
                "distance_bonus": 2.0
            }
        elif self.personality == "aggressive_speedster":
            self.reward_weights = {
                "speed_bonus": 4.0,
                "overtake_bonus": 6.0,
                "crash_penalty": -20.0,
                "risk_bonus": 4.0,
                "aggression_bonus": 3.0
            }
        elif self.personality == "balanced_racer":
            self.reward_weights = {
                "speed_bonus": 2.0,
                "consistency_bonus": 2.0,
                "crash_penalty": -12.0,
                "balance_bonus": 3.0,
                "distance_bonus": 1.5
            }
        elif self.personality == "cautious_speedster":
            self.reward_weights = {
                "speed_bonus": 3.5,
                "safety_bonus": 4.0,
                "crash_penalty": -8.0,
                "careful_speed_bonus": 5.0,
                "distance_bonus": 2.5
            }
        elif self.personality == "aggressive_cruiser":
            self.reward_weights = {
                "speed_bonus": 1.0,
                "overtake_bonus": 8.0,
                "crash_penalty": -15.0,
                "aggression_bonus": 4.0,
                "blocking_bonus": 3.0
            }
        elif self.personality == "speed_demon":
            self.reward_weights = {
                "speed_bonus": 5.0,
                "max_speed_bonus": 8.0,
                "crash_penalty": -25.0,
                "top_speed_bonus": 12.0,
                "distance_bonus": 4.0
            }
        elif self.personality == "conservative_speedster":
            self.reward_weights = {
                "speed_bonus": 3.5,
                "safety_bonus": 5.0,
                "crash_penalty": -6.0,
                "smart_speed_bonus": 6.0,
                "consistency_bonus": 3.0
            }
        elif self.personality == "wild_racer":
            self.reward_weights = {
                "speed_bonus": 4.5,
                "chaos_bonus": 8.0,
                "crash_penalty": -30.0,
                "wild_moves_bonus": 10.0,
                "risk_bonus": 6.0
            }
        # Legacy personality support (backwards compatibility)
        elif self.personality == "aggressive":
            self.reward_weights = {
                "speed_bonus": 2.0,
                "overtake_bonus": 5.0,
                "crash_penalty": -15.0,
                "safety_penalty": -1.0,
                "risk_bonus": 3.0
            }
        elif self.personality == "conservative":
            self.reward_weights = {
                "speed_bonus": 0.5,
                "safety_bonus": 3.0,
                "crash_penalty": -25.0,
                "consistency_bonus": 2.0,
                "risk_penalty": -2.0
            }
        else:  # balanced or fallback
            self.reward_weights = {
                "speed_bonus": 1.0,
                "safety_bonus": 1.0,
                "crash_penalty": -15.0,
                "consistency_bonus": 1.0,
                "balance_bonus": 2.0
            }
    
    def calculate_personality_reward(self, base_reward: float, info: Dict, action: np.ndarray) -> float:
        """Calculate additional reward based on hybrid personality traits."""
        personality_reward = 0.0
        current_speed = info.get('speed', 0.0)
        
        # Track behavioral metrics
        self.speed_records.append(current_speed)
        if len(self.speed_records) > 100:  # Keep last 100 speed records
            self.speed_records.pop(0)
            
        # Universal speed-based rewards
        if current_speed > 0.6 and "speed_bonus" in self.reward_weights:
            personality_reward += self.reward_weights["speed_bonus"] * (current_speed - 0.6)
        
        # High-speed rewards for speedster types
        if "speedster" in self.personality or "speed_demon" in self.personality:
            if current_speed > 0.8:
                if "max_speed_bonus" in self.reward_weights:
                    personality_reward += self.reward_weights["max_speed_bonus"]
                if "top_speed_bonus" in self.reward_weights:
                    personality_reward += self.reward_weights["top_speed_bonus"] * (current_speed - 0.8)
        
        # Conservative driving rewards
        if "conservative" in self.personality or "cautious" in self.personality:
            # Reward steady, consistent speed
            if 0.4 <= current_speed <= 0.7:
                if "safety_bonus" in self.reward_weights:
                    personality_reward += self.reward_weights["safety_bonus"]
                self.safe_drives += 1
                
            # Consistency bonus
            if len(self.speed_records) > 10:
                speed_variance = np.var(self.speed_records[-10:])
                if speed_variance < 0.1 and "consistency_bonus" in self.reward_weights:
                    personality_reward += self.reward_weights["consistency_bonus"]
        
        # Aggressive driving behaviors
        if "aggressive" in self.personality:
            # Reward aggressive acceleration
            if current_speed > self.last_speed + 0.1:
                if "risk_bonus" in self.reward_weights:
                    personality_reward += self.reward_weights["risk_bonus"]
                if "aggression_bonus" in self.reward_weights:
                    personality_reward += self.reward_weights["aggression_bonus"]
                self.aggressive_moves += 1
        
        # Wild racer unpredictable bonuses
        if "wild" in self.personality:
            # Random bonus for unpredictable behavior
            if abs(current_speed - self.last_speed) > 0.2:
                if "wild_moves_bonus" in self.reward_weights:
                    personality_reward += self.reward_weights["wild_moves_bonus"] * 0.5
            if "chaos_bonus" in self.reward_weights:
                personality_reward += self.reward_weights["chaos_bonus"] * np.random.uniform(0, 0.1)
        
        # Balanced rewards
        if "balanced" in self.personality:
            if 0.5 <= current_speed <= 0.8:
                if "balance_bonus" in self.reward_weights:
                    personality_reward += self.reward_weights["balance_bonus"]
        
        # Distance progression bonus for all personalities
        if "distance_bonus" in self.reward_weights and base_reward > 0:
            personality_reward += self.reward_weights["distance_bonus"] * 0.1
        
        # Common crash penalty for all personalities
        if info.get('crash', False) or base_reward < -5:
            personality_reward += self.reward_weights["crash_penalty"]
            
        self.last_speed = current_speed
        return personality_reward
    
    def reset(self, **kwargs):
        """Reset environment and personality metrics."""
        self.step_count = 0
        self.last_speed = 0.0
        self.aggressive_moves = 0
        self.safe_drives = 0
        self.speed_records = []
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step environment and add personality-based rewards."""
        obs, base_reward, done, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # Add personality-specific reward
        personality_reward = self.calculate_personality_reward(base_reward, info, action)
        total_reward = base_reward + personality_reward
        
        # Add personality metrics to info
        info['personality_reward'] = personality_reward
        info['base_reward'] = base_reward
        info['personality'] = self.personality
        info['aggressive_moves'] = self.aggressive_moves
        info['safe_drives'] = self.safe_drives
        info['avg_speed'] = np.mean(self.speed_records) if self.speed_records else 0.0
        
        return obs, total_reward, done, truncated, info
    
    def __getattr__(self, name):
        """Delegate missing attributes to base environment."""
        return getattr(self.env, name)


def create_personality_racing_environment(
    track_name: str = 'right_oval',
    personality: str = 'balanced',
    use_render: bool = False,
    start_seed: int = None,
    **kwargs
):
    """Create a personality-aware racing environment for training."""
    return PersonalityRacingEnv(
        track_name=track_name,
        personality=personality,
        use_render=use_render,
        start_seed=start_seed,
        **kwargs
    )