"""
AI Multi-Agent Racing Championship using custom right_oval track.

This script creates multiple AI-controlled cars racing simultaneously on the custom right_oval track.
Each car uses different hybrid personalities combining speed preferences with driving styles.
"""
import argparse
import os
import sys
import time
import glob
import numpy as np
import random
from typing import List, Dict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from metadrive.envs import MultiAgentMetaDrive
    from environments.multi_agent_oval_right_env import MultiAgentOvalEnv
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    print("Warning: MultiAgentMetaDrive not available")
    MULTI_AGENT_AVAILABLE = False


class AIRacingPolicy:
    """Simple AI racing policy with personality-based rewards."""
    
    def __init__(self, name: str, strategy: str = "balanced"):
        self.name = name
        self.strategy = strategy
        self.step_count = 0
        self.total_distance = 0.0
        self.crashes = 0
        self.aggressive_moves = 0
        self.safe_drives = 0
        self.speed_records = []
        self.personality_score = 0.0
        
        # Hybrid personality system: speed preference + driving style
        if strategy == "conservative_cruiser":
            self.max_speed = 0.4
            self.steering_sensitivity = 0.3
            self.brake_threshold = 0.6
            self.reward_weights = {
                "speed_bonus": 0.5,
                "consistency_bonus": 5.0,
                "crash_penalty": -3.0,
                "safety_bonus": 3.0,
                "distance_bonus": 2.0
            }
        elif strategy == "aggressive_speedster":
            self.max_speed = 0.9
            self.steering_sensitivity = 0.8
            self.brake_threshold = 0.2
            self.reward_weights = {
                "speed_bonus": 4.0,
                "overtake_bonus": 6.0,
                "crash_penalty": -20.0,
                "risk_bonus": 4.0,
                "aggression_bonus": 2.0
            }
        elif strategy == "balanced_racer":
            self.max_speed = 0.7
            self.steering_sensitivity = 0.5
            self.brake_threshold = 0.4
            self.reward_weights = {
                "speed_bonus": 2.0,
                "consistency_bonus": 2.0,
                "crash_penalty": -10.0,
                "balance_bonus": 3.0,
                "distance_bonus": 1.5
            }
        elif strategy == "cautious_speedster":
            self.max_speed = 0.8
            self.steering_sensitivity = 0.4
            self.brake_threshold = 0.5
            self.reward_weights = {
                "speed_bonus": 3.0,
                "safety_bonus": 4.0,
                "crash_penalty": -5.0,
                "careful_speed_bonus": 5.0,
                "distance_bonus": 2.5
            }
        elif strategy == "aggressive_cruiser":
            self.max_speed = 0.5
            self.steering_sensitivity = 0.7
            self.brake_threshold = 0.3
            self.reward_weights = {
                "speed_bonus": 1.0,
                "overtake_bonus": 8.0,
                "crash_penalty": -15.0,
                "aggression_bonus": 4.0,
                "blocking_bonus": 3.0
            }
        elif strategy == "speed_demon":
            self.max_speed = 1.0
            self.steering_sensitivity = 0.6
            self.brake_threshold = 0.2
            self.reward_weights = {
                "speed_bonus": 6.0,
                "top_speed_bonus": 12.0,
                "crash_penalty": -25.0,
                "brake_penalty": -3.0,
                "distance_bonus": 4.0
            }
        elif strategy == "conservative_speedster":
            self.max_speed = 0.8
            self.steering_sensitivity = 0.3
            self.brake_threshold = 0.6
            self.reward_weights = {
                "speed_bonus": 3.5,
                "safety_bonus": 5.0,
                "crash_penalty": -2.0,
                "smart_speed_bonus": 6.0,
                "consistency_bonus": 3.0
            }
        elif strategy == "wild_racer":
            self.max_speed = 1.0
            self.steering_sensitivity = 0.9
            self.brake_threshold = 0.1
            self.reward_weights = {
                "speed_bonus": 5.0,
                "chaos_bonus": 8.0,
                "crash_penalty": -30.0,
                "wild_moves_bonus": 10.0,
                "risk_bonus": 6.0
            }
        else:  # fallback to balanced
            self.max_speed = 0.7
            self.steering_sensitivity = 0.5
            self.brake_threshold = 0.4
            # Balanced rewards: moderate bonuses for everything
            self.reward_weights = {
                "speed_bonus": 1.0,
                "consistency_bonus": 2.0,
                "crash_penalty": -10.0,
                "safety_bonus": 1.0,
                "distance_bonus": 1.0
            }
    
    def calculate_personality_reward(self, base_reward: float, info: dict, current_speed: float = 0.0):
        """Calculate additional reward based on agent personality."""
        personality_reward = 0.0
        
        # Track performance metrics
        self.speed_records.append(current_speed)
        
        # Speed-based rewards
        if current_speed > 0.7:  # High speed
            personality_reward += self.reward_weights.get("speed_bonus", 0) * current_speed
            if self.strategy == "speed_demon" and current_speed > 0.9:
                personality_reward += self.reward_weights.get("top_speed_bonus", 0)
        
        # Safety and consistency rewards
        if self.strategy == "conservative":
            # Reward consistent moderate speed
            if 0.3 <= current_speed <= 0.6:
                personality_reward += self.reward_weights.get("consistency_bonus", 0)
                self.safe_drives += 1
            # Bonus for distance traveled safely
            if self.step_count > 50 and not info.get("crash_vehicle", False):
                personality_reward += self.reward_weights.get("safety_bonus", 0)
        
        # Aggressive behavior rewards
        if self.strategy == "aggressive":
            # Reward risky driving (high speed + close to obstacles)
            if current_speed > 0.6:
                personality_reward += self.reward_weights.get("risk_bonus", 0)
                self.aggressive_moves += 1
            # Penalty for being too conservative
            if current_speed < 0.3:
                personality_reward += self.reward_weights.get("safety_penalty", 0)
        
        # Speed demon specific rewards
        if self.strategy == "speed_demon":
            # Massive bonus for maintaining top speed
            if current_speed > 0.8:
                personality_reward += self.reward_weights.get("top_speed_bonus", 0)
            # Penalty for braking/slowing down
            if len(self.speed_records) > 1 and current_speed < self.speed_records[-2]:
                personality_reward += self.reward_weights.get("brake_penalty", 0)
        
        # Crash penalties (personality-specific)
        if info.get("crash_vehicle", False) or info.get("crash_object", False) or info.get("out_of_road", False):
            personality_reward += self.reward_weights.get("crash_penalty", 0)
            self.crashes += 1
        
        # Distance bonus for all strategies
        if "distance_bonus" in self.reward_weights:
            personality_reward += self.reward_weights["distance_bonus"] * 0.1
        
        # Update personality score
        self.personality_score += personality_reward
        
        return base_reward + personality_reward
    
    def act(self, observation):
        """Generate racing action based on strategy and observation."""
        self.step_count += 1
        
        # Add startup delay based on strategy to prevent initial crashes
        agent_num = 0
        if hasattr(self, 'name') and 'agent' in str(self.name):
            try:
                agent_num = int(str(self.name).replace('agent', ''))
            except:
                agent_num = 0
        
        startup_delay = agent_num * 60  # Extended delay (60 steps per agent)
        if self.step_count <= startup_delay:
            # Wait before starting racing to space out agents
            return [0.0, 0.02]  # Minimal forward movement during delay
        
        try:
            # Extract relevant information from observation with robust parsing
            if len(observation) >= 91:  # MetaDrive observation format
                # Get LIDAR data with proper indexing
                lidar_readings = observation[:240] if len(observation) >= 240 else observation[:min(72, len(observation))]
                
                # Vehicle state information
                speed_idx = min(72, len(observation)-1)
                current_speed = observation[speed_idx] if speed_idx < len(observation) else 0.0
                
                # Enhanced 360-degree situational awareness
                total_readings = len(lidar_readings)
                if total_readings >= 72:
                    # Map LIDAR readings to directions (assuming 240-point LIDAR)
                    readings_per_sector = max(1, total_readings // 8)
                    
                    # Front sensors (critical for collision avoidance)
                    front_center = np.mean(lidar_readings[total_readings//2-readings_per_sector//2:total_readings//2+readings_per_sector//2])
                    front_left = np.mean(lidar_readings[total_readings//2+readings_per_sector:total_readings//2+2*readings_per_sector])
                    front_right = np.mean(lidar_readings[total_readings//2-2*readings_per_sector:total_readings//2-readings_per_sector])
                    
                    # Side sensors
                    left_side = np.mean(lidar_readings[3*total_readings//4:total_readings-1]) if total_readings > 4 else 1.0
                    right_side = np.mean(lidar_readings[1:total_readings//4]) if total_readings > 4 else 1.0
                    
                    # Calculate minimum safe distances
                    min_front = min(front_center, front_left, front_right)
                else:
                    # Simplified for smaller LIDAR arrays
                    front_center = np.mean(lidar_readings[len(lidar_readings)//2-2:len(lidar_readings)//2+2])
                    left_side = np.mean(lidar_readings[:len(lidar_readings)//4])
                    right_side = np.mean(lidar_readings[-len(lidar_readings)//4:])
                    min_front = front_center
                
                # Intelligent decision making with progressive responses
                steering = 0.0
                throttle = 0.3  # Conservative base speed
                
                # Multi-level collision avoidance system with personality-based thresholds
                if "conservative" in self.strategy or "cautious" in self.strategy:
                    critical_distance = 0.20   # More cautious
                    warning_distance = 0.6     # Earlier warnings
                    safe_distance = 1.0        # Larger safety buffer
                elif "aggressive" in self.strategy or "wild" in self.strategy:
                    critical_distance = 0.10   # Risk-taking
                    warning_distance = 0.3     # Late warnings
                    safe_distance = 0.5        # Smaller safety buffer
                else:  # balanced, speed_demon
                    critical_distance = 0.15   # Moderate
                    warning_distance = 0.4     # Standard warnings
                    safe_distance = 0.8        # Normal safety buffer
                
                if min_front < critical_distance:
                    # EMERGENCY: Immediate evasive action
                    throttle = -0.8  # Emergency brake
                    
                    # Choose best escape route
                    if left_side > right_side and left_side > 0.3:
                        steering = -0.8  # Sharp left
                    elif right_side > 0.3:
                        steering = 0.8   # Sharp right
                    else:
                        steering = 0.0   # Straight brake if no escape
                        
                elif min_front < warning_distance:
                    # WARNING: Prepare for collision
                    throttle = 0.1  # Slow down significantly
                    
                    # Gentle avoidance maneuver
                    if left_side > right_side + 0.1:
                        steering = -0.5  # Moderate left
                    elif right_side > left_side + 0.1:
                        steering = 0.5   # Moderate right
                    
                elif min_front < safe_distance:
                    # CAUTION: Maintain safe distance
                    throttle = 0.2
                    
                    # Subtle positioning adjustment
                    if left_side > right_side + 0.05:
                        steering = -0.2
                    elif right_side > left_side + 0.05:
                        steering = 0.2
                
                else:
                    # CLEAR: Normal racing with hybrid personality-based behavior
                    base_throttle = self.max_speed * 0.8  # Use personality max speed
                    
                    if "conservative" in self.strategy:
                        # Conservative driving styles - prioritize safety
                        throttle = min(base_throttle, throttle + 0.1)
                        steering += random.uniform(-0.02, 0.02)  # Very smooth steering
                        
                    elif "aggressive" in self.strategy:
                        # Aggressive driving styles - take risks for position
                        throttle = min(base_throttle, throttle + 0.4)
                        steering += random.uniform(-0.08, 0.08)  # More erratic steering
                        
                        # Aggressive overtaking attempts
                        if min_front < safe_distance * 1.2:  # Earlier overtaking attempts
                            if left_side > right_side + 0.1:
                                steering -= 0.3  # Dive to the left
                            elif right_side > left_side + 0.1:
                                steering += 0.3  # Dive to the right
                                
                    elif "cautious" in self.strategy:
                        # High speed but careful - wait for clear opportunities
                        if min_front > safe_distance * 1.5:  # Only speed up when very clear
                            throttle = min(base_throttle, throttle + 0.5)
                        else:
                            throttle = min(0.4, throttle + 0.1)  # Very conservative in traffic
                        steering += random.uniform(-0.01, 0.01)  # Ultra-smooth steering
                        
                    elif "speed_demon" in self.strategy:
                        # Pure speed focus
                        throttle = min(base_throttle, throttle + 0.7)
                        
                    elif "wild" in self.strategy:
                        # Unpredictable and risky
                        throttle = min(base_throttle, throttle + 0.6)
                        steering += random.uniform(-0.15, 0.15)  # Very erratic
                        
                        # Wild moves - sudden lane changes
                        if random.random() < 0.02:  # 2% chance per step
                            steering += random.choice([-0.5, 0.5])
                            
                    elif "balanced" in self.strategy:
                        # Balanced approach
                        throttle = min(base_throttle, throttle + 0.3)
                        steering += random.uniform(-0.03, 0.03)
                        
                    else:  # fallback
                        throttle = min(0.7, throttle + 0.3)
                
                # Side boundary protection
                if left_side < 0.3:
                    steering += 0.3  # Move away from left obstacle
                if right_side < 0.3:
                    steering -= 0.3  # Move away from right obstacle
                
                # Smooth steering to prevent oscillation
                if hasattr(self, 'last_steering'):
                    steering_change = steering - self.last_steering
                    max_change = 0.4
                    if abs(steering_change) > max_change:
                        steering = self.last_steering + np.sign(steering_change) * max_change
                
                self.last_steering = steering
                
                # Final safety limits
                steering = np.clip(steering, -1.0, 1.0)
                throttle = np.clip(throttle, -1.0, 1.0)
                
                return [steering, throttle]
            
            else:
                # Fallback for unexpected observation format
                return [0.0, 0.3]
                
        except Exception as e:
            print(f"Error in {self.name} act(): {e}")
            return [0.0, 0.2]  # Safe fallback action


def create_racing_environment(num_agents: int = 4):
    """Create multi-agent racing environment with custom right_oval track."""
    
    # Configuration for custom right_oval environment with spread-out starts
    custom_config = {
        "num_agents": num_agents,
        "start_seed": random.randint(1, 1000),
        "traffic_density": 0.0,
        "use_render": True,
        "crash_done": True,
        "out_of_road_done": True,
        "horizon": 5000,  # Longer races for more action
        "success_reward": 20.0,
        "driving_reward": 2.0,
        "speed_reward": 1.0,
        "out_of_road_penalty": 3.0,
        "crash_vehicle_penalty": 8.0,
        "map_config": {
            "lane_num": 1,  # Single lane racing
            "lane_width": 20.0,  # Match right_oval config
        },
        "vehicle_config": {
            "show_lidar": False,
            "show_lane_line_detector": False,
            "show_side_detector": False,
            "enable_reverse": False,
        },
        # Custom spawn configuration to spread agents out
        "agent_configs": {
            f"agent{i}": {
                "spawn_longitude": i * -80.0,  # Massive spacing - 80 units apart
                "spawn_lateral": (i % 2) * 10.0 - 5.0,  # Even wider lateral offset
            } for i in range(num_agents)
        }
    }
    
    # Configuration for standard multi-agent environment with spread-out starts
    standard_config = {
        "num_agents": num_agents,
        "map": "O",  # Standard oval track
        "start_seed": random.randint(1, 1000),
        "traffic_density": 0.0,
        "use_render": True,
        "crash_done": True,
        "out_of_road_done": True,
        "horizon": 5000,  # Longer races
        "success_reward": 20.0,
        "driving_reward": 2.0,
        "speed_reward": 1.0,
        "out_of_road_penalty": 3.0,
        "crash_vehicle_penalty": 8.0,
        "vehicle_config": {
            "show_lidar": False,
            "show_lane_line_detector": False,
            "show_side_detector": False,
            "enable_reverse": False,
        },
        # Custom spawn configuration to spread agents out
        "agent_configs": {
            f"agent{i}": {
                "spawn_longitude": i * -35.0,  # More space - 35 units apart
                "spawn_lateral": (i % 2) * 4.0 - 2.0,  # Wider staggered positioning
            } for i in range(num_agents)
        }
    }
    
    # Try to use custom right_oval environment first
    if MULTI_AGENT_AVAILABLE:
        try:
            env = MultiAgentOvalEnv(custom_config)
            print("🏁 Using custom right-turn oval track (single lane racing with spread-out starts)!")
            return env
        except Exception as e:
            print(f"⚠️  Custom right_oval failed ({e}), using standard track")
    
    # Fallback to standard multi-agent environment
    env = MultiAgentMetaDrive(standard_config)
    print("🏁 Using standard oval racing track with spread-out starts!")
    return env


def run_multi_agent_race(num_agents: int = 4, episodes: int = 3):
    """Run a multi-agent race with different AI strategies."""
    
    print(f"\n🏁 Starting {num_agents} AI racers!")
    print("=" * 60)
    
    # Create selected hybrid racing personalities (4 chosen personalities)
    strategies = [
        "cautious_speedster",     # High speed, but careful driving
        "balanced_racer",         # Medium speed, balanced approach
        "aggressive_speedster",   # High speed, risky driving  
        "conservative_cruiser",   # Low speed, very safe driving
    ]
    agent_names = [
        "Careful Carl",       # cautious_speedster
        "Balanced Bob",       # balanced_racer
        "Rapid Rick",         # aggressive_speedster  
        "Safe Sam",           # conservative_cruiser
    ]
    
    # Create environment first to get actual agent keys
    env = create_racing_environment(num_agents)
    
    # Reset once to get agent keys
    observations = env.reset()
    if isinstance(observations, tuple):
        observations = observations[0]
    
    # Use actual agent keys from environment
    agent_ids = list(observations.keys())
    
    # Create AI policies
    policies = {}
    for i, agent_id in enumerate(agent_ids):
        strategy = strategies[i % len(strategies)]
        name = agent_names[i % len(agent_names)]
        if i >= len(agent_names):
            name = f"{name} {i+1}"
        policies[agent_id] = AIRacingPolicy(name, strategy)
    
    print("🏎️  Racing lineup:")
    for agent_id, policy in policies.items():
        print(f"   {agent_id}: {policy.name} ({policy.strategy})")
    
    try:
        for episode in range(episodes):
            print(f"\n🏁 Race {episode + 1}/{episodes}")
            print("-" * 40)
            
            # Reset environment (but keep the same agent keys)
            if episode > 0:  # Only reset for subsequent episodes
                observations = env.reset()
                if isinstance(observations, tuple):
                    observations = observations[0]
            
            # Race tracking
            total_rewards = {agent_id: 0.0 for agent_id in agent_ids}
            active_agents = set(agent_ids)  # Track which agents are still racing
            crashed_agents = set()  # Track which agents have crashed
            race_steps = 0
            max_steps = 3000
            
            print("🚦 Lights out and away we go!")
            
            # Run the race
            while race_steps < max_steps:
                # Get actions only for active agents that haven't crashed
                actions = {}
                
                # Only generate actions for agents that are still active and in observations
                for agent_id in observations.keys():
                    if agent_id in active_agents and agent_id in policies:
                        obs = observations[agent_id]
                        action = policies[agent_id].act(obs)
                        actions[agent_id] = action
                    elif agent_id in crashed_agents:
                        # Crashed agents get no action (they should be removed)
                        continue
                    else:
                        # Fallback for any edge cases
                        actions[agent_id] = [0.0, 0.0]  # No throttle, no steering
                
                # Step environment
                step_result = env.step(actions)
                
                if len(step_result) == 4:
                    observations, rewards, done, infos = step_result
                else:
                    observations, rewards, terminated, truncated, infos = step_result
                    done = terminated
                
                # Update rewards with personality-based scoring
                if isinstance(rewards, dict):
                    for agent_id in observations.keys():
                        if agent_id in rewards and agent_id in total_rewards and agent_id in policies:
                            base_reward = rewards[agent_id]
                            
                            # Get current speed from observation
                            obs = observations[agent_id]
                            current_speed = 0.0
                            if len(obs) >= 91:
                                speed_idx = min(72, len(obs)-1)
                                current_speed = obs[speed_idx] if speed_idx < len(obs) else 0.0
                            
                            # Get info for personality reward calculation
                            agent_info = {}
                            if isinstance(infos, dict) and agent_id in infos:
                                agent_info = infos[agent_id]
                            
                            # Calculate personality-based reward
                            policy = policies[agent_id]
                            personality_reward = policy.calculate_personality_reward(base_reward, agent_info, current_speed)
                            total_rewards[agent_id] += personality_reward
                
                # Check for individual agent crashes/done states
                if isinstance(done, dict):
                    for agent_id, is_done in done.items():
                        if is_done and agent_id != "__all__" and agent_id in policies and agent_id in active_agents:
                            # Agent has crashed or finished - remove from active agents
                            active_agents.discard(agent_id)
                            crashed_agents.add(agent_id)
                            
                            # Check crash reason and display
                            if isinstance(infos, dict) and agent_id in infos:
                                info = infos[agent_id]
                                crash_reason = ""
                                if info.get("crash_vehicle", False):
                                    crash_reason = "💥 vehicle collision"
                                elif info.get("crash_object", False):
                                    crash_reason = "💥 object collision"
                                elif info.get("out_of_road", False):
                                    crash_reason = "🚧 went off track"
                                elif info.get("lane_line_collision", False):
                                    crash_reason = "⚠️ hit track edge"
                                else:
                                    crash_reason = "🏁 finished"
                                
                                policy = policies[agent_id]
                                print(f"   🚨 {policy.name} ({policy.strategy}) - {crash_reason} [ELIMINATED]")
                
                race_steps += 1
                
                # Check if all agents are done (crashed or finished)
                all_agents_done = len(active_agents) == 0
                if isinstance(done, dict):
                    # Also check the global done flag
                    if done.get("__all__", False) or all_agents_done:
                        if all_agents_done:
                            print(f"   🏁 All agents eliminated! Race ended at step {race_steps}")
                        break
                elif done:
                    break
                
                # Print active agents count every 500 steps
                if race_steps % 500 == 0:
                    active_count = len(active_agents)
                    print(f"   Lap progress: {race_steps}/{max_steps} steps | Active agents: {active_count}")
                
                # Small delay for smooth visualization
                time.sleep(0.01)
            
            # Race results with personality metrics
            print(f"\n🏆 Race {episode + 1} Results (after {race_steps} steps):")
            sorted_results = sorted(total_rewards.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (agent_id, reward) in enumerate(sorted_results, 1):
                policy = policies[agent_id]
                emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "🏎️"
                
                # Personality-specific stats
                personality_stats = ""
                if policy.strategy == "aggressive":
                    personality_stats = f" | Aggressive moves: {policy.aggressive_moves}"
                elif policy.strategy == "conservative":
                    personality_stats = f" | Safe drives: {policy.safe_drives}"
                elif policy.strategy == "speed_demon":
                    avg_speed = np.mean(policy.speed_records) if policy.speed_records else 0
                    personality_stats = f" | Avg speed: {avg_speed:.2f}"
                elif policy.strategy == "balanced":
                    personality_stats = f" | Steps: {policy.step_count}"
                
                print(f"   {emoji} {rank}. {policy.name} ({policy.strategy}): {reward:.2f} points{personality_stats}")
                print(f"      💎 Personality score: {policy.personality_score:.2f} | Crashes: {policy.crashes}")
            
            if episode < episodes - 1:
                input("\n⏸️  Press Enter for next race...")
        
        print(f"\n🏁 Racing session completed!")
        print("   Thank you for watching the AI Racing Championship!")
    
    except KeyboardInterrupt:
        print("\n🛑 Racing stopped by user")
    
    finally:
        try:
            env.close()
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description='AI Multi-Agent Racing Championship')
    parser.add_argument('--agents', type=int, default=4,
                       help='Number of AI racing agents (2-8)')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of races to run')
    
    args = parser.parse_args()
    
    if not MULTI_AGENT_AVAILABLE:
        print("❌ MultiAgentMetaDrive not available!")
        print("   This demo requires MetaDrive with multi-agent support.")
        return
    
    # Limit agents to reasonable range
    num_agents = max(2, min(8, args.agents))
    
    print("🏁 AI Multi-Agent Racing Championship")
    print("=" * 50)
    print(f"🏎️  Number of AI racers: {num_agents}")
    print(f"🏆 Number of races: {args.episodes}")
    print(f"🎯 Each AI has a different racing strategy!")
    print("\n🎮 Controls:")
    print("   - Watch the AI racers compete")
    print("   - Press Ctrl+C to stop anytime")
    print("   - Close the window to exit")
    
    input("\n🚦 Press Enter to start the AI Racing Championship...")
    
    try:
        run_multi_agent_race(num_agents, args.episodes)
    except KeyboardInterrupt:
        print("\n🛑 Championship stopped by user")


if __name__ == '__main__':
    main()