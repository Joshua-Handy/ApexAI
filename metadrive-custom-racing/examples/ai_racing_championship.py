"""
AI Multi-Agent Racing Championship using custom right_oval track.

This script creates multiple AI-controlled cars racing simultaneously on the custom right_oval track.
Each car uses different strategies (aggressive, conservative, speed demon, balanced).
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
    """Simple AI racing policy for demonstration."""
    
    def __init__(self, name: str, strategy: str = "balanced"):
        self.name = name
        self.strategy = strategy
        self.step_count = 0
        
        # Different racing strategies
        if strategy == "aggressive":
            self.max_speed = 0.8
            self.steering_sensitivity = 0.6
            self.brake_threshold = 0.3
        elif strategy == "conservative":
            self.max_speed = 0.5
            self.steering_sensitivity = 0.4
            self.brake_threshold = 0.5
        elif strategy == "speed_demon":
            self.max_speed = 1.0
            self.steering_sensitivity = 0.8
            self.brake_threshold = 0.2
        else:  # balanced
            self.max_speed = 0.7
            self.steering_sensitivity = 0.5
            self.brake_threshold = 0.4
    
    def act(self, observation):
        """Generate racing action based on strategy and observation."""
        self.step_count += 1
        
        try:
            # Extract relevant information from observation
            if len(observation) >= 91:  # MetaDrive observation format
                # Lidar readings (first 240 values typically)
                lidar_start = 0
                lidar_readings = observation[lidar_start:lidar_start+72] if len(observation) > 72 else observation[:min(72, len(observation))]
                
                # Vehicle state (speed, position, etc.)
                speed_idx = min(72, len(observation)-1)
                current_speed = observation[speed_idx] if speed_idx < len(observation) else 0.0
                
                # Simple racing logic based on lidar
                front_distance = np.mean(lidar_readings[30:42]) if len(lidar_readings) > 42 else 1.0
                left_distance = np.mean(lidar_readings[0:24]) if len(lidar_readings) > 24 else 1.0
                right_distance = np.mean(lidar_readings[48:72]) if len(lidar_readings) > 48 else 1.0
                
                # Calculate actions
                throttle = self.max_speed
                steering = 0.0
                
                # Obstacle avoidance
                if front_distance < self.brake_threshold:
                    throttle = -0.3  # Brake
                    if left_distance > right_distance:
                        steering = -self.steering_sensitivity  # Turn left
                    else:
                        steering = self.steering_sensitivity   # Turn right
                else:
                    # Normal racing - slight corrections
                    if left_distance < 0.3:
                        steering = 0.2  # Slight right
                    elif right_distance < 0.3:
                        steering = -0.2  # Slight left
                
                # Add some strategy-specific behavior
                if self.strategy == "aggressive":
                    steering *= 1.2  # More aggressive steering
                elif self.strategy == "conservative":
                    throttle *= 0.8  # More cautious speed
                elif self.strategy == "speed_demon":
                    if front_distance > 0.5:
                        throttle = 1.0  # Full speed when clear
                
                # Add slight random variation to make it more interesting
                steering += random.uniform(-0.1, 0.1)
                throttle += random.uniform(-0.1, 0.1)
                
                # Clamp values
                throttle = np.clip(throttle, -1.0, 1.0)
                steering = np.clip(steering, -1.0, 1.0)
                
                return [steering, throttle]
            
            else:
                # Fallback for unknown observation format
                return [random.uniform(-0.3, 0.3), random.uniform(0.3, 0.8)]
                
        except Exception as e:
            # Fallback action
            return [0.0, 0.5]


def create_racing_environment(num_agents: int = 4):
    """Create multi-agent racing environment with custom right_oval track."""
    
    # Configuration for custom right_oval environment
    custom_config = {
        "num_agents": num_agents,
        "start_seed": random.randint(1, 1000),
        "traffic_density": 0.0,
        "use_render": True,
        "crash_done": True,
        "out_of_road_done": True,
        "horizon": 3000,  # Longer races
        "success_reward": 10.0,
        "driving_reward": 1.0,
        "speed_reward": 0.2,
        "out_of_road_penalty": 5.0,
        "crash_vehicle_penalty": 10.0,
        "map_config": {
            "lane_num": 1,
            "lane_width": 20.0,  # Match right_oval config
        },
        "vehicle_config": {
            "show_lidar": False,
            "show_lane_line_detector": False,
            "show_side_detector": False,
            "enable_reverse": False,
        }
    }
    
    # Configuration for standard multi-agent environment
    standard_config = {
        "num_agents": num_agents,
        "map": "O",  # Standard oval track
        "start_seed": random.randint(1, 1000),
        "traffic_density": 0.0,
        "use_render": True,
        "crash_done": True,
        "out_of_road_done": True,
        "horizon": 3000,  # Longer races
        "success_reward": 10.0,
        "driving_reward": 1.0,
        "speed_reward": 0.2,
        "out_of_road_penalty": 5.0,
        "crash_vehicle_penalty": 10.0,
        "vehicle_config": {
            "show_lidar": False,
            "show_lane_line_detector": False,
            "show_side_detector": False,
            "enable_reverse": False,
        }
    }
    
    # Try to use custom right_oval environment first
    if MULTI_AGENT_AVAILABLE:
        try:
            env = MultiAgentOvalEnv(custom_config)
            print("🏁 Using custom right-turn oval track (PGBlocks)!")
            return env
        except Exception as e:
            print(f"⚠️  Custom right_oval failed ({e}), using standard track")
    
    # Fallback to standard multi-agent environment
    env = MultiAgentMetaDrive(standard_config)
    print("🏁 Using standard oval racing track for multi-agent competition!")
    return env


def run_multi_agent_race(num_agents: int = 4, episodes: int = 3):
    """Run a multi-agent race with different AI strategies."""
    
    print(f"\n🏁 Starting {num_agents} AI racers!")
    print("=" * 60)
    
    # Create different racing strategies
    strategies = ["aggressive", "conservative", "speed_demon", "balanced"]
    agent_names = [
        "Lightning McQueen",  # speed_demon
        "Professor Prudent",  # conservative  
        "Turbo Tornado",     # aggressive
        "Steady Steve",      # balanced
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
            race_steps = 0
            max_steps = 3000
            
            print("🚦 Lights out and away we go!")
            
            # Run the race
            while race_steps < max_steps:
                # Get actions from all AI policies for existing agents only
                actions = {}
                
                # Only generate actions for agents that exist in current observations
                for agent_id in observations.keys():
                    if agent_id in policies:
                        obs = observations[agent_id]
                        action = policies[agent_id].act(obs)
                        actions[agent_id] = action
                    else:
                        # Fallback action for any missing policy
                        actions[agent_id] = [0.0, 0.5]
                
                # Step environment
                step_result = env.step(actions)
                
                if len(step_result) == 4:
                    observations, rewards, done, infos = step_result
                else:
                    observations, rewards, terminated, truncated, infos = step_result
                    done = terminated
                
                # Update rewards only for existing agents
                if isinstance(rewards, dict):
                    for agent_id in observations.keys():
                        if agent_id in rewards and agent_id in total_rewards:
                            total_rewards[agent_id] += rewards[agent_id]
                
                # Check for individual agent crashes/done states
                if isinstance(done, dict):
                    for agent_id, is_done in done.items():
                        if is_done and agent_id != "__all__" and agent_id in policies:
                            # Check if it was a crash
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
                                print(f"   🚨 {policy.name} ({policy.strategy}) - {crash_reason}")
                
                race_steps += 1
                
                # Check if all agents are done (crashed or finished)
                all_agents_done = False
                if isinstance(done, dict):
                    # Check if all individual agents are done
                    active_agents = [aid for aid in observations.keys() if aid != "__all__"]
                    if active_agents:
                        all_agents_done = all(done.get(aid, False) for aid in active_agents)
                    
                    # Also check the global done flag
                    if done.get("__all__", False) or all_agents_done:
                        if all_agents_done:
                            print(f"   🏁 All agents crashed/finished! Race ended at step {race_steps}")
                        break
                elif done:
                    break
                
                # Print progress every 500 steps
                if race_steps % 500 == 0:
                    print(f"   Lap progress: {race_steps}/{max_steps} steps")
                
                # Small delay for smooth visualization
                time.sleep(0.01)
            
            # Race results
            print(f"\n🏆 Race {episode + 1} Results (after {race_steps} steps):")
            sorted_results = sorted(total_rewards.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (agent_id, reward) in enumerate(sorted_results, 1):
                policy = policies[agent_id]
                emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "🏎️"
                print(f"   {emoji} {rank}. {policy.name} ({policy.strategy}): {reward:.2f} points")
            
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