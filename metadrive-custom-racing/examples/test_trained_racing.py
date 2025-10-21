"""
Test the trained racing agents to see their actual performance.
"""
import os
import sys
import numpy as np
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stable_baselines3 import PPO

try:
    from environments.multi_agent_oval_right_env import MultiAgentOvalEnv
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    print("Warning: MultiAgentOvalEnv not available, using standard MetaDrive")
    from metadrive.envs import MultiAgentMetaDrive
    MULTI_AGENT_AVAILABLE = False


def create_test_environment(num_agents: int = 4):
    """Create test environment matching training setup with proper spacing."""
    
    # RACING GRID positions with proper lane assignments (MATCHES TRAINING)
    racing_grid_configs = {}
    for i in range(num_agents):
        # 2x2 grid formation like training
        row = i // 2
        lane_index = i % 2
        longitude = -row * 30.0  # 30 units between rows
        racing_grid_configs[f"agent{i}"] = {
            "spawn_longitude": longitude,
            "spawn_lane_index": lane_index,  # Lane 0 or 1
        }
    
    # Test environment config - FIXED to prevent respawn loops
    config = {
        "num_agents": num_agents,
        "traffic_density": 0.0,
        "use_render": True,  # Enable rendering for visual testing
        "crash_done": True,  # FIXED: Terminate on crash (no respawn)
        "out_of_road_done": True,  # FIXED: Terminate when out of bounds (no respawn)
        "allow_respawn": False,  # CRITICAL: Prevents infinite respawn loops
        "horizon": 5000,  # Longer episodes for testing
        "success_reward": 15.0,
        "driving_reward": 1.5,
        "speed_reward": 0.8,
        "use_lateral_reward": True,
        "out_of_road_penalty": 1.0,
        "crash_vehicle_penalty": 2.0,
        "crash_object_penalty": 1.6,
        "map_config": {
            "lane_num": 2,  # CRITICAL: Must match training (2 lanes for racing grid)
            "lane_width": 30.0,
        },
        "vehicle_config": {
            "show_lidar": True,
            "show_lane_line_detector": True,
            "show_side_detector": True,
            "enable_reverse": False,
            # CRITICAL: Match training configuration exactly to avoid observation shape mismatch
            # Remove the lidar config that's causing the shape difference
        },
        "agent_configs": racing_grid_configs  # FIXED: Use racing grid configs
    }
    
    # Try custom environment first
    if MULTI_AGENT_AVAILABLE:
        try:
            env = MultiAgentOvalEnv(config)
            print("🏁 Using custom right-turn oval track for testing!")
            return env
        except Exception as e:
            print(f"⚠️  Custom environment failed: {e}")
    
    # Fallback to standard
    from metadrive.envs import MultiAgentMetaDrive
    config["map"] = "O"
    env = MultiAgentMetaDrive(config)
    print("🏁 Using standard oval track for testing!")
    return env


def test_trained_agents(models_dir: str = "racing_training_v1/final_models", episodes: int = 3):
    """Test the trained racing agents."""
    
    print("🏁 TESTING TRAINED RACING AGENTS")
    print("=" * 50)
    print(f"Models directory: {models_dir}")
    print(f"Test episodes: {episodes}")
    print()
    
    # Create test environment
    env = create_test_environment(4)
    
    # Load trained models
    models = {}
    agent_ids = ["agent0", "agent1", "agent2", "agent3"]
    
    for agent_id in agent_ids:
        model_path = os.path.join(models_dir, f"{agent_id}_final.zip")
        if os.path.exists(model_path):
            try:
                models[agent_id] = PPO.load(model_path)
                print(f"✅ Loaded {agent_id} from {model_path}")
            except Exception as e:
                print(f"❌ Failed to load {agent_id}: {e}")
        else:
            print(f"⚠️  Model not found: {model_path}")
    
    if not models:
        print("❌ No models loaded! Training may not have completed.")
        return
    
    print(f"\n🏎️  Testing {len(models)} trained agents...")
    print()
    
    # Test episodes
    for episode in range(episodes):
        print(f"\n🏁 Test Episode {episode + 1}/{episodes}")
        print("-" * 30)
        
        obs_dict = env.reset()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]
        
        episode_rewards = {agent_id: 0.0 for agent_id in agent_ids}
        episode_lengths = {agent_id: 0 for agent_id in agent_ids}
        episode_crashes = {agent_id: 0 for agent_id in agent_ids}
        episode_out_bounds = {agent_id: 0 for agent_id in agent_ids}
        
        done = False
        step = 0
        
        while not done and step < 5000:
            actions = {}
            
            # Get actions from trained models
            for agent_id in agent_ids:
                if agent_id in obs_dict and agent_id in models:
                    try:
                        action, _ = models[agent_id].predict(obs_dict[agent_id], deterministic=True)
                        actions[agent_id] = action
                    except Exception as e:
                        print(f"⚠️  {agent_id} prediction failed: {e}")
                        actions[agent_id] = [0.5, 0.0]  # Safe default
                else:
                    actions[agent_id] = [0.5, 0.0]  # Safe default
            
            # Step environment
            step_result = env.step(actions)
            
            if len(step_result) == 4:
                obs_dict, reward_dict, done_dict, info_dict = step_result
            else:
                obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = step_result
                done_dict = {k: terminated_dict.get(k, False) or truncated_dict.get(k, False) 
                            for k in terminated_dict.keys()}
            
            # Update metrics
            for agent_id in agent_ids:
                if agent_id in reward_dict:
                    episode_rewards[agent_id] += reward_dict[agent_id]
                    episode_lengths[agent_id] += 1
                    
                    # Check for crashes and out-of-bounds
                    if agent_id in info_dict:
                        info = info_dict[agent_id]
                        if info.get('crashed', False):
                            episode_crashes[agent_id] += 1
                        if info.get('out_of_road', False):
                            episode_out_bounds[agent_id] += 1
            
            # Check if all agents are done
            done = all(done_dict.get(agent_id, False) for agent_id in agent_ids)
            step += 1
            
            # Print progress every 1000 steps
            if step % 1000 == 0:
                print(f"   Step {step}: Still racing...")
        
        # Episode summary
        print(f"\n📊 Episode {episode + 1} Results:")
        print("Agent      | Reward  | Length | Crashes | Out-of-Bounds")
        print("-" * 55)
        
        total_reward = 0
        for agent_id in agent_ids:
            reward = episode_rewards[agent_id]
            length = episode_lengths[agent_id]
            crashes = episode_crashes[agent_id]
            out_bounds = episode_out_bounds[agent_id]
            total_reward += reward
            
            print(f"{agent_id:10} | {reward:7.1f} | {length:6} | {crashes:7} | {out_bounds:13}")
        
        print(f"Total Reward: {total_reward:.1f}")
        print(f"Episode Length: {step} steps")
        
        time.sleep(1)  # Brief pause between episodes
    
    env.close()
    print("\n✅ Testing complete!")
    print("\n🎯 If you see:")
    print("   - High rewards (>150): Excellent racing performance")
    print("   - Long episodes (>3000): Agents completing full races")
    print("   - Low crashes (<5): Good track control")
    print("   - Smooth racing: Agents learned proper racing behavior")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained racing agents')
    parser.add_argument('--models-dir', type=str, default='racing_training_v1/final_models',
                       help='Directory containing trained models')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of test episodes')
    
    args = parser.parse_args()
    
    test_trained_agents(args.models_dir, args.episodes)
