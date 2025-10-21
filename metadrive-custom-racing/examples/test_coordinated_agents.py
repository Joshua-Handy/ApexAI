"""
Test Curriculum-Based Multi-Agent Racing Models

This script loads and tests the trained curriculum-based racing agents,
allowing you to watch them compete in real-time racing scenarios.

Key Features:
- Load models trained with curriculum learning
- Visual racing with all 4 agents
- Per-agent performance tracking
- Racing statistics and analysis
- Phase-appropriate testing environments
"""
import os
import sys
import argparse
import numpy as np
import time
from typing import Dict, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stable_baselines3 import PPO
import gymnasium as gym

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from environments.multi_agent_oval_right_env import MultiAgentOvalEnv
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    print("Warning: MultiAgentOvalEnv not available, using standard MetaDrive")
    from metadrive.envs import MultiAgentMetaDrive
    MULTI_AGENT_AVAILABLE = False


class CoordinatedRacingTester:
    """Test coordinated multi-agent racing models."""
    
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.models = {}
        self.agent_stats = {}
        
    def load_models(self) -> List[str]:
        """Load all trained coordinated models."""
        print("🏎️  Loading coordinated racing models...")
        
        # Find model files - use the correct directory name
        final_models_dir = os.path.join(self.models_dir, "final_models")
        if not os.path.exists(final_models_dir):
            print(f"❌ Models directory not found: {final_models_dir}")
            # Try alternative directories
            alt_dirs = [
                "coordinated_4agent_hyper_conservative_v2/final_models",
                "coordinated_4agent_hyper_conservative/final_models", 
                "coordinated_multi_agent_results/final_models"
            ]
            
            for alt_dir in alt_dirs:
                alt_path = os.path.join(os.path.dirname(self.models_dir), alt_dir)
                if os.path.exists(alt_path):
                    print(f"✅ Found models in: {alt_path}")
                    final_models_dir = alt_path
                    break
            else:
                print("❌ No trained models found!")
                return []
        
        agent_ids = []
        for file in os.listdir(final_models_dir):
            if file.endswith("_final.zip"):
                agent_id = file.replace("_final.zip", "")
                model_path = os.path.join(final_models_dir, file)
                
                try:
                    # Load the PPO model
                    model = PPO.load(model_path)
                    self.models[agent_id] = model
                    agent_ids.append(agent_id)
                    
                    # Initialize stats including respawns
                    self.agent_stats[agent_id] = {
                        'total_reward': 0.0,
                        'episode_count': 0,
                        'total_steps': 0,
                        'crashes': 0,
                        'out_of_road': 0,
                        'completed_laps': 0,
                        'respawns': 0
                    }
                    
                    print(f"   ✅ Loaded {agent_id} from {model_path}")
                    
                except Exception as e:
                    print(f"   ❌ Failed to load {agent_id}: {e}")
        
        print(f"\n🏁 Successfully loaded {len(agent_ids)} coordinated racing agents!")
        return agent_ids
    
    def create_racing_environment(self, agent_ids: List[str], render: bool = True):
        """Create racing environment for testing."""
        
        def get_nascar_grid_position(agent_index: int, total_agents: int):
            """NASCAR-style grid positioning with SAFE SPACING for testing."""
            delay_pattern = [180, 0, 120, 60]
            agent_delay = delay_pattern[agent_index % len(delay_pattern)]
            
            starting_order_map = {0: 0, 60: 1, 120: 2, 180: 3}
            position = starting_order_map[agent_delay]
            
            # CRITICAL FIX: Add longitudinal spacing to prevent immediate collisions!
            # During training, agents learned with simple opponent policies
            # In testing, all agents use trained policies simultaneously - need more space
            longitudinal_spacing = position * -50.0  # Stagger agents along track (50 units apart)
            lateral_offset = (position - (total_agents - 1) / 2) * 25.0  # Reduce lateral spacing too
                
            return longitudinal_spacing, lateral_offset
        
        # Generate NASCAR-style grid positions with SAFE SPACING
        nascar_spawn_configs = {}
        for i, agent_id in enumerate(agent_ids):
            longitude, lateral = get_nascar_grid_position(i, len(agent_ids))
            nascar_spawn_configs[agent_id] = {
                "spawn_longitude": longitude,
                "spawn_lateral": lateral,
            }
            print(f"   🏁 {agent_id}: longitude={longitude:.1f}, lateral={lateral:.1f}")
        
        print(f"🏎️  Agent spacing: Longitudinal stagger to prevent immediate collisions!")
        
        # Environment config - SAME as training
        config = {
            "num_agents": len(agent_ids),
            "traffic_density": 0.0,
            "use_render": render,
            "crash_done": True,
            "out_of_road_done": True,
            "horizon": 8000,  # SAME as training
            "success_reward": 30.0,  # SAME as training
            "driving_reward": 3.0,   # SAME as training
            "speed_reward": 2.0,     # SAME as training
            "out_of_road_penalty": 5.0,     # SAME as training
            "crash_vehicle_penalty": 15.0,  # SAME as training
            "map_config": {
                "lane_num": 1,
                "lane_width": 30.0,
            },
            "vehicle_config": {
                "show_lidar": False,
                "show_lane_line_detector": False,
                "show_side_detector": False,
                "enable_reverse": False,
            },
            "agent_configs": nascar_spawn_configs
        }
        
        # Create environment
        if MULTI_AGENT_AVAILABLE:
            try:
                env = MultiAgentOvalEnv(config)
                print("🏁 Using custom right-turn oval track for testing!")
                return env
            except Exception as e:
                print(f"⚠️  Custom environment failed: {e}")
        
        # Fallback to standard multi-agent
        from metadrive.envs import MultiAgentMetaDrive
        config["map"] = "O"  # Standard oval
        env = MultiAgentMetaDrive(config)
        print("🏁 Using standard oval track for testing!")
        return env
    
    def run_race(self, agent_ids: List[str], episode_num: int, render: bool = True):
        """Run a single race with all agents."""
        print(f"\n🏁 Starting Race {episode_num + 1}")
        print("-" * 40)
        
        # Create environment
        env = self.create_racing_environment(agent_ids, render)
        
        try:
            # Reset environment
            obs_dict = env.reset()
            if isinstance(obs_dict, tuple):
                obs_dict = obs_dict[0]
            
            episode_rewards = {agent_id: 0.0 for agent_id in agent_ids}
            episode_steps = {agent_id: 0 for agent_id in agent_ids}
            done = False
            step_count = 0
            
            print(f"🚀 Race in progress...")
            start_time = time.time()
            
            while not done:
                # Get actions from all agents
                actions = {}
                for agent_id in agent_ids:
                    if agent_id in obs_dict and agent_id in self.models:
                        # Extract individual agent observation from dict (CRITICAL FIX!)
                        obs = obs_dict[agent_id]
                        
                        # Debug observation shape
                        if step_count == 0:  # Only print on first step
                            print(f"   🔍 {agent_id} obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
                        
                        try:
                            action, _ = self.models[agent_id].predict(obs, deterministic=True)
                            actions[agent_id] = action
                        except Exception as e:
                            print(f"   ⚠️  {agent_id} prediction failed: {e}")
                            actions[agent_id] = [0.0, 0.0]  # Safe default action
                    else:
                        actions[agent_id] = [0.0, 0.0]  # Stop if no observation/model
                
                # Step environment
                try:
                    step_result = env.step(actions)
                    
                    if len(step_result) == 4:
                        obs_dict, reward_dict, done_dict, info_dict = step_result
                    else:
                        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = step_result
                        done_dict = {k: terminated_dict.get(k, False) or truncated_dict.get(k, False) 
                                    for k in terminated_dict.keys()}
                    
                    # Check for respawns and display info
                    for agent_id in agent_ids:
                        if agent_id in info_dict:
                            info = info_dict.get(agent_id, {})
                            if info.get('respawned', False):
                                reason = info.get('respawn_reason', 'unknown')
                                print(f"   🔄 Step {step_count}: {agent_id} respawned ({reason})")
                    
                    # Check if all agents are still done (shouldn't happen with respawning)
                    if step_count < 100:  # Only debug first 100 steps
                        active_agents = sum(1 for k, v in done_dict.items() if k != "__all__" and not v)
                        if active_agents < len(agent_ids):
                            print(f"   ⚠️  Step {step_count}: {len(agent_ids) - active_agents} agents still done after respawn check!")
                            for agent_id in agent_ids:
                                if done_dict.get(agent_id, False):
                                    info = info_dict.get(agent_id, {})
                                    reason = "unknown"
                                    if info.get('crash', False):
                                        reason = "crash"
                                    elif info.get('out_of_road', False):
                                        reason = "out_of_road"
                                    elif info.get('max_step', False):
                                        reason = "max_steps"
                                    print(f"      {agent_id}: {reason} (reward: {reward_dict.get(agent_id, 0):.1f})")
                
                except Exception as e:
                    print(f"   ❌ Step {step_count} failed: {e}")
                    break
                
                # Update episode stats
                for agent_id in agent_ids:
                    if agent_id in reward_dict:
                        episode_rewards[agent_id] += reward_dict[agent_id]
                        episode_steps[agent_id] += 1
                        
                        # Check for special events including respawns
                        if agent_id in info_dict:
                            info = info_dict[agent_id]
                            if info.get('crash', False):
                                self.agent_stats[agent_id]['crashes'] += 1
                            if info.get('out_of_road', False):
                                self.agent_stats[agent_id]['out_of_road'] += 1
                            if info.get('respawned', False):
                                # Track respawns as a special stat
                                if 'respawns' not in self.agent_stats[agent_id]:
                                    self.agent_stats[agent_id]['respawns'] = 0
                                self.agent_stats[agent_id]['respawns'] += 1
                
                step_count += 1
                
                # Check if race is done
                done = done_dict.get("__all__", False) or any(done_dict.values())
                
                # Print progress every 1000 steps
                if step_count % 1000 == 0:
                    elapsed = time.time() - start_time
                    print(f"   Step {step_count:,} | Elapsed: {elapsed:.1f}s")
                    for agent_id in agent_ids:
                        reward = episode_rewards[agent_id]
                        print(f"      {agent_id}: {reward:.1f} reward")
            
            race_time = time.time() - start_time
            
            # Update global stats
            for agent_id in agent_ids:
                stats = self.agent_stats[agent_id]
                stats['total_reward'] += episode_rewards[agent_id]
                stats['episode_count'] += 1
                stats['total_steps'] += episode_steps[agent_id]
            
            # Print race results
            print(f"\n🏆 Race {episode_num + 1} Results (Duration: {race_time:.1f}s, Steps: {step_count:,})")
            sorted_results = sorted(episode_rewards.items(), key=lambda x: x[1], reverse=True)
            
            for i, (agent_id, reward) in enumerate(sorted_results):
                steps = episode_steps[agent_id]
                print(f"   {i+1}. {agent_id:8} | Reward: {reward:7.1f} | Steps: {steps:,}")
            
            env.close()
            return episode_rewards
            
        except Exception as e:
            print(f"❌ Race failed: {e}")
            env.close()
            return {agent_id: 0.0 for agent_id in agent_ids}
    
    def run_championship(self, agent_ids: List[str], num_races: int = 5, render: bool = True):
        """Run a full racing championship."""
        print(f"\n🏁 COORDINATED MULTI-AGENT RACING CHAMPIONSHIP")
        print("=" * 60)
        print(f"Participants: {', '.join(agent_ids)}")
        print(f"Races: {num_races}")
        print(f"Track: Right-turn oval")
        print(f"Render: {'Yes' if render else 'No'}")
        print()
        
        championship_results = []
        
        try:
            for race_num in range(num_races):
                race_results = self.run_race(agent_ids, race_num, render)
                championship_results.append(race_results)
                
                # Brief pause between races
                if race_num < num_races - 1:
                    print("⏳ Preparing for next race...")
                    time.sleep(2)
            
            # Calculate championship standings
            self.print_championship_results(agent_ids, championship_results)
            
        except KeyboardInterrupt:
            print("\n🛑 Championship interrupted by user")
            self.print_championship_results(agent_ids, championship_results)
    
    def print_championship_results(self, agent_ids: List[str], championship_results: List[Dict]):
        """Print final championship results."""
        print(f"\n🏆 CHAMPIONSHIP RESULTS")
        print("=" * 50)
        
        # Calculate averages and totals
        final_standings = []
        for agent_id in agent_ids:
            stats = self.agent_stats[agent_id]
            if stats['episode_count'] > 0:
                avg_reward = stats['total_reward'] / stats['episode_count']
                avg_steps = stats['total_steps'] / stats['episode_count']
                
                final_standings.append({
                    'agent_id': agent_id,
                    'avg_reward': avg_reward,
                    'total_reward': stats['total_reward'],
                    'avg_steps': avg_steps,
                    'races': stats['episode_count'],
                    'crashes': stats['crashes'],
                    'out_of_road': stats['out_of_road']
                })
        
        # Sort by average reward
        final_standings.sort(key=lambda x: x['avg_reward'], reverse=True)
        
        print(f"\nFinal Standings:")
        print("-" * 50)
        for i, stats in enumerate(final_standings):
            agent_id = stats['agent_id']
            avg_reward = stats['avg_reward']
            total_reward = stats['total_reward']
            avg_steps = stats['avg_steps']
            crashes = stats['crashes']
            out_of_road = stats['out_of_road']
            respawns = stats.get('respawns', 0)
            
            print(f"{i+1}. {agent_id:8} | Avg: {avg_reward:7.1f} | Total: {total_reward:7.1f}")
            print(f"   {'':10} | Steps: {avg_steps:6.0f} | Crashes: {crashes} | Out: {out_of_road} | Respawns: {respawns}")
        
        # Race-by-race breakdown
        if championship_results:
            print(f"\nRace-by-Race Results:")
            print("-" * 50)
            for race_num, race_results in enumerate(championship_results):
                print(f"Race {race_num + 1}:")
                sorted_race = sorted(race_results.items(), key=lambda x: x[1], reverse=True)
                for pos, (agent_id, reward) in enumerate(sorted_race):
                    print(f"   {pos+1}. {agent_id}: {reward:.1f}")
        
        # Performance analysis
        if len(final_standings) > 1:
            winner = final_standings[0]
            runner_up = final_standings[1]
            margin = winner['avg_reward'] - runner_up['avg_reward']
            
            print(f"\n🥇 CHAMPION: {winner['agent_id']}")
            print(f"   Average Reward: {winner['avg_reward']:.1f}")
            print(f"   Margin of Victory: {margin:.1f} points")
            print(f"   Consistency: {winner['crashes']} crashes, {winner['out_of_road']} out-of-road, {winner.get('respawns', 0)} respawns")


def main():
    parser = argparse.ArgumentParser(description='Test Coordinated Multi-Agent Racing Models')
    parser.add_argument('--models-dir', type=str, default='coordinated_4agent_hyper_conservative_v2',
                       help='Directory containing trained models')
    parser.add_argument('--races', type=int, default=5,
                       help='Number of races in championship')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering for faster testing')
    parser.add_argument('--agents', type=str, nargs='+',
                       help='Specific agents to test (default: all)')
    
    args = parser.parse_args()
    
    # Create tester
    tester = CoordinatedRacingTester(args.models_dir)
    
    # Load models
    available_agents = tester.load_models()
    if not available_agents:
        print("❌ No trained models found!")
        return
    
    # Select agents to test
    if args.agents:
        test_agents = [agent for agent in args.agents if agent in available_agents]
        if not test_agents:
            print(f"❌ None of the specified agents found in {available_agents}")
            return
    else:
        test_agents = available_agents
    
    print(f"\n🏎️  Testing agents: {test_agents}")
    
    # Run championship
    render = not args.no_render
    tester.run_championship(test_agents, args.races, render)


if __name__ == '__main__':
    main()
