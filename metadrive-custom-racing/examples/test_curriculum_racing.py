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

# Enable Panda3D software renderer early if requested
try:
    if '--software-render' in sys.argv or os.environ.get('METADRIVE_SOFTWARE_RENDER') == '1':
        from panda3d.core import loadPrcFileData
        # Use correct plugin name for pip Panda3D
        loadPrcFileData('', 'load-display p3tinydisplay')
        loadPrcFileData('', 'aux-display p3tinydisplay')
        loadPrcFileData('', 'win-size 1024 768')
except Exception:
    pass

from stable_baselines3 import PPO
from metadrive.component.pgblock.first_block import FirstPGBlock

try:
    from environments.multi_agent_oval_right_env import MultiAgentOvalEnv
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    print("Warning: MultiAgentOvalEnv not available, using standard MetaDrive")
    from metadrive.envs import MultiAgentMetaDrive
    MULTI_AGENT_AVAILABLE = False


class CurriculumRacingTester:
    """Test curriculum-trained multi-agent racing models."""
    
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.models = {}
        self.agent_stats = {}
        
    def load_models(self) -> List[str]:
        """Load all trained curriculum models."""
        print("Loading curriculum-trained racing models...")
        
        # Find model files
        final_models_dir = os.path.join(self.models_dir, "final_models")
        if not os.path.exists(final_models_dir):
            print(f"[ERROR] Models directory not found: {final_models_dir}")
            
            # Try alternative directories
            alt_dirs = [
                "multi_agent_racing_results/final_models",
                "curriculum_racing_results/final_models",
                self.models_dir  # Direct path
            ]
            
            for alt_dir in alt_dirs:
                if os.path.exists(alt_dir):
                    final_models_dir = alt_dir
                    print(f"[OK] Found models in: {final_models_dir}")
                    break
            else:
                print(f"[ERROR] No model directory found!")
                return []
        
        agent_ids = []
        for file in os.listdir(final_models_dir):
            if file.endswith("_final.zip"):
                agent_id = file.replace("_final.zip", "")
                try:
                    model_path = os.path.join(final_models_dir, file)
                    self.models[agent_id] = PPO.load(model_path)
                    agent_ids.append(agent_id)
                    print(f"[OK] Loaded {agent_id}")
                except Exception as e:
                    print(f"[ERROR] Failed to load {agent_id}: {e}")
        
        # Initialize stats tracking
        for agent_id in agent_ids:
            self.agent_stats[agent_id] = {
                'total_reward': 0.0,
                'total_races': 0,
                'crashes': 0,
                'laps_completed': 0,
                'race_completions': 0,
                'avg_speed': 0.0,
                'best_lap_time': float('inf')
            }
        
        print(f"\nSuccessfully loaded {len(agent_ids)} curriculum-trained agents!")
        return agent_ids
    
    def create_racing_environment(self, agent_ids: List[str], render: bool = True, phase: str = 'racing', software_render: bool = False):
        """Create racing environment for testing with phase-appropriate settings."""
        if software_render:
            try:
                from panda3d.core import loadPrcFileData
                loadPrcFileData('', 'load-display tinydisplay')
            except Exception:
                pass
        
        # Phase configurations (matching curriculum training)
        phase_configs = {
            'safety': {
                'horizon': 2000,
                'speed_limit': 0.3,
                'safety_weight': 3.0,
                'description': 'Safe driving test (low speed)'
            },
            'control': {
                'horizon': 2500,
                'speed_limit': 0.6,
                'safety_weight': 2.0,
                'description': 'Control precision test (medium speed)'
            },
            'speed': {
                'horizon': 3000,
                'speed_limit': 1.0,
                'safety_weight': 1.0,
                'description': 'Speed efficiency test (high speed)'
            },
            'racing': {
                'horizon': 4000,
                'speed_limit': 1.5,
                'safety_weight': 0.5,
                'description': 'Full racing competition'
            }
        }
        
        config = phase_configs.get(phase, phase_configs['racing'])
        print(f"Testing in {phase.upper()} phase: {config['description']}")
        
        # Safer NASCAR-style staggered spacing to reduce collisions
        def get_nascar_grid_position(agent_index: int, total_agents: int):
            delay_pattern = [180, 0, 120, 60]
            pos = delay_pattern[agent_index % len(delay_pattern)]
            starting_order_map = {0: 0, 60: 1, 120: 2, 180: 3}
            order_index = starting_order_map[pos]
            longitudinal = -50.0 * order_index
            lateral = (order_index - (total_agents - 1) / 2.0) * 20.0
            return longitudinal, lateral
        
        # Generate racing grid configurations
        racing_grid_configs = {}
        lane_width = 20.0
        for i, agent_id in enumerate(agent_ids):
            longitude, lateral = get_nascar_grid_position(i, len(agent_ids))
            racing_grid_configs[agent_id] = {
                "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, i % len(agent_ids)),
                "spawn_longitude": longitude,
                "spawn_lateral": lateral,
                "spawn_velocity": [3.0, 0.0],
            }
        
        # Environment configuration
        env_config = {
            "num_agents": len(agent_ids),
            "traffic_density": 0.0,
            "use_render": render,
            
            # Per-agent done logic
            "crash_done": False,
            "out_of_road_done": True,
            "on_continuous_line_done": True,
            "on_broken_line_done": False,
            "allow_respawn": False,
            "crash_vehicle_done": False,
            "crash_object_done": False,
            "horizon": config['horizon'],
            
            # Phase-appropriate rewards
            "success_reward": 10.0,
            "driving_reward": 2.0,
            "speed_reward": config['speed_limit'],
            "use_lateral_reward": True,
            "out_of_road_penalty": config['safety_weight'] * 2.0,
            "crash_vehicle_penalty": config['safety_weight'] * 5.0,
            "crash_object_penalty": config['safety_weight'] * 4.0,
            
            # Track configuration
            "map_config": {
                "lane_num": len(agent_ids),
                "lane_width": lane_width,
            },
            
            # Vehicle configuration with lidar
            "vehicle_config": {
                "show_lidar": True,
                "show_lane_line_detector": True,
                "show_side_detector": True,
                "enable_reverse": False,
                "lidar": {
                    "num_others": 4,
                    "distance": 50,
                    "num_lasers": 72,
                },
            },
            
            "agent_configs": racing_grid_configs
        }
        
        # Create environment
        if MULTI_AGENT_AVAILABLE:
            try:
                env = MultiAgentOvalEnv(env_config)
                print("Using custom right-turn oval track!")
                return env
            except Exception as e:
                print(f"⚠️  Custom environment failed: {e}")
        
        # Fallback to standard MetaDrive
        from metadrive.envs import MultiAgentMetaDrive
        env_config["map"] = "O"
        env = MultiAgentMetaDrive(env_config)
        print("Using standard oval track!")
        return env
    
    def run_race(self, agent_ids: List[str], episode_num: int, render: bool = True, phase: str = 'racing', software_render: bool = False):
        """Run a single race with curriculum-trained agents."""
        print(f"\nStarting Race {episode_num + 1} ({phase.upper()} phase)")
        print("-" * 50)
        
        # Create environment
        env = self.create_racing_environment(agent_ids, render, phase, software_render)
        
        try:
            # Reset environment
            obs_dict = env.reset()
            if isinstance(obs_dict, tuple):
                obs_dict = obs_dict[0]
            
            # Initialize race tracking
            race_rewards = {agent_id: 0.0 for agent_id in agent_ids}
            race_lengths = {agent_id: 0 for agent_id in agent_ids}
            race_crashes = {agent_id: 0 for agent_id in agent_ids}
            race_positions = {agent_id: np.array([0.0, 0.0]) for agent_id in agent_ids}
            race_laps = {agent_id: 0 for agent_id in agent_ids}
            agent_done = {agent_id: False for agent_id in agent_ids}
            
            print(f"Racing agents: {', '.join(agent_ids)}")
            print(f"Tracking: rewards, crashes, laps, positions")
            
            step = 0
            max_steps = 5000
            
            while step < max_steps:
                actions = {}
                # Build actions for currently active env agents to prevent KeyError
                current_env_agents = list(obs_dict.keys())
                for env_agent_id in current_env_agents:
                    if env_agent_id in self.models and env_agent_id in obs_dict and not agent_done.get(env_agent_id, False):
                        try:
                            action, _ = self.models[env_agent_id].predict(obs_dict[env_agent_id], deterministic=True)
                            original_steering = action[0]
                            original_throttle = action[1]
                            transformed_throttle = 0.25 + ((action[1] + 1.0) / 2.0) * 0.6
                            transformed_steering = np.clip(action[0], -0.2, 0.2)
                            # Initial movement boost to avoid stalling
                            min_throttle = 0.45 if step < 120 else 0.0
                            action = [transformed_steering, float(np.clip(max(transformed_throttle, min_throttle), 0.0, 0.8))]
                            if step < 5:
                                print(f"   Adjust {env_agent_id}: original=({original_throttle:.3f},{original_steering:.3f}) -> racing=({transformed_throttle:.3f},{transformed_steering:.3f})")
                            actions[env_agent_id] = action
                        except Exception as e:
                            print(f"[WARN] {env_agent_id} prediction failed: {e}")
                            actions[env_agent_id] = [0.0, 0.5]
                    else:
                        # Unknown or done agents: keep them neutral to avoid crashes
                        actions[env_agent_id] = [0.0, 0.4]
                
                # Step environment
                try:
                    step_result = env.step(actions)
                except Exception as e:
                    try:
                        print(f"   Exception during env.step at step {step}: {e}")
                        if hasattr(env, 'agents'):
                            print(f"   Env agents: {list(env.agents.keys())}")
                        print(f"   Actions keys: {list(actions.keys())}")
                    except Exception:
                        pass
                    raise
                
                if len(step_result) == 4:
                    obs_dict, reward_dict, done_dict, info_dict = step_result
                else:
                    obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = step_result
                    done_dict = {k: terminated_dict.get(k, False) or truncated_dict.get(k, False) 
                                for k in terminated_dict.keys()}
                
                # Update race metrics
                for agent_id in agent_ids:
                    try:
                        if not agent_done[agent_id]:
                            if agent_id in reward_dict:
                                race_rewards[agent_id] += reward_dict[agent_id]
                                race_lengths[agent_id] += 1
                            if agent_id in info_dict:
                                info = info_dict[agent_id]
                                if done_dict.get(agent_id, False):
                                    if not agent_done[agent_id]:
                                        if info.get('crashed', False):
                                            race_crashes[agent_id] += 1
                                        agent_done[agent_id] = True
                                        reason = 'crash' if info.get('crashed', False) else 'done'
                                        print(f"   Agent {agent_id} {reason} at step {step}")
                                if 'vehicle_state' in info:
                                    vehicle_state = info['vehicle_state']
                                    position = np.array(vehicle_state.get('position', [0, 0]))
                                    prev_x = race_positions[agent_id][0]
                                    curr_x = position[0]
                                    if curr_x > 100 and prev_x < -100:
                                        race_laps[agent_id] += 1
                                        print(f"   Lap {agent_id} completed {race_laps[agent_id]}")
                                    race_positions[agent_id] = position
                    except KeyError as ke:
                        print(f"   Key error updating metrics for {agent_id}: {ke}")
                        print(f"   reward_dict keys: {list(reward_dict.keys())}")
                        print(f"   done_dict keys: {list(done_dict.keys())}")
                        print(f"   info_dict keys: {list(info_dict.keys())}")
                        raise
                
                step += 1
                
                # Progress updates with action debugging
                if step % 500 == 0:
                    active_agents = sum(1 for done in agent_done.values() if not done)
                    print(f"   Step {step}: {active_agents} agents still racing...")
                # Stop if env has no active agents
                try:
                    if hasattr(env, 'agents') and len(env.agents) == 0:
                        print("   No active agents left; ending race.")
                        break
                except Exception:
                    pass
                    
                    # Debug: Show current vehicle positions
                    for agent_id in agent_ids[:2]:  # Show first 2 agents
                        if not agent_done.get(agent_id, False):
                            if agent_id in env.agents:
                                vehicle = env.agents[agent_id]
                                pos = vehicle.position
                                speed = vehicle.speed_km_h
                                print(f"      {agent_id}: pos=({pos[0]:.1f}, {pos[1]:.1f}), speed={speed:.1f}km/h")
                            if agent_id in actions:
                                action = actions[agent_id]
                                steering = action[0] if len(action) > 0 else 0.0
                                throttle = action[1] if len(action) > 1 else 0.0
                                print(f"      {agent_id}: steering={steering:.3f}, throttle={throttle:.3f}")
            
            # Race summary
            print(f"\nRace {episode_num + 1} Results:")
            print("Agent      | Reward  | Length | Crashes | Laps | Status")
            print("-" * 60)
            
            race_results = {}
            for agent_id in agent_ids:
                reward = race_rewards[agent_id]
                length = race_lengths[agent_id]
                crashes = race_crashes[agent_id]
                laps = race_laps[agent_id]
                status = "Done" if agent_done[agent_id] else "Active"
                
                print(f"{agent_id:10} | {reward:7.1f} | {length:6} | {crashes:7} | {laps:4} | {status}")
                
                # Update global stats
                self.agent_stats[agent_id]['total_reward'] += reward
                self.agent_stats[agent_id]['total_races'] += 1
                self.agent_stats[agent_id]['crashes'] += crashes
                self.agent_stats[agent_id]['laps_completed'] += laps
                
                if length > 1000:  # Completed significant portion
                    self.agent_stats[agent_id]['race_completions'] += 1
                
                race_results[agent_id] = {
                    'reward': reward,
                    'length': length,
                    'crashes': crashes,
                    'laps': laps,
                    'completed': length > 1000
                }
            
            print(f"Race Duration: {step} steps")
            return race_results
            
        except Exception as e:
            print(f"[ERROR] Race failed: {e}")
            return {agent_id: {'reward': 0, 'length': 0, 'crashes': 1, 'laps': 0, 'completed': False} 
                    for agent_id in agent_ids}
        finally:
            env.close()
    
    def run_curriculum_testing(self, agent_ids: List[str], races_per_phase: int = 3, render: bool = True, software_render: bool = False):
        """Run comprehensive testing across all curriculum phases."""
        phases = ['safety', 'control', 'speed', 'racing']
        
        print(f"\nCURRICULUM TESTING SUITE")
        print("=" * 60)
        print(f"Agents: {', '.join(agent_ids)}")
        print(f"Phases: {len(phases)} (Safety -> Control -> Speed -> Racing)")
        print(f"Races per phase: {races_per_phase}")
        print(f"Total races: {len(phases) * races_per_phase}")
        print()
        
        all_results = {}
        
        for phase in phases:
            print(f"\nTESTING PHASE: {phase.upper()}")
            print("-" * 40)
            
            phase_results = []
            
            for race_num in range(races_per_phase):
                race_result = self.run_race(agent_ids, race_num, render, phase, software_render)
                phase_results.append(race_result)
                
                # Brief pause between races
                if render:
                    time.sleep(1)
            
            all_results[phase] = phase_results
            
            # Phase summary
            print(f"\n{phase.upper()} Phase Summary:")
            for agent_id in agent_ids:
                phase_rewards = [r[agent_id]['reward'] for r in phase_results]
                phase_completions = sum(1 for r in phase_results if r[agent_id]['completed'])
                avg_reward = np.mean(phase_rewards)
                
                print(f"   {agent_id}: Avg Reward {avg_reward:.1f}, Completions {phase_completions}/{races_per_phase}")
        
        return all_results
    
    def print_final_statistics(self, agent_ids: List[str]):
        """Print comprehensive performance statistics."""
        print(f"\nFINAL CURRICULUM TESTING STATISTICS")
        print("=" * 60)
        
        # Agent rankings by performance
        agent_rankings = []
        for agent_id in agent_ids:
            stats = self.agent_stats[agent_id]
            avg_reward = stats['total_reward'] / max(1, stats['total_races'])
            completion_rate = stats['race_completions'] / max(1, stats['total_races'])
            crash_rate = stats['crashes'] / max(1, stats['total_races'])
            
            agent_rankings.append({
                'agent_id': agent_id,
                'avg_reward': avg_reward,
                'completion_rate': completion_rate,
                'crash_rate': crash_rate,
                'total_laps': stats['laps_completed']
            })
        
        # Sort by average reward
        agent_rankings.sort(key=lambda x: x['avg_reward'], reverse=True)
        
        print(f"\\nAgent Rankings (by avg reward):")
        print("Rank | Agent      | Avg Reward | Completion | Crash Rate | Total Laps")
        print("-" * 70)
        
        for i, stats in enumerate(agent_rankings):
            print(f"{i+1:4} | {stats['agent_id']:10} | {stats['avg_reward']:10.1f} | "
                  f"{stats['completion_rate']:9.1%} | {stats['crash_rate']:9.1%} | {stats['total_laps']:10}")
        
        # Performance analysis
        best_agent = agent_rankings[0]
        worst_agent = agent_rankings[-1]
        
        print(f"\nPerformance Analysis:")
        print(f"   Best Agent: {best_agent['agent_id']} (Avg Reward: {best_agent['avg_reward']:.1f})")
        print(f"   Needs Improvement: {worst_agent['agent_id']} (Avg Reward: {worst_agent['avg_reward']:.1f})")
        
        avg_completion = np.mean([r['completion_rate'] for r in agent_rankings])
        avg_crash_rate = np.mean([r['crash_rate'] for r in agent_rankings])
        
        print(f"   Overall Completion Rate: {avg_completion:.1%}")
        print(f"   Overall Crash Rate: {avg_crash_rate:.1%}")
        
        print(f"\nCurriculum training assessment:")
        if avg_completion > 0.7:
            print("   EXCELLENT: Agents learned to complete races consistently!")
        elif avg_completion > 0.5:
            print("   GOOD: Agents show solid racing performance")
        elif avg_completion > 0.3:
            print("   FAIR: Agents learned basic skills but need improvement")
        else:
            print("   POOR: Agents need more training")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Test Curriculum-Trained Multi-Agent Racing Models')
    parser.add_argument('--models-dir', type=str, default='multi_agent_racing_results',
                       help='Directory containing trained models')
    parser.add_argument('--races-per-phase', type=int, default=3,
                       help='Number of races per curriculum phase')
    parser.add_argument('--phase', type=str, default='all',
                       choices=['safety', 'control', 'speed', 'racing', 'all'],
                       help='Specific phase to test (default: all phases)')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering for faster testing')
    parser.add_argument('--agents', type=str, nargs='+',
                       help='Specific agents to test (default: all)')
    parser.add_argument('--software-render', action='store_true',
                       help='Use Panda3D software renderer to avoid GPU driver issues')
    
    args = parser.parse_args()
    
    # Create tester
    tester = CurriculumRacingTester(args.models_dir)
    
    # Load models
    available_agents = tester.load_models()
    if not available_agents:
        print("[ERROR] No trained models found!")
        return
    
    # Select agents to test
    if args.agents:
        test_agents = [agent for agent in args.agents if agent in available_agents]
        if not test_agents:
            print(f"[ERROR] No specified agents found! Available: {available_agents}")
            return
    else:
        test_agents = available_agents
    
    print(f"\nTesting agents: {test_agents}")
    
    # Run testing
    render = not args.no_render
    
    if args.phase == 'all':
        # Full curriculum testing
        tester.run_curriculum_testing(test_agents, args.races_per_phase, render, args.software_render)
    else:
        # Single phase testing
        print(f"\nTesting {args.phase.upper()} phase only...")
        for race_num in range(args.races_per_phase):
            tester.run_race(test_agents, race_num, render, args.phase, args.software_render)
    
    # Print final statistics
    tester.print_final_statistics(test_agents)


if __name__ == '__main__':
    main()
