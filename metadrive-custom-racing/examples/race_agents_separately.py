"""
Watch multiple agents race SEPARATELY on the custom speedway track and compare their performance.

Since the agents were trained for single-agent racing, this script runs them one at a time
and compares their lap times, speeds, and overall performance.

Usage:
    python race_agents_separately.py --agents Alek Saegan --episodes 3
"""
import os
import sys
import argparse
from typing import List, Tuple, Dict

# Add src to path
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments.single_car_racing import create_racing_environment


def load_agent_model(model_path: str, vecnorm_path: str = None) -> Tuple[PPO, str]:
    """Load a trained agent model."""
    print(f"📦 Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = PPO.load(model_path)
    
    vecnorm_loaded = None
    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"   VecNormalize: {os.path.basename(vecnorm_path)}")
        vecnorm_loaded = vecnorm_path
    
    return model, vecnorm_loaded


def race_single_agent(agent_name: str, model: PPO, vecnorm_path: str, 
                     track: str, episodes: int, render: bool) -> Dict:
    """Race a single agent and return performance statistics."""
    
    print(f"\n{'='*70}")
    print(f"🏁 Racing: {agent_name}")
    print(f"{'='*70}")
    
    # Create environment
    base_env = create_racing_environment(track_name=track, use_render=render)
    vec_env = DummyVecEnv([lambda: base_env])
    
    # Apply VecNormalize if available
    if vecnorm_path:
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    
    stats = {
        'total_reward': 0.0,
        'total_steps': 0,
        'episodes_completed': 0,
        'crashes': 0,
        'max_speed': 0.0,
        'avg_speed': 0.0,
        'episode_rewards': [],
        'episode_steps': [],
    }
    
    for episode in range(episodes):
        print(f"\n  Episode {episode + 1}/{episodes}")
        
        obs = vec_env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        speeds = []
        crashed = False
        
        while not done and episode_steps < 5000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            
            episode_reward += float(reward[0])
            episode_steps += 1
            
            # Extract speed from info
            if len(info) > 0:
                speed = info[0].get('speed', 0.0)
                speeds.append(speed)
                
                # Check for crash
                if info[0].get('crash', False) or info[0].get('crash_vehicle', False):
                    crashed = True
                    print(f"    💥 Crashed at step {episode_steps}")
        
        # Update statistics
        stats['total_reward'] += episode_reward
        stats['total_steps'] += episode_steps
        stats['episodes_completed'] += 1
        stats['episode_rewards'].append(episode_reward)
        stats['episode_steps'].append(episode_steps)
        
        if crashed:
            stats['crashes'] += 1
        
        if speeds:
            avg_speed = np.mean(speeds)
            max_speed = np.max(speeds)
            stats['avg_speed'] += avg_speed
            stats['max_speed'] = max(stats['max_speed'], max_speed)
            print(f"    Reward: {episode_reward:.2f} | Steps: {episode_steps} | Avg Speed: {avg_speed:.2f} | Max Speed: {max_speed:.2f}")
        else:
            print(f"    Reward: {episode_reward:.2f} | Steps: {episode_steps}")
    
    # Calculate averages
    if stats['episodes_completed'] > 0:
        stats['avg_reward'] = stats['total_reward'] / stats['episodes_completed']
        stats['avg_steps'] = stats['total_steps'] / stats['episodes_completed']
        stats['avg_speed'] = stats['avg_speed'] / stats['episodes_completed']
    
    vec_env.close()
    return stats


def find_agent_models(results_dir: str, track_name: str = 'custom_speedway') -> List[Tuple[str, str, str]]:
    """Auto-detect trained agent models."""
    agents = []
    
    search_dirs = [results_dir, os.path.join(os.path.dirname(results_dir), 'examples')]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        for item in os.listdir(search_dir):
            if item.startswith('results_agent_'):
                agent_name = item.replace('results_agent_', '')
                agent_dir = os.path.join(search_dir, item)
                
                if not os.path.isdir(agent_dir):
                    continue
                
                model_patterns = [
                    f'{agent_name}_{track_name}.zip',
                    f'ppo_{track_name}.zip',
                    f'ppo_{track_name}_v3.zip',
                    f'ppo_{track_name}_v2.zip',
                    f'best_model.zip',
                ]
                
                model_path = None
                for pattern in model_patterns:
                    candidate = os.path.join(agent_dir, pattern)
                    if os.path.exists(candidate):
                        model_path = candidate
                        break
                
                if model_path:
                    vecnorm_patterns = [
                        f'vecnorm_{agent_name}_{track_name}.pkl',
                        f'vecnorm_{track_name}.pkl',
                        f'vecnorm_{track_name}_v3.pkl',
                        f'vecnorm_{track_name}_v2.pkl',
                    ]
                    
                    vecnorm_path = None
                    for pattern in vecnorm_patterns:
                        candidate = os.path.join(agent_dir, pattern)
                        if os.path.exists(candidate):
                            vecnorm_path = candidate
                            break
                    
                    agents.append((agent_name, model_path, vecnorm_path))
    
    return agents


def main():
    parser = argparse.ArgumentParser(description='Race multiple agents separately and compare results.')
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--agents', type=str, nargs='+', help='Agent names')
    group.add_argument('--auto-detect', action='store_true', help='Auto-detect all agents')
    
    parser.add_argument('--results-dir', type=str, default=None, help='Results directory')
    parser.add_argument('--track', type=str, default='custom_speedway', help='Track name')
    parser.add_argument('--episodes', type=int, default=3, help='Episodes per agent')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    
    args = parser.parse_args()
    
    results_dir = args.results_dir or os.path.join(os.path.dirname(__file__), '..', 'results')
    
    # Find agents
    agent_data = []
    
    if args.agents:
        for agent_name in args.agents:
            search_dirs = [
                os.path.join(results_dir, f'results_agent_{agent_name}'),
                os.path.join(os.path.dirname(__file__), f'results_agent_{agent_name}'),
            ]
            
            found = False
            for agent_dir in search_dirs:
                if not os.path.exists(agent_dir):
                    continue
                
                model_patterns = [
                    f'{agent_name}_{args.track}.zip',
                    f'ppo_{args.track}.zip',
                    f'ppo_{args.track}_v3.zip',
                    f'ppo_{args.track}_v2.zip',
                    f'best_model.zip',
                ]
                
                model_path = None
                for pattern in model_patterns:
                    candidate = os.path.join(agent_dir, pattern)
                    if os.path.exists(candidate):
                        model_path = candidate
                        break
                
                if model_path:
                    vecnorm_patterns = [
                        f'vecnorm_{agent_name}_{args.track}.pkl',
                        f'vecnorm_{args.track}.pkl',
                        f'vecnorm_{args.track}_v3.pkl',
                        f'vecnorm_{args.track}_v2.pkl',
                    ]
                    
                    vecnorm_path = None
                    for pattern in vecnorm_patterns:
                        candidate = os.path.join(agent_dir, pattern)
                        if os.path.exists(candidate):
                            vecnorm_path = candidate
                            break
                    
                    agent_data.append((agent_name, model_path, vecnorm_path))
                    print(f"✅ Found {agent_name}: {os.path.basename(model_path)}")
                    found = True
                    break
            
            if not found:
                print(f"❌ Model not found for {agent_name}")
                return
    
    elif args.auto_detect:
        print("🔍 Auto-detecting agents...")
        agent_data = find_agent_models(results_dir, args.track)
        if len(agent_data) < 2:
            print(f"❌ Found only {len(agent_data)} agent(s). Need at least 2.")
            return
    
    else:
        print("❌ Please specify --agents or --auto-detect")
        return
    
    if not agent_data:
        print("❌ No agents found!")
        return
    
    print(f"\n{'='*70}")
    print(f"🏁 SEQUENTIAL AGENT RACING CHAMPIONSHIP")
    print(f"{'='*70}")
    print(f"Track: {args.track}")
    print(f"Agents: {len(agent_data)}")
    print(f"Episodes per agent: {args.episodes}")
    print(f"{'='*70}")
    
    # Load and race each agent
    all_stats = {}
    
    for agent_name, model_path, vecnorm_path in agent_data:
        try:
            model, vecnorm = load_agent_model(model_path, vecnorm_path)
            stats = race_single_agent(
                agent_name, model, vecnorm, 
                args.track, args.episodes, 
                not args.no_render
            )
            all_stats[agent_name] = stats
        except Exception as e:
            print(f"\n❌ Error racing {agent_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final leaderboard
    print(f"\n{'='*70}")
    print(f"🏆 FINAL LEADERBOARD")
    print(f"{'='*70}")
    
    sorted_agents = sorted(all_stats.items(), key=lambda x: x[1]['avg_reward'], reverse=True)
    
    for rank, (agent_name, stats) in enumerate(sorted_agents, 1):
        print(f"\n{rank}. {agent_name}")
        print(f"   Avg Reward: {stats['avg_reward']:.2f}")
        print(f"   Avg Steps: {stats['avg_steps']:.1f}")
        print(f"   Avg Speed: {stats['avg_speed']:.2f}")
        print(f"   Max Speed: {stats['max_speed']:.2f}")
        print(f"   Crashes: {stats['crashes']}/{stats['episodes_completed']}")
        print(f"   Episode Rewards: {[f'{r:.1f}' for r in stats['episode_rewards']]}")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
