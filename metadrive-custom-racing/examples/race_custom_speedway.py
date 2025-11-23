"""
Watch multiple agents race together on the custom speedway track.

This script loads multiple trained agents (e.g., Alek and Saegan) and
has them race simultaneously on the custom speedway track.

Usage:
    # Race 2 specific agents
    python race_custom_speedway.py --agents Alek Saegan
    
    # Race all agents found in results directory
    python race_custom_speedway.py --auto-detect
    
    # Specify custom paths
    python race_custom_speedway.py --model-paths results/results_agent_Alek/Alek_custom_speedway.zip results/results_agent_Saegan/Saegan_custom_speedway.zip
"""
import os
import sys
import argparse
import time
from typing import List, Tuple, Dict

# Add src to path
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from environments.multi_agent_custom_speedway_env import MultiAgentCustomSpeedwayEnv
    from environments.multi_agent_oval_right_env import MultiAgentOvalEnv
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    print("❌ Error: Multi-agent environments not available")
    print("   Make sure multi_agent_custom_speedway_env.py exists in src/environments/")
    MULTI_AGENT_AVAILABLE = False
    sys.exit(1)


def load_agent_model(model_path: str, vecnorm_path: str = None) -> Tuple[PPO, dict]:
    """Load a trained agent model and its VecNormalize stats if available."""
    print(f"📦 Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = PPO.load(model_path)
    
    # Try to load VecNormalize stats - we'll load the stats manually
    vec_normalize_data = None
    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"   Loading VecNormalize stats from: {vecnorm_path}")
        # Load just the normalization parameters
        import pickle
        with open(vecnorm_path, 'rb') as f:
            vec_normalize_data = pickle.load(f)
    
    return model, vec_normalize_data


def find_agent_models(results_dir: str, track_name: str = 'custom_speedway') -> List[Tuple[str, str, str]]:
    """Auto-detect trained agent models in results directory.
    
    Returns:
        List of (agent_name, model_path, vecnorm_path) tuples
    """
    agents = []
    
    if not os.path.exists(results_dir):
        return agents
    
    # Look for results_agent_* directories (in both results/ and examples/)
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
                
                # Try multiple naming patterns
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
                    # Look for vecnorm file with multiple patterns
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
                    print(f"✅ Found agent: {agent_name}")
                    print(f"   Model: {os.path.basename(model_path)}")
                    if vecnorm_path:
                        print(f"   VecNorm: {os.path.basename(vecnorm_path)}")
    
    return agents


def race_agents(agent_data: List[Tuple[str, str, str]], episodes: int = 3, render: bool = True):
    """Race multiple agents together on the custom speedway.
    
    Args:
        agent_data: List of (agent_name, model_path, vecnorm_path) tuples
        episodes: Number of racing episodes
        render: Whether to render the environment
    """
    num_agents = len(agent_data)
    
    if num_agents < 2:
        print("⚠️  Need at least 2 agents to race!")
        return
    
    print(f"\n{'='*70}")
    print(f"🏁 MULTI-AGENT RACE ON CUSTOM SPEEDWAY")
    print(f"{'='*70}")
    print(f"Number of agents: {num_agents}")
    print(f"Episodes: {episodes}")
    print(f"Track: Custom Speedway (complex layout)")
    print(f"{'='*70}\n")
    
    # Load all agent models
    models = []
    for agent_name, model_path, vecnorm_path in agent_data:
        try:
            model, vec_norm_data = load_agent_model(model_path, vecnorm_path)
            models.append((agent_name, model, vec_norm_data))
        except Exception as e:
            print(f"❌ Failed to load {agent_name}: {e}")
            import traceback
            traceback.print_exc()
            return
    
    print(f"\n✅ All {num_agents} agents loaded successfully!\n")
    
    # Create multi-agent environment
    # NOTE: Using the simpler right_oval track for now since custom speedway
    # has complex curves that need special handling in multi-agent mode
    env_config = {
        "num_agents": num_agents,
        "start_seed": 42,
        "use_render": render,
        "crash_done": False,  # Don't end episode on crash
        "horizon": 10000,  # Long horizon for racing
        "success_reward": 10.0,
        "driving_reward": 1.0,
        "speed_reward": 0.5,
        "map_config": {
            "lane_num": 1,  # Single lane for racing
            "lane_width": 20.0,
        },
        # Stagger starting positions
        "agent_configs": {
            f"agent{i}": {
                "spawn_longitude": -i * 50.0,  # Space agents 50 units apart
                "spawn_lateral": 0.0,
            } for i in range(num_agents)
        }
    }
    
    print("🏗️  Creating multi-agent racing environment...")
    print("   (Using right_oval track - complex speedway multi-agent support coming soon)")
    env = MultiAgentOvalEnv(env_config)
    
    # Track statistics
    agent_stats = {name: {'total_reward': 0, 'crashes': 0, 'steps': 0, 'episodes': 0} 
                   for name, _, _ in models}
    
    # Run episodes
    for episode in range(episodes):
        print(f"\n{'='*70}")
        print(f"🏁 Episode {episode + 1}/{episodes}")
        print(f"{'='*70}")
        
        reset_result = env.reset()
        # Handle both tuple (obs, info) and dict returns
        if isinstance(reset_result, tuple):
            obs_dict = reset_result[0]
        else:
            obs_dict = reset_result
            
        done_dict = {f"agent{i}": False for i in range(num_agents)}
        done_dict["__all__"] = False
        
        episode_rewards = {name: 0 for name, _, _ in models}
        episode_steps = 0
        max_steps = 5000
        
        while not done_dict["__all__"] and episode_steps < max_steps:
            actions = {}
            
            # Get action from each agent's model
            for i, (agent_name, model, vec_norm_data) in enumerate(models):
                agent_id = f"agent{i}"
                
                if not done_dict.get(agent_id, False):
                    obs = obs_dict[agent_id]
                    
                    # Normalize observation if VecNormalize was used
                    if vec_norm_data is not None and hasattr(vec_norm_data, 'obs_rms'):
                        # Apply running mean/std normalization
                        obs_rms = vec_norm_data.obs_rms
                        epsilon = 1e-8
                        
                        # Check if observation shapes match
                        if obs.shape == obs_rms.mean.shape:
                            obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -10, 10)
                        else:
                            # Observation space mismatch - skip normalization
                            # This happens when training env differs from racing env
                            pass
                    
                    action, _ = model.predict(obs, deterministic=True)
                    actions[agent_id] = action
                else:
                    # Agent is done, provide zero action
                    actions[agent_id] = np.array([0.0, 0.0])
            
            # Step environment
            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(actions)
            
            # Update done dictionary
            for agent_id in done_dict.keys():
                if agent_id != "__all__":
                    done_dict[agent_id] = terminated_dict.get(agent_id, False) or truncated_dict.get(agent_id, False)
            done_dict["__all__"] = terminated_dict.get("__all__", False) or truncated_dict.get("__all__", False)
            
            # Track rewards
            for i, (agent_name, _, _) in enumerate(models):
                agent_id = f"agent{i}"
                reward = reward_dict.get(agent_id, 0.0)
                episode_rewards[agent_name] += reward
                
                # Check for crashes
                if info_dict.get(agent_id, {}).get('crashed', False):
                    agent_stats[agent_name]['crashes'] += 1
                    print(f"💥 {agent_name} crashed at step {episode_steps}")
            
            episode_steps += 1
            
            # Print progress every 500 steps
            if episode_steps % 500 == 0:
                print(f"   Step {episode_steps}: " + 
                      " | ".join([f"{name}: {episode_rewards[name]:.1f}" for name, _, _ in models]))
        
        # Episode summary
        print(f"\n📊 Episode {episode + 1} Results:")
        print(f"   Total steps: {episode_steps}")
        for agent_name, _, _ in models:
            episode_rewards[agent_name] = episode_rewards[agent_name]
            agent_stats[agent_name]['total_reward'] += episode_rewards[agent_name]
            agent_stats[agent_name]['steps'] += episode_steps
            agent_stats[agent_name]['episodes'] += 1
            print(f"   {agent_name}: {episode_rewards[agent_name]:.2f} reward")
        
        time.sleep(1)
    
    # Final statistics
    print(f"\n{'='*70}")
    print(f"🏆 FINAL RACE STATISTICS")
    print(f"{'='*70}")
    
    sorted_agents = sorted(agent_stats.items(), key=lambda x: x[1]['total_reward'], reverse=True)
    
    for rank, (agent_name, stats) in enumerate(sorted_agents, 1):
        avg_reward = stats['total_reward'] / max(stats['episodes'], 1)
        print(f"{rank}. {agent_name}:")
        print(f"   Total Reward: {stats['total_reward']:.2f}")
        print(f"   Avg Reward: {avg_reward:.2f}")
        print(f"   Crashes: {stats['crashes']}")
        print(f"   Total Steps: {stats['steps']}")
    
    print(f"{'='*70}\n")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Race multiple agents on custom speedway track.')
    
    # Agent selection methods (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--agents', type=str, nargs='+', 
                      help='Agent names (e.g., --agents Alek Saegan)')
    group.add_argument('--model-paths', type=str, nargs='+',
                      help='Direct paths to model files')
    group.add_argument('--auto-detect', action='store_true',
                      help='Auto-detect all trained agents in results directory')
    
    # Configuration
    parser.add_argument('--results-dir', type=str, default=None,
                      help='Results directory (default: ../results)')
    parser.add_argument('--track', type=str, default='custom_speedway',
                      help='Track name')
    parser.add_argument('--episodes', type=int, default=3,
                      help='Number of racing episodes')
    parser.add_argument('--no-render', action='store_true',
                      help='Disable rendering (faster but no visualization)')
    
    args = parser.parse_args()
    
    # Setup results directory
    results_dir = args.results_dir or os.path.join(os.path.dirname(__file__), '..', 'results')
    
    # Determine which agents to race
    agent_data = []
    
    if args.model_paths:
        # Direct model paths provided
        for model_path in args.model_paths:
            agent_name = os.path.basename(model_path).replace('.zip', '').replace(f'_{args.track}', '')
            agent_dir = os.path.dirname(model_path)
            vecnorm_path = os.path.join(agent_dir, f'vecnorm_{agent_name}_{args.track}.pkl')
            if not os.path.exists(vecnorm_path):
                vecnorm_path = None
            agent_data.append((agent_name, model_path, vecnorm_path))
    
    elif args.agents:
        # Agent names provided, find their models
        for agent_name in args.agents:
            # Search in multiple locations
            search_dirs = [
                os.path.join(results_dir, f'results_agent_{agent_name}'),
                os.path.join(os.path.dirname(__file__), f'results_agent_{agent_name}'),
            ]
            
            found = False
            for agent_dir in search_dirs:
                if not os.path.exists(agent_dir):
                    continue
                
                # Try multiple naming patterns
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
                    # Look for vecnorm file
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
                    if vecnorm_path:
                        print(f"   VecNorm: {os.path.basename(vecnorm_path)}")
                    found = True
                    break
            
            if not found:
                print(f"❌ Model not found for {agent_name}")
                print(f"   Searched in:")
                for search_dir in search_dirs:
                    print(f"   - {search_dir}")
                return
    
    elif args.auto_detect:
        # Auto-detect all agents
        print("🔍 Auto-detecting trained agents...")
        agent_data = find_agent_models(results_dir, args.track)
        
        if len(agent_data) < 2:
            print(f"❌ Found only {len(agent_data)} agent(s). Need at least 2 to race.")
            return
    
    else:
        print("❌ Please specify agents using --agents, --model-paths, or --auto-detect")
        parser.print_help()
        return
    
    if not agent_data:
        print("❌ No agents found to race!")
        return
    
    # Start the race!
    race_agents(agent_data, episodes=args.episodes, render=not args.no_render)


if __name__ == '__main__':
    main()
