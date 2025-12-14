"""
Watch multiple trained agents race side-by-side on custom speedway with SIMPLE CLEAN REWARD SYSTEM.

Agents are trained with F13's simple clean formula:
- PROGRESS: +20 * forward movement (main reward driver)
- SPEED: +1 * normalized speed (0-120 km/h)
- COMPLETION: +1 * track completion bonus
- TIME PENALTY: -0.02 per step (encourages forward movement)
- MAJOR PENALTIES: -20 for off-road, crash, white-line
- STRICT BOUNDARIES: out_of_road_done=True (terminates immediately)

Usage:
    python race_side_by_side_curve.py --agents F14 F15 --episodes 3
    python race_side_by_side_curve.py --agents TEST agent_0v2 --episodes 5
"""
import os
import sys
import argparse
from typing import List, Tuple, Dict

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

# Add src to path
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments.multi_agent_custom_speedway_curve_env import MultiAgentCustomSpeedwayCurveEnv


def load_agent_model(model_path: str, vecnorm_path: str = None, skip_vecnorm: bool = False) -> Tuple[PPO, object]:
    """Load a trained agent model and VecNormalize stats."""

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = PPO.load(model_path)

    # Load VecNormalize stats if available - use SB3's method
    vecnorm_stats = None
    if vecnorm_path and os.path.exists(vecnorm_path) and not skip_vecnorm:
        try:
            from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
            
            # Create a dummy env for VecNormalize to attach to
            def make_dummy():
                import gymnasium as gym
                return gym.make('CartPole-v1')  # Dummy env, we only need obs_rms
            
            dummy_venv = DummyVecEnv([make_dummy])
            vecnorm_stats = VecNormalize.load(vecnorm_path, dummy_venv)
            dummy_venv.close()
        except Exception as e:
            print(f"   WARNING: VecNormalize shape mismatch - continuing without normalization")
            vecnorm_stats = None

    return model, vecnorm_stats


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


def race_agents_synchronized(agent_data: List[Tuple[str, str, str]], 
                             track: str, episodes: int, render: bool):
    """Race multiple agents side-by-side in multi-agent environment."""
    
    print(f"\n{'='*70}")
    print(f"🏁 SIDE-BY-SIDE RACING (SIMPLE CLEAN REWARD SYSTEM)")
    print(f"{'='*70}")
    print(f"Agents: {', '.join([name for name, _, _ in agent_data])}")
    print(f"Track: custom_speedway_curve")
    print(f"Episodes: {episodes}")
    print(f"Reward: +20*progress | +1*speed | +1*completion | -0.02*time | -20 penalties")
    print(f"Boundaries: STRICT (out_of_road_done=True - terminates on off-road)")
    print(f"{'='*70}\n")
    
    # Load all agent models
    agent_models = {}
    agent_vecnorms = {}
    agent_names = [name for name, _, _ in agent_data]
    
    for idx, (agent_name, model_path, vecnorm_path) in enumerate(agent_data):
        model, vecnorm = load_agent_model(model_path, vecnorm_path, skip_vecnorm=False)
        agent_id = f"agent{idx}"  # MATCH TRAINING: use agent0, agent1 format
        agent_models[agent_id] = model
        agent_vecnorms[agent_id] = vecnorm
    
    # Create multi-agent CURVE-AWARE environment
    print(f"\n🏗️  Creating curve-aware multi-agent environment...")

    # MATCH TRAINING EXACTLY - no staggered spawning, no respawn configs
    env_config = {
        'num_agents': len(agent_data),
        'use_render': render,
        'map_config': {
            'lane_num': 2,  # 2 lanes = one per agent
            'lane_width': 12.0,  # Extra wide lanes for curve navigation
        },
        'start_seed': 42,

        # LENIENT MODE: Let agents continue even after crashes/off-road
        'crash_vehicle_done': False,  # Don't terminate on crash
        'crash_object_done': False,   # Don't terminate on object crash
        'out_of_road_done': False,    # Don't terminate on off-road
        'random_spawn_lane_index': False,  # MATCH TRAINING: Fixed spawn lanes

        'horizon': 1500,
        'vehicle_config': {
            'max_speed_km_h': 120,
        },

        # MATCH TRAINING: Enable terrain and sidewalk
        'show_terrain': True,
        'show_sidewalk': True,
    }
    
    # Race episodes - create fresh environment for each episode to prevent agent spawning issues
    for episode in range(episodes):
        print(f"\n{'='*70}")
        print(f"🏁 Episode {episode + 1}/{episodes}")
        print(f"{'='*70}")

        # Create fresh environment for this episode
        env = MultiAgentCustomSpeedwayCurveEnv(env_config)
        obs_dict, _ = env.reset()
        
        # Map environment agent IDs to our loaded models
        # Environment might create agent0, agent1, OR different IDs
        env_agent_ids = sorted(obs_dict.keys())
        if len(env_agent_ids) != len(agent_models):
            print(f"⚠️  Warning: Environment has {len(env_agent_ids)} agents but we loaded {len(agent_models)} models")
            print(f"   Environment agents: {env_agent_ids}")
            print(f"   Loaded models: {list(agent_models.keys())}")
        
        # Create mapping from environment agent IDs to our models
        agent_id_map = {}
        for idx, env_agent_id in enumerate(env_agent_ids):
            if idx < len(agent_models):
                model_agent_id = f"agent{idx}"  # Match dictionary keys: agent0, agent1
                agent_id_map[env_agent_id] = model_agent_id
        
        # Track stats
        episode_stats = {name: {
            'reward': 0.0,
            'steps': 0,
            'speeds': [],
            'crashed': False,
            'crash_reported': False,
        } for name in agent_names}
        
        done = False
        step = 0
        max_steps = 5000
        terminated_dict = {}  # Initialize
        truncated_dict = {}
        # Track which agents have terminated (sticky flags that stay True once set)
        agents_finished = {f"agent{i}": False for i in range(len(agent_names))}

        while not done and step < max_steps:
            step += 1

            # Check if our agents are done BEFORE stepping (prevent crash)
            if step > 10:
                # Check our sticky termination tracker
                all_finished = all(agents_finished.values())
                if all_finished:
                    print(f"   ✅ Both agents terminated. Ending episode at step {step}")
                    break

            actions = {}

            # Get actions from each agent's model using the mapping
            for env_agent_id, obs in obs_dict.items():
                # Map environment agent ID to our loaded model
                model_agent_id = agent_id_map.get(env_agent_id)
                if model_agent_id is None:
                    # This agent doesn't have a model, use neutral action [0, 0]
                    actions[env_agent_id] = np.array([0.0, 0.0])
                    continue

                model = agent_models[model_agent_id]
                vecnorm = agent_vecnorms[model_agent_id]

                # Apply VecNormalize if available (CRITICAL!)
                if vecnorm is not None and hasattr(vecnorm, 'obs_rms'):
                    obs_rms = vecnorm.obs_rms
                    epsilon = 1e-8
                    # Normalize: (obs - mean) / sqrt(var + epsilon), then clip
                    obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -10, 10)

                # Add batch dimension for prediction
                obs = obs.reshape(1, -1)
                action, _ = model.predict(obs, deterministic=True)
                
                # Flatten action if it's 2D (batch dimension)
                if isinstance(action, np.ndarray) and action.ndim > 1:
                    action = action.flatten()
                
                if step == 1:  # Debug first step
                    obs_stats = f"min={obs.min():.3f}, max={obs.max():.3f}, mean={obs.mean():.3f}"
                    vecnorm_status = "WITH VecNorm" if vecnorm is not None else "NO VecNorm"
                    print(f"   DEBUG {env_agent_id}->{model_agent_id} ({vecnorm_status}): obs_shape={obs.shape}, {obs_stats}")
                    print(f"   DEBUG {env_agent_id}->{model_agent_id}: action_shape={action.shape}, action={action}")

                actions[env_agent_id] = action
            
            # Step environment
            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(actions)

            # Debug vehicle states on first few steps
            if step <= 3:
                for env_agent_id in obs_dict.keys():
                    model_agent_id = agent_id_map.get(env_agent_id)
                    if model_agent_id is None:
                        continue
                    agent_idx = int(model_agent_id.replace('agent', ''))  # Parse agent0, agent1 format
                    if agent_idx >= len(agent_names):
                        continue
                    name = agent_names[agent_idx]
                    info = info_dict.get(env_agent_id, {})
                    vehicle_state = info.get('vehicle_state', {})
                    reward_components = info.get('reward_components', {})
                    print(f"   [{step}] {name}: speed={vehicle_state.get('speed', 0)*120:.1f}km/h, "
                          f"on_road={vehicle_state.get('on_road', False)}, "
                          f"crashed={vehicle_state.get('crashed', False)}, "
                          f"reward={reward_dict.get(env_agent_id, 0):.1f}")

            # Update stats using the mapping
            for env_agent_id in obs_dict.keys():
                model_agent_id = agent_id_map.get(env_agent_id)
                if model_agent_id is None:
                    continue  # Skip unmapped agents
                
                # Get the index for agent_names
                agent_idx = int(model_agent_id.replace('agent', ''))  # Parse agent0, agent1 format
                if agent_idx >= len(agent_names):
                    continue
                    
                name = agent_names[agent_idx]
                episode_stats[name]['reward'] += float(reward_dict[env_agent_id])
                episode_stats[name]['steps'] = step

                info = info_dict.get(env_agent_id, {})
                # Extract speed from vehicle_state (it's normalized 0-1, multiply by max_speed)
                vehicle_state = info.get('vehicle_state', {})
                speed_normalized = vehicle_state.get('speed', 0.0)
                speed_kmh = speed_normalized * 120.0  # max_speed_km_h is 120
                episode_stats[name]['speeds'].append(speed_kmh)
                
                # Only report crash once
                if (info.get('crash', False) or info.get('crashed', False)) and not episode_stats[name]['crash_reported']:
                    episode_stats[name]['crashed'] = True
                    episode_stats[name]['crash_reported'] = True
                    print(f"   💥 {name} crashed at step {step}")
            
            # Check if OUR two agents are done (ignore any extra agents MetaDrive might spawn)
            our_agents_terminated = all(
                terminated_dict.get(f"agent{i}", False) or truncated_dict.get(f"agent{i}", False)
                for i in range(len(agent_names))
            )
            if our_agents_terminated and step > 10:  # Add step check to avoid early termination
                print(f"   ✅ Both agents finished! Ending episode at step {step}")
                done = True
            else:
                done = False
            
            # If individual agents terminate, mark them in our sticky tracker
            for env_agent_id, is_terminated in terminated_dict.items():
                if env_agent_id != '__all__' and is_terminated:
                    # Mark as finished in our tracker
                    if env_agent_id in agents_finished:
                        agents_finished[env_agent_id] = True

                    model_agent_id = agent_id_map.get(env_agent_id)
                    if model_agent_id:
                        agent_idx = int(model_agent_id.replace('agent', ''))
                        if agent_idx < len(agent_names) and not episode_stats[agent_names[agent_idx]]['crash_reported']:
                            name = agent_names[agent_idx]
                            print(f"   🏁 {name} finished/terminated at step {step}")
                            episode_stats[name]['crash_reported'] = True
        
        # Episode summary
        print(f"\n{'─'*70}")
        print(f"📊 Episode {episode + 1} Results:")
        print(f"{'─'*70}")

        sorted_names = sorted(agent_names, key=lambda n: episode_stats[n]['reward'], reverse=True)

        for rank, name in enumerate(sorted_names, 1):
            stats = episode_stats[name]
            avg_speed = np.mean(stats['speeds']) if stats['speeds'] else 0.0
            max_speed = np.max(stats['speeds']) if stats['speeds'] else 0.0
            status = "💥 CRASHED" if stats['crashed'] else "✅ COMPLETED"
            print(f"{rank}. {name:12s} {status}")
            print(f"   Reward: {stats['reward']:7.1f} | Steps: {stats['steps']:4d}")
            print(f"   Avg Speed: {avg_speed:5.1f} | Max Speed: {max_speed:5.1f}")

        # Close environment after each episode
        env.close()
    
    print(f"\n{'='*70}")
    print(f"🏁 Racing Complete!")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Race agents side-by-side in synchronized environments.')
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--agents', type=str, nargs='+', help='Agent names')
    group.add_argument('--auto-detect', action='store_true', help='Auto-detect all agents')
    
    parser.add_argument('--results-dir', type=str, default=None, help='Results directory')
    parser.add_argument('--track', type=str, default='custom_speedway', help='Track name')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes')
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
                    f'ppo_{args.track}.zip',  # Primary: ppo_custom_speedway.zip
                    f'{agent_name}_{args.track}.zip',  # Secondary: agent_name_custom_speedway.zip
                    f'ppo_custom_speedway_curve.zip',  # Fallback: curve variant
                    f'{agent_name}_custom_speedway_curve.zip',  # Fallback: agent curve variant
                    f'best_model.zip',  # Last resort
                ]
                
                model_path = None
                for pattern in model_patterns:
                    candidate = os.path.join(agent_dir, pattern)
                    if os.path.exists(candidate):
                        model_path = candidate
                        break
                
                if model_path:
                    vecnorm_patterns = [
                        f'vecnorm_{args.track}.pkl',  # Primary: vecnorm_custom_speedway.pkl
                        f'vecnorm_{agent_name}_{args.track}.pkl',  # Secondary: vecnorm_agent_name_custom_speedway.pkl
                        f'vecnorm_custom_speedway_curve.pkl',  # Fallback: curve variant
                        f'vecnorm_{agent_name}_custom_speedway_curve.pkl',  # Fallback: agent curve variant
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
    
    # Race!
    race_agents_synchronized(
        agent_data, 
        args.track, 
        args.episodes, 
        not args.no_render
    )


if __name__ == '__main__':
    main()
