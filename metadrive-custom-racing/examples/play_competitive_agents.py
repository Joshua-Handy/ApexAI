"""
Play multiple trained competitive agents on MetaDrive racing tracks with visualization.

This script loads multiple saved single-agent PPO models and runs them sequentially
to demonstrate competitive racing behavior.
"""
import argparse
import os
import sys
import time
import glob
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from environments.single_car_racing import create_racing_environment
except ImportError:
    print("Warning: Could not import custom environment, using basic MetaDrive")
    from metadrive import MetaDriveEnv
    
    def create_racing_environment(track_name: str, use_render: bool = False):
        """Fallback environment builder"""
        env_config = {
            "num_scenarios": 1,
            "traffic_density": 0.0,
            "start_seed": 42,
            "map": "O",  # Use built-in oval
            "use_render": use_render,
        }
        return MetaDriveEnv(env_config)


def load_agent(model_path: str) -> tuple[PPO, VecNormalize]:
    """Load a trained agent from a saved model."""
    print(f"Loading agent: {os.path.basename(model_path)}")
    
    # Create a dummy environment for loading the model
    dummy_env = create_racing_environment(track_name='right_oval', use_render=False)
    
    # Load the model
    model = PPO.load(model_path, env=dummy_env)
    
    # Check for VecNormalize stats
    stats_path = model_path.replace('.zip', '_vecnormalize.pkl')
    vec_normalize = None
    if os.path.exists(stats_path):
        print(f"  Loading VecNormalize stats from {stats_path}")
        vec_normalize = VecNormalize.load(stats_path, dummy_env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False
    
    dummy_env.close()
    return model, vec_normalize


def play_agents_sequentially(agent_models: list, track: str, episodes_per_agent: int = 3):
    """Play multiple agents sequentially on the same track."""
    
    for i, (agent_name, model, vec_normalize) in enumerate(agent_models):
        print(f"\n🏁 Now watching Agent {i}: {agent_name}")
        print("=" * 50)
        
        # Create environment for this agent
        env = create_racing_environment(track_name=track, use_render=True)
        
        if vec_normalize:
            # Wrap with VecNormalize if available
            vec_env = DummyVecEnv([lambda: env])
            vec_env = vec_normalize
            current_env = vec_env
        else:
            current_env = env
        
        # Play episodes for this agent
        for episode in range(episodes_per_agent):
            print(f"Episode {episode + 1}/{episodes_per_agent}")
            
            obs = current_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
                
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 3000:  # Max 3000 steps per episode
                action, _ = model.predict(obs, deterministic=True)
                
                step_result = current_env.step(action)
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                
                total_reward += reward
                steps += 1
                
                # Check if agent completed the track
                if isinstance(info, list) and len(info) > 0:
                    info = info[0]
                if isinstance(info, dict) and info.get('arrive_dest', False):
                    print(f"  🏆 Agent completed the track!")
                    break
            
            print(f"  Final reward: {total_reward:.2f}, Steps: {steps}")
            
            if episode < episodes_per_agent - 1:
                input("  Press Enter for next episode...")
        
        env.close()
        
        if i < len(agent_models) - 1:
            input(f"\nPress Enter to watch next agent...")


def main():
    parser = argparse.ArgumentParser(description='Play multiple trained competitive agents')
    parser.add_argument('--models-dir', type=str, default='results',
                       help='Directory containing trained models')
    parser.add_argument('--model-pattern', type=str, default='competitive_agent_*',
                       help='Pattern to match competitive agent model files')
    parser.add_argument('--track', type=str, default='right_oval',
                       help='Track name (should match training track)')
    parser.add_argument('--episodes-per-agent', type=int, default=3,
                       help='Number of episodes to play per agent')
    parser.add_argument('--specific-models', nargs='+', default=None,
                       help='Specific model files to play (instead of pattern matching)')
    
    args = parser.parse_args()
    
    print("🏁 Multi-Agent Racing Playback")
    print("=" * 40)
    print(f"Track: {args.track}")
    print(f"Episodes per agent: {args.episodes_per_agent}")
    
    # Find models to play
    if args.specific_models:
        model_files = args.specific_models
    else:
        pattern = os.path.join(args.models_dir, f"{args.model_pattern}_{args.track}.zip")
        model_files = glob.glob(pattern)
        
        if not model_files:
            # Try without track suffix
            pattern = os.path.join(args.models_dir, f"{args.model_pattern}.zip")
            model_files = glob.glob(pattern)
    
    if not model_files:
        print(f"❌ No models found matching pattern: {args.model_pattern}")
        print(f"   Searched in: {args.models_dir}")
        print(f"   Train competitive agents first:")
        print(f"   python examples/train_multi_competitive.py --num-agents 4")
        return
    
    print(f"Found {len(model_files)} competitive agents:")
    for model_file in model_files:
        print(f"  - {os.path.basename(model_file)}")
    
    # Load all models
    agent_models = []
    for model_file in model_files:
        try:
            agent_name = os.path.basename(model_file).replace('.zip', '')
            model, vec_normalize = load_agent(model_file)
            agent_models.append((agent_name, model, vec_normalize))
        except Exception as e:
            print(f"❌ Failed to load {model_file}: {e}")
    
    if not agent_models:
        print("❌ No models could be loaded successfully")
        return
    
    print(f"\n✅ Successfully loaded {len(agent_models)} agents")
    print(f"\n🎮 Starting sequential playback...")
    print(f"   Close the window or press Ctrl+C to stop")
    
    try:
        play_agents_sequentially(agent_models, args.track, args.episodes_per_agent)
        print(f"\n🏁 Playback completed!")
    except KeyboardInterrupt:
        print(f"\n🛑 Playback stopped by user")


if __name__ == '__main__':
    main()