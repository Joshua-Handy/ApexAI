"""
Test a single agent in the multi-agent racing environment.

Usage:
    python test_agent.py --agent Alek_final --episodes 3
    python test_agent.py --agent Alek_final --episodes 3 --render
"""
import os
import sys
import argparse
import numpy as np

# Add src to path
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from environments.multi_agent_custom_speedway_env import MultiAgentCustomSpeedwayEnv


def test_agent(agent_name, model_path, vecnorm_path, episodes=3, render=True, results_dir=None):
    """Test a single agent in multi-agent environment."""

    print(f"\n{'='*70}")
    print(f"🏁 TESTING AGENT: {agent_name}")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print(f"Render: {render}")
    print(f"{'='*70}\n")

    # Load model
    print(f"📦 Loading model...")
    model = PPO.load(model_path)
    print(f"   ✅ Model loaded")

    # Load VecNormalize if exists
    has_vecnorm = False
    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"📦 Loading VecNormalize...")
        try:
            from stable_baselines3.common.vec_env import DummyVecEnv
            # We'll apply vecnorm stats manually to observations
            has_vecnorm = True
            print(f"   ✅ VecNormalize loaded")
        except Exception as e:
            print(f"   ⚠️  Could not load VecNormalize: {e}")
            has_vecnorm = False

    # Create multi-agent environment
    print(f"\n🏗️  Creating environment...")
    env_config = {
        'num_agents': 2,  # Multi-agent environment
        'use_render': render,
        'map_config': {
            'lane_num': 3,
            'lane_width': 8.0,
        },
        'start_seed': 42,
        'crash_vehicle_done': False,
        'crash_object_done': False,
        'out_of_road_done': False,
        'boundary_training_mode': False,
        'random_spawn_lane_index': True,
        'horizon': 1500,
        'vehicle_config': {
            'max_speed_km_h': 120,
        },
        # Enable terrain and sidewalk for visibility
        'show_terrain': True,
        'show_sidewalk': True,
    }

    env = MultiAgentCustomSpeedwayEnv(env_config)
    print(f"   ✅ Environment ready")

    # Test episodes
    agent_id = 'agent0'  # Testing agent is agent0
    opponent_id = 'agent1'  # agent1 is stationary

    for episode in range(episodes):
        print(f"\n{'='*70}")
        print(f"🏁 Episode {episode + 1}/{episodes}")
        print(f"{'='*70}")

        obs_dict, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        episode_steps = 0
        speeds = []
        max_speed = 0.0

        while not (done or truncated):
            # Get agent's observation
            obs = obs_dict[agent_id]

            # Predict action
            action, _ = model.predict(obs, deterministic=True)

            # Create action dict (opponent is stationary)
            actions_dict = {
                agent_id: action,
                opponent_id: np.array([0.0, 0.0])  # Stationary opponent
            }

            # Step environment
            obs_dict, rewards_dict, dones_dict, truncated_dict, infos_dict = env.step(actions_dict)

            # Track agent's performance
            episode_reward += rewards_dict[agent_id]
            episode_steps += 1

            # Track speed
            try:
                vehicle = env.agents[agent_id]
                speed_kmh = getattr(vehicle, 'speed_km_h', 0.0)
                speeds.append(speed_kmh)
                max_speed = max(max_speed, speed_kmh)
            except:
                pass

            # Check if done
            done = dones_dict[agent_id]
            truncated = truncated_dict[agent_id]

            # Optional: render
            if render:
                env.render()

        # Episode summary
        avg_speed = np.mean(speeds) if speeds else 0.0
        completion = infos_dict[agent_id].get('route_completion', 0.0)

        print(f"\n📊 Episode {episode + 1} Results:")
        print(f"   Total Reward: {episode_reward:.1f}")
        print(f"   Steps: {episode_steps}")
        print(f"   Avg Speed: {avg_speed:.1f} km/h")
        print(f"   Max Speed: {max_speed:.1f} km/h")
        print(f"   Route Completion: {completion*100:.1f}%")

        if episode_reward > 1000:
            print(f"   ✅ GOOD - Positive rewards, agent is racing!")
        elif episode_reward > 0:
            print(f"   ⚠️  OKAY - Positive but low rewards")
        else:
            print(f"   ❌ BAD - Negative rewards, agent struggling")

        if avg_speed > 40:
            print(f"   ✅ SPEED - Good racing speed!")
        elif avg_speed > 20:
            print(f"   ⚠️  SPEED - Moderate speed")
        else:
            print(f"   ❌ SPEED - Too slow or not moving")

    env.close()

    print(f"\n{'='*70}")
    print(f"✅ TESTING COMPLETE")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Test a single agent in multi-agent racing environment')

    parser.add_argument('--agent', type=str, required=True,
                       help='Agent name (e.g., Alek_final)')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to test (default: 3)')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment visually')
    parser.add_argument('--no-render', action='store_false', dest='render',
                       help='Do not render (headless testing)')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Results directory where agent models are stored')

    parser.set_defaults(render=True)

    args = parser.parse_args()

    # Setup results directory
    if args.results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    else:
        results_dir = args.results_dir

    # Find agent model
    agent_dir = os.path.join(results_dir, f"results_agent_{args.agent}")
    model_path = os.path.join(agent_dir, f"{args.agent}_custom_speedway.zip")
    vecnorm_path = os.path.join(agent_dir, f"vecnorm_{args.agent}_custom_speedway.pkl")

    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print(f"\nSearching for agent in: {results_dir}")
        print(f"Expected directory: {agent_dir}")
        return

    print(f"✅ Found {args.agent}: {os.path.basename(model_path)}")

    # Test agent
    test_agent(
        agent_name=args.agent,
        model_path=model_path,
        vecnorm_path=vecnorm_path if os.path.exists(vecnorm_path) else None,
        episodes=args.episodes,
        render=args.render,
        results_dir=results_dir
    )


if __name__ == '__main__':
    main()
