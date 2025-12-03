"""
Manually drive on the custom speedway track.

Controls:
    W/Up Arrow    - Accelerate
    S/Down Arrow  - Brake/Reverse
    A/Left Arrow  - Steer Left
    D/Right Arrow - Steer Right
    R             - Reset
    ESC           - Quit

Usage:
    python examples\manual_drive.py
"""
import os
import sys

# Add src to path
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from environments.multi_agent_custom_speedway_env import MultiAgentCustomSpeedwayEnv


def main():
    print("\n" + "="*70)
    print("🏎️  MANUAL DRIVE - Custom Speedway")
    print("="*70)
    print("\nControls:")
    print("  W/↑  - Accelerate")
    print("  S/↓  - Brake/Reverse")
    print("  A/←  - Steer Left")
    print("  D/→  - Steer Right")
    print("  R    - Reset")
    print("  ESC  - Quit")
    print("="*70 + "\n")

    # Create environment with manual control enabled
    env_config = {
        'num_agents': 1,  # Just you
        'use_render': True,
        'manual_control': True,  # Enable keyboard control!
        'map_config': {
            'lane_num': 3,
            'lane_width': 8.0,
        },
        'start_seed': 42,
        'crash_vehicle_done': False,
        'crash_object_done': False,
        'out_of_road_done': False,
        'horizon': 10000,  # Long time to drive around
        'vehicle_config': {
            'max_speed_km_h': 120,
        }
    }

    env = MultiAgentCustomSpeedwayEnv(env_config)

    print("🏁 Starting manual drive... Press ESC to quit.\n")

    try:
        # Reset environment
        obs_dict, _ = env.reset()

        done = False
        step = 0
        total_reward = 0.0

        while not done:
            step += 1

            # MetaDrive handles input automatically in manual_control mode
            # We just need to step with dummy actions
            actions = {agent_id: [0, 0] for agent_id in obs_dict.keys()}

            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(actions)

            # Get reward and info for the player
            agent_id = list(reward_dict.keys())[0]
            reward = reward_dict[agent_id]
            total_reward += reward

            done = terminated_dict.get('__all__', False) or truncated_dict.get('__all__', False)

            # Print stats every 100 steps
            if step % 100 == 0:
                info = info_dict.get(agent_id, {})
                vehicle_state = info.get('vehicle_state', {})
                speed = vehicle_state.get('speed', 0) * 120.0  # Convert to km/h
                print(f"Step {step:5d} | Speed: {speed:5.1f} km/h | Reward: {total_reward:7.1f}")

        print(f"\n{'='*70}")
        print(f"🏁 Drive Complete!")
        print(f"Total Steps: {step}")
        print(f"Total Reward: {total_reward:.1f}")
        print(f"{'='*70}\n")

    except KeyboardInterrupt:
        print("\n\n🛑 Manual drive stopped by user.\n")

    finally:
        env.close()


if __name__ == '__main__':
    main()
