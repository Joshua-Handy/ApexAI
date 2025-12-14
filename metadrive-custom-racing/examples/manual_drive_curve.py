"""
Manually drive on the custom speedway with AGENT_0V2 REWARD SYSTEM.

Test the proven reward formula that successfully trained agent_0v2:
- +1000 progress, +30 speed, +10 throttle, +20 completion
- Lenient boundaries (out_of_road_done=False)
- Focus on SPEED and SUSTAINED PROGRESS

Controls:
    W/S or Up/Down: Throttle/Brake
    A/D or Left/Right: Steering
    ESC: Quit

Usage:
    python manual_drive_curve.py
"""
import os
import sys

# Add src to path
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from environments.multi_agent_custom_speedway_curve_env import MultiAgentCustomSpeedwayCurveEnv


def main():
    print("\n" + "="*70)
    print("🏁 CUSTOM SPEEDWAY - MANUAL DRIVE (AGENT_0V2 REWARDS)")
    print("="*70)
    print("Controls:")
    print("  W/S or Up/Down: Throttle/Brake")
    print("  A/D or Left/Right: Steering")
    print("  ESC: Quit")
    print("\nReward System: Agent_0v2 proven formula")
    print("  +1000 progress | +30 speed | +10 throttle | +20 completion")
    print("  Lenient boundaries: -5 off-road (no instant death)")
    print("  Focus: SPEED and SUSTAINED PROGRESS")
    print("="*70 + "\n")

    # Create environment with rendering enabled
    config = {
        'num_agents': 1,  # Single agent for manual control
        'use_render': True,
        'manual_control': True,  # Enable keyboard control
        'map_config': {
            'lane_num': 2,
            'lane_width': 8.0,  # Wider lanes - more room for error
        },
        'start_seed': 42,

        # STRICT boundaries for curve-aware training
        'crash_vehicle_done': True,
        'crash_object_done': True,
        'out_of_road_done': True,  # Episode ends if you go off track!

        'horizon': 5000,  # Long horizon for testing
        'vehicle_config': {
            'max_speed_km_h': 120,
        },
        'show_terrain': True,
        'show_sidewalk': True,
    }

    env = MultiAgentCustomSpeedwayCurveEnv(config)

    try:
        obs_dict, info = env.reset()
        print("🏁 Track loaded! Press W to accelerate and start driving.")
        print("   Try different speeds on straights vs curves!\n")

        done = False
        step = 0
        total_reward = 0.0
        last_curve_strength = -1.0

        while not done:
            # In manual mode, the environment handles keyboard input internally
            # We just pass a dummy action
            actions = {agent_id: [0.0, 0.0] for agent_id in obs_dict.keys()}

            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(actions)

            # Accumulate reward for display
            for agent_id, reward in reward_dict.items():
                total_reward += reward

            step += 1

            # Display stats every 50 steps or when curve_strength changes
            if step % 50 == 0:
                for agent_id in obs_dict.keys():
                    info = info_dict.get(agent_id, {})
                    metrics = info.get('metrics', {})
                    vehicle_state = info.get('vehicle_state', {})

                    speed = metrics.get('speed_kmh', 0.0)
                    completion = metrics.get('route_completion', 0.0)
                    curve_strength = metrics.get('curve_strength', 0.0)
                    on_road = vehicle_state.get('on_road', True)

                    # Show curve strength change
                    if abs(curve_strength - last_curve_strength) > 0.01:
                        if curve_strength == 0.0:
                            print(f"\n  🏁 STRAIGHT DETECTED! Full throttle rewarded!")
                        else:
                            sharpness = "SHARP" if curve_strength > 0.7 else "MEDIUM" if curve_strength > 0.4 else "GENTLE"
                            print(f"\n  🔄 CURVE DETECTED! ({sharpness}, strength={curve_strength:.2f}) Brake now!")
                        last_curve_strength = curve_strength

                    road_status = "ON ROAD" if on_road else "⚠️ OFF ROAD"
                    print(f"Step {step}: Speed={speed:.1f}km/h, Progress={completion:.1%}, "
                          f"Curve={curve_strength:.2f}, {road_status}, Reward={total_reward:.1f}")

            # Check if done (will trigger if you go off track!)
            done = terminated_dict.get('__all__', False) or truncated_dict.get('__all__', False)

            if done:
                for agent_id in obs_dict.keys():
                    info = info_dict.get(agent_id, {})
                    if not info.get('vehicle_state', {}).get('on_road', True):
                        print("\n❌ EPISODE ENDED: You went off track!")
                        print("   (This is how curve-aware training prevents track escape exploits)")

        print(f"\n🏁 Drive complete!")
        print(f"   Total steps: {step}")
        print(f"   Total reward: {total_reward:.1f}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Manual drive interrupted by user")
    finally:
        env.close()
        print("Environment closed.\n")


if __name__ == '__main__':
    main()
