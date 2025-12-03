"""
Debug script to test if rewards are working correctly.
Tests different scenarios and prints actual rewards.
"""
import os
import sys

# Add src to path
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from environments.multi_agent_custom_speedway_env import MultiAgentCustomSpeedwayEnv

def test_rewards():
    """Test reward function with different scenarios."""

    env_config = {
        'num_agents': 2,
        'use_render': False,
        'map_config': {
            'lane_num': 3,
            'lane_width': 8.0,
        },
        'start_seed': 42,
        'crash_vehicle_done': False,
        'crash_object_done': False,
        'out_of_road_done': False,
        'boundary_training_mode': True,
        'horizon': 1500,
    }

    print("\n" + "="*70)
    print("🔍 REWARD SYSTEM DEBUG TEST")
    print("="*70)

    env = MultiAgentCustomSpeedwayEnv(env_config)
    obs = env.reset()

    print("\n✅ Environment created successfully")
    print(f"   Number of agents: {len(env.agents)}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    # Test 1: Do nothing (stay still)
    print("\n" + "="*70)
    print("TEST 1: Do nothing (should get -10 per step for being slow)")
    print("="*70)

    total_reward = 0
    steps = 0
    for i in range(100):
        # Action: [steering, throttle] = [0, 0] = stay still
        actions = {
            'agent0': [0.0, 0.0],
            'agent1': [0.0, 0.0]
        }
        obs, rewards, dones, truncated, infos = env.step(actions)

        if i < 5:  # Print first 5 steps
            agent0_reward = rewards['agent0']
            agent0_info = infos.get('agent0', {})
            speed = agent0_info.get('vehicle_state', {}).get('speed', 0) * 120  # Convert to km/h

            print(f"   Step {i+1}: reward={agent0_reward:+.1f}, speed={speed:.1f} km/h")

            # Print reward components if available
            if 'reward_components' in agent0_info:
                components = agent0_info['reward_components']
                print(f"      Components: {components}")

        total_reward += rewards['agent0']
        steps += 1

        if dones['agent0'] or dones['__all__']:
            break

    avg_reward = total_reward / steps
    print(f"\n   Total reward over {steps} steps: {total_reward:.1f}")
    print(f"   Average per step: {avg_reward:.1f}")
    print(f"   Expected: -10 per step")
    print(f"   ✅ PASS" if -15 < avg_reward < -5 else f"   ❌ FAIL - rewards are wrong!")

    # Test 2: Full throttle forward
    print("\n" + "="*70)
    print("TEST 2: Full throttle forward (should accelerate and get positive rewards)")
    print("="*70)

    env.reset()
    total_reward = 0
    steps = 0
    max_speed = 0

    for i in range(100):
        # Action: [steering, throttle] = [0, 1] = straight full throttle
        actions = {
            'agent0': [0.0, 1.0],
            'agent1': [0.0, 0.0]
        }
        obs, rewards, dones, truncated, infos = env.step(actions)

        agent0_reward = rewards['agent0']
        agent0_info = infos.get('agent0', {})
        speed = agent0_info.get('vehicle_state', {}).get('speed', 0) * 120  # Convert to km/h
        max_speed = max(max_speed, speed)

        if i < 5 or i % 20 == 0:  # Print first 5 and every 20 steps
            print(f"   Step {i+1}: reward={agent0_reward:+.1f}, speed={speed:.1f} km/h")

            if 'reward_components' in agent0_info:
                components = agent0_info['reward_components']
                print(f"      Components: {components}")

        total_reward += rewards['agent0']
        steps += 1

        if dones['agent0'] or dones['__all__']:
            print(f"   ⚠️  Episode ended at step {i+1}")
            break

    avg_reward = total_reward / steps
    print(f"\n   Total reward over {steps} steps: {total_reward:.1f}")
    print(f"   Average per step: {avg_reward:.1f}")
    print(f"   Max speed reached: {max_speed:.1f} km/h")
    print(f"   Expected: Positive rewards once speed > 40 km/h")
    print(f"   ✅ PASS" if avg_reward > 0 and max_speed > 40 else f"   ❌ FAIL - agent not learning to go fast!")

    env.close()

    print("\n" + "="*70)
    print("🎯 DEBUG TEST COMPLETE")
    print("="*70)
    print("\nIf both tests PASS: Reward system is working correctly")
    print("If tests FAIL: There's a bug in the reward calculation")
    print("\n")

if __name__ == '__main__':
    try:
        test_rewards()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
