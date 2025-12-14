"""
Filter demonstrations to keep only the best performing episodes.
"""
# Fix Windows console encoding
import sys
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

import pickle
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description='Filter demonstrations by performance')
    parser.add_argument('--input', type=str, required=True, help='Input demonstrations file')
    parser.add_argument('--output', type=str, required=True, help='Output filtered demonstrations file')
    parser.add_argument('--top-percent', type=float, default=30.0, help='Keep top X percent of episodes (default: 30)')
    parser.add_argument('--min-reward', type=float, default=None, help='Minimum reward threshold (optional)')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("🔍 FILTERING BEST DEMONSTRATIONS")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Keep top: {args.top_percent}% of episodes")
    if args.min_reward:
        print(f"Minimum reward: {args.min_reward}")
    print("="*70 + "\n")

    # Load demonstrations
    print("Loading demonstrations...")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    episodes = data['episodes']
    total_episodes = len(episodes)

    print(f"Loaded {total_episodes} episodes\n")

    # Analyze episodes
    print("Analyzing episode performance...")
    episode_scores = []
    for i, ep in enumerate(episodes):
        reward = ep['reward']
        steps = ep['steps']
        success = ep['success']

        # Score based on reward and completion
        score = reward
        if success:
            score += 100  # Bonus for completing

        episode_scores.append({
            'index': i,
            'reward': reward,
            'steps': steps,
            'success': success,
            'score': score
        })

    # Sort by score (descending)
    episode_scores.sort(key=lambda x: x['score'], reverse=True)

    # Print top 10
    print("\nTop 10 episodes:")
    print("-" * 70)
    for i, ep in enumerate(episode_scores[:10]):
        status = "✅ SUCCESS" if ep['success'] else "❌ FAILED"
        print(f"{i+1:2d}. Episode {ep['index']:3d} | Reward: {ep['reward']:7.1f} | Steps: {ep['steps']:4d} | {status}")

    print("\nBottom 10 episodes:")
    print("-" * 70)
    for i, ep in enumerate(episode_scores[-10:]):
        status = "✅ SUCCESS" if ep['success'] else "❌ FAILED"
        print(f"{i+1:2d}. Episode {ep['index']:3d} | Reward: {ep['reward']:7.1f} | Steps: {ep['steps']:4d} | {status}")

    # Filter episodes
    keep_count = int(total_episodes * args.top_percent / 100)
    keep_count = max(1, keep_count)  # Keep at least 1

    print(f"\n{'='*70}")
    print(f"Filtering: Keeping top {keep_count}/{total_episodes} episodes ({args.top_percent}%)")
    print(f"{'='*70}\n")

    # Get indices to keep
    keep_indices = [ep['index'] for ep in episode_scores[:keep_count]]

    # Filter episodes
    filtered_episodes = [episodes[i] for i in keep_indices]

    # Recalculate statistics
    filtered_observations = []
    filtered_actions = []
    total_steps = 0
    success_count = 0

    for ep in filtered_episodes:
        filtered_observations.extend(ep['observations'])
        filtered_actions.extend(ep['actions'])
        total_steps += ep['steps']
        if ep['success']:
            success_count += 1

    # Create filtered dataset
    filtered_data = {
        'observations': filtered_observations,
        'actions': filtered_actions,
        'episodes': filtered_episodes,
        'total_steps': total_steps,
        'num_episodes': len(filtered_episodes),
        'observation_shape': filtered_observations[0].shape,
        'action_shape': filtered_actions[0].shape,
        'success_rate': success_count / len(filtered_episodes) * 100
    }

    # Save filtered data
    with open(args.output, 'wb') as f:
        pickle.dump(filtered_data, f)

    print("✅ Filtered demonstrations saved!")
    print(f"\nFiltered dataset statistics:")
    print(f"  Episodes: {len(filtered_episodes)} (was {total_episodes})")
    print(f"  Total steps: {total_steps}")
    print(f"  Success rate: {filtered_data['success_rate']:.1f}%")
    print(f"  Avg reward: {np.mean([ep['reward'] for ep in filtered_episodes]):.1f}")
    print(f"  Avg steps: {np.mean([ep['steps'] for ep in filtered_episodes]):.1f}")
    print(f"\nFile saved: {args.output}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
