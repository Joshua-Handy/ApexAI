"""
Manual driving demo for MetaDrive custom tracks.

Usage:
  python examples/drive_manual.py --track right_oval

Controls:
  W = throttle, S = brake/reverse
  A = steer left, D = steer right
  B = toggle top-down view, Q = third-person
  R = reset scenario, [ / ] = previous/next seed
"""
import sys
import os
import argparse

# Ensure src is on path
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from environments.single_car_racing import create_racing_environment

def main():
    parser = argparse.ArgumentParser(description="Manual driving demo for MetaDrive custom tracks.")
    parser.add_argument('--track', type=str, default='right_oval', help='Track name (default: right_oval)')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    args = parser.parse_args()

    env = create_racing_environment(args.track, use_render=True, manual_control=True)
    print(f"Manual driving enabled on track: {args.track}")
    print("Controls: W/A/S/D, B (top-down), Q (third-person), R (reset), [ / ] (seed)")

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        while not done:
            # Action is ignored when manual_control=True and use_render=True
            out = env.step([0, 0])
            if len(out) == 4:
                obs, reward, done, info = out
            else:
                obs, reward, terminated, truncated, info = out
                done = terminated or truncated
        print(f"Episode {ep+1} finished. Press R to reset, or close window to exit.")
    env.close()

if __name__ == "__main__":
    main()
