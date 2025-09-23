"""Minimal PPO training + evaluation on custom MetaDrive racing map.

This mirrors the user's simple example but uses our environment factory and
track loader. It trains without rendering, then evaluates with rendering.

Usage (PowerShell):
  python .\examples\ppo_minimal.py --track custom_speedway --timesteps 200000 --episodes 3
"""
import os
import sys
import argparse

# Ensure src on path
ROOT = os.path.dirname(__file__)
SRC = os.path.join(os.path.dirname(ROOT), 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from environments.single_car_racing import create_racing_environment


def make_env(track_name: str, use_render: bool = False):
    def _init():
        env = create_racing_environment(track_name, use_render=use_render)
        return Monitor(env)
    return _init


def _unpack_reset(ret):
    # Support gym reset returning (obs) or (obs, info)
    if isinstance(ret, tuple) and len(ret) >= 1:
        return ret[0]
    return ret


def _unpack_step(ret):
    # Support 4-tuple and 5-tuple step signatures
    if len(ret) == 4:
        obs, reward, done, info = ret
        return obs, reward, bool(done), info
    elif len(ret) == 5:
        obs, reward, terminated, truncated, info = ret
        return obs, reward, bool(terminated) or bool(truncated), info
    else:
        obs = ret[0]
        reward = ret[1] if len(ret) > 1 else 0.0
        done = bool(ret[2]) if len(ret) > 2 else False
        info = ret[-1] if len(ret) > 3 else {}
        return obs, reward, done, info


def train(track: str, timesteps: int, lr: float, n_steps: int, batch_size: int, gamma: float, logdir: str):
    os.makedirs(logdir, exist_ok=True)
    vec_env = DummyVecEnv([make_env(track, use_render=False)])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        batch_size=batch_size,
        learning_rate=lr,
        n_steps=n_steps,
        gamma=gamma,
        tensorboard_log=logdir,
    )
    model.learn(total_timesteps=timesteps)

    out_dir = os.path.join(os.path.dirname(ROOT), 'results')
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f'ppo_min_{track}.zip')
    model.save(model_path)
    print("Model saved to", model_path)
    # Important: close the training envs to properly shut down MetaDrive engine
    try:
        # Close the env attached to the model as well
        if hasattr(model, 'env') and model.env is not None:
            model.env.close()
    except Exception:
        pass
    try:
        vec_env.close()
    except Exception:
        pass
    # Encourage GC to finalize any lingering references
    try:
        import gc, time
        del model
        del vec_env
        gc.collect()
        time.sleep(0.2)
    except Exception:
        pass
    return model_path


def evaluate(model_path: str, track: str, episodes: int):
    import numpy as np
    # Use a single non-vector env for rendering
    env = create_racing_environment(track, use_render=True)
    model = PPO.load(model_path)

    rewards = []
    for ep in range(episodes):
        obs = _unpack_reset(env.reset())
        total = 0.0
        done = False
        while not done:
            obs_in = np.array(obs) if isinstance(obs, (list, tuple)) else obs
            action, _ = model.predict(obs_in, deterministic=True)
            obs, r, done, info = _unpack_step(env.step(action))
            total += float(r)
        rewards.append(total)
        print(f"Episode {ep+1} total reward: {total:.2f}")

    avg = sum(rewards) / max(len(rewards), 1)
    print(f"Average reward over {len(rewards)} episodes: {avg:.2f}")
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=str, default='custom_speedway')
    parser.add_argument('--timesteps', type=int, default=200_000)
    parser.add_argument('--episodes', type=int, default=3, help='Episodes to render during eval')
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--n-steps', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    args = parser.parse_args()

    logdir = os.path.join(os.path.dirname(ROOT), 'results', 'tensorboard')
    model_path = train(
        track=args.track,
        timesteps=args.timesteps,
        lr=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        logdir=logdir,
    )

    evaluate(model_path, track=args.track, episodes=args.episodes)


if __name__ == '__main__':
    main()
