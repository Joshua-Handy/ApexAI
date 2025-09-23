"""Play back a saved PPO model in the MetaDrive environment with rendering.

Usage (PowerShell):
    python examples/play_model.py --model ../results/ppo_custom_speedway.zip --track custom_speedway --episodes 5
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
from environments.single_car_racing import create_racing_environment


def play(model_path: str, track: str = 'custom_speedway', episodes: int = 5):
    """Play back a trained PPO model with optional VecNormalize stats.

    If VecNormalize stats exist, we will wrap the env in a DummyVecEnv and load
    the normalization so observations are processed the same as during training.
    """
    base_env = create_racing_environment(track, use_render=True)
    # Load model
    model = PPO.load(model_path)

    def _unpack_reset(ret):
        # env.reset() may return obs or (obs, info)
        if isinstance(ret, tuple) and len(ret) >= 1:
            return ret[0]
        return ret

    def _unpack_step(ret):
        # env.step() may return (obs, reward, done, info) or
        # (obs, reward, terminated, truncated, info)
        if len(ret) == 4:
            obs, reward, done, info = ret
        elif len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            done = bool(terminated) or bool(truncated)
        else:
            # Unknown format: try best-effort
            obs = ret[0]
            reward = ret[1] if len(ret) > 1 else 0.0
            done = bool(ret[2]) if len(ret) > 2 else False
            info = ret[-1] if len(ret) > 3 else {}
        return obs, reward, done, info

    import numpy as _np

    # Optional: apply VecNormalize stats if present alongside the model
    vecnorm_path = os.path.join(os.path.dirname(model_path), f'vecnorm_{track}.pkl')
    is_vec = False
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        if os.path.exists(vecnorm_path):
            print('Loading VecNormalize stats from', vecnorm_path)
            # Create a vectorized env for normalization to attach
            def _make():
                return create_racing_environment(track, use_render=True)
            vec_env = DummyVecEnv([_make])
            env = VecNormalize.load(vecnorm_path, vec_env)
            # Eval mode: do not update running stats, and don't normalize rewards for display
            env.training = False
            env.norm_reward = False
            is_vec = True
        else:
            env = base_env
    except Exception:
        env = base_env

    # Helper to access the underlying env for rendering frames
    def _render_rgb():
        try:
            if is_vec:
                return env.envs[0].render(mode='rgb_array')
            else:
                return env.render(mode='rgb_array')
        except Exception:
            return None

    # Prepare recording
    do_record = False
    recorder = None
    try:
        import cv2
        do_record = True
    except Exception:
        do_record = False

    frames = [] if do_record else None

    for ep in range(episodes):
        reset_ret = env.reset()
        obs = _unpack_reset(reset_ret)
        total_reward = 0.0
        done = False
        step_i = 0

        while not done:
            # Prepare observation for model.predict
            if is_vec:
                # When using VecEnv, obs is already batched (n_envs=1)
                obs_in = obs
            else:
                # Non-vector env: ensure numpy array
                obs_in = _np.array(obs) if isinstance(obs, (list, tuple)) else obs

            action, _ = model.predict(obs_in, deterministic=True)

            # Step and unpack based on env type
            step_ret = env.step(action)
            if is_vec:
                # VecEnv returns arrays; convert to scalars for logging
                if len(step_ret) == 4:
                    obs, reward, done, info = step_ret
                else:
                    # 5-tuple variant
                    obs, reward, terminated, truncated, info = step_ret
                    done = _np.array(terminated) | _np.array(truncated)
                r0 = float(_np.asarray(reward).reshape(-1)[0])
                d0 = bool(_np.asarray(done).reshape(-1)[0])
                total_reward += r0
                done = d0
            else:
                obs, reward, done, info = _unpack_step(step_ret)
                total_reward += float(reward)

            step_i += 1

            # Recording: grab screen buffer
            if do_record:
                img = _render_rgb()
                if img is not None:
                    try:
                        cv2.putText(img, f'Step: {step_i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        cv2.putText(img, f'Rew: {total_reward:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    except Exception:
                        pass
                    frames.append(img)

        print(f'Episode {ep+1} reward: {total_reward}')

    try:
        env.close()
    except Exception:
        pass

    # Save MP4 and GIF if frames were recorded
    if frames:
        try:
            import imageio
            mp4_path = os.path.join(os.path.dirname(model_path), f'playback_{track}.mp4')
            print('Writing MP4 to', mp4_path)
            imageio.mimwrite(mp4_path, [f[:, :, ::-1] for f in frames], fps=30, format='ffmpeg')
        except Exception as e:
            print('Failed to write MP4 (imageio/ffmpeg missing)', e)

        try:
            import imageio
            gif_path = os.path.join(os.path.dirname(model_path), f'playback_{track}.gif')
            print('Writing GIF to', gif_path)
            imageio.mimsave(gif_path, [f[:, :, ::-1] for f in frames], fps=15)
        except Exception as e:
            print('Failed to write GIF (imageio missing)', e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--track', type=str, default='custom_speedway')
    parser.add_argument('--episodes', type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print('Model not found:', args.model)
        return
    play(args.model, track=args.track, episodes=args.episodes)


if __name__ == '__main__':
    main()
