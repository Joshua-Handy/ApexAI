import os
import sys
import argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stable_baselines3 import PPO
from environments.multi_agent_oval_right_env import MultiAgentOvalEnv
from metadrive.component.pgblock.first_block import FirstPGBlock

def latest(dir_path: str) -> str | None:
    if not os.path.isdir(dir_path):
        return None
    c = [f for f in os.listdir(dir_path) if f.endswith('.zip')]
    if not c:
        return None
    c.sort(key=lambda n: os.path.getmtime(os.path.join(dir_path, n)))
    return os.path.join(dir_path, c[-1])

def build_env(render: bool) -> MultiAgentOvalEnv:
    spawn_cfgs = {
        "agent0": {"spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 0), "spawn_longitude": 4.0, "spawn_lateral": 0.0},
        "agent1": {"spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 1), "spawn_longitude": 4.0, "spawn_lateral": 0.0},
    }
    cfg = {
        "num_agents": 2,
        "traffic_density": 0.0,
        "use_render": render,
        "window_size": (800, 600),
        "crash_done": True,
        "out_of_road_done": False,
        "on_continuous_line_done": False,
        "on_broken_line_done": False,
        "allow_respawn": False,
        "horizon": 100000,
        "crash_only_done": True,
        "success_reward": 10.0,
        "driving_reward": 2.0,
        "speed_reward": 1.0,
        "out_of_road_penalty": 5.0,
        "crash_vehicle_penalty": 10.0,
        "crash_object_penalty": 10.0,
        "accident_prob": 0.0,
        "static_traffic_object": False,
        "map_config": {"lane_num": 2, "lane_width": 30.0},
        "vehicle_config": {"show_lidar": True, "show_lane_line_detector": True, "show_side_detector": True, "enable_reverse": False},
        "agent_configs": spawn_cfgs,
    }
    return MultiAgentOvalEnv(cfg)

def _adapt_obs(arr: np.ndarray, dim: int | None) -> np.ndarray:
    v = np.asarray(arr, dtype=np.float32).reshape(-1)
    if dim is None:
        return v
    if v.shape[0] == dim:
        return v
    if v.shape[0] > dim:
        return v[:dim]
    out = np.zeros((dim,), dtype=np.float32)
    out[:v.shape[0]] = v
    return out

def step_loop(env: MultiAgentOvalEnv, m0: PPO, m1: PPO, max_steps: int):
    ret = env.reset()
    obs = ret[0] if isinstance(ret, tuple) else ret
    done = {"agent0": False, "agent1": False}
    total = {"agent0": 0.0, "agent1": 0.0}
    # Expected dims from models
    try:
        dim0 = int(np.prod(m0.observation_space.shape))
    except Exception:
        dim0 = None
    try:
        dim1 = int(np.prod(m1.observation_space.shape))
    except Exception:
        dim1 = None
    for i in range(max_steps):
        actions = {}
        for aid, model in (("agent0", m0), ("agent1", m1)):
            if aid in obs:
                dim = dim0 if aid == "agent0" else dim1
                o_in = _adapt_obs(obs[aid], dim)
                a, _ = model.predict(o_in, deterministic=True)
                steer = float(np.clip(a[0], -0.3, 0.3))
                thr = 0.3 + ((float(a[1]) + 1.0) / 2.0) * 0.7
                thr = float(np.clip(thr, 0.3, 1.0))
                actions[aid] = [steer, thr]
            else:
                actions[aid] = [0.0, 0.0]
        sr = env.step(actions)
        if len(sr) == 4:
            obs, rew, dn, info = sr
        else:
            obs, rew, term, trunc, info = sr
            dn = {k: bool(term.get(k, False) or trunc.get(k, False)) for k in term.keys()}
        for aid in ("agent0", "agent1"):
            total[aid] += float(rew.get(aid, 0.0)) if isinstance(rew, dict) else 0.0
            crashed = False
            if isinstance(info, dict) and aid in info:
                crashed = bool(info[aid].get('crashed', False))
            if crashed:
                done[aid] = True
        if any(done.values()):
            break
    return total

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agent1', type=str, default=None)
    p.add_argument('--agent2', type=str, default=None)
    p.add_argument('--agent1-dir', type=str, default='results_agent_1')
    p.add_argument('--agent2-dir', type=str, default='results_agent_2')
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=2000)
    p.add_argument('--no-render', action='store_true')
    args = p.parse_args()

    a1_path = args.agent1 or latest(args.agent1_dir)
    a2_path = args.agent2 or latest(args.agent2_dir)
    if not a1_path or not os.path.exists(a1_path):
        print('Agent1 model not found')
        return
    if not a2_path or not os.path.exists(a2_path):
        print('Agent2 model not found')
        return
    m0 = PPO.load(a1_path)
    m1 = PPO.load(a2_path)
    env = build_env(render=not args.no_render)
    try:
        for ep in range(args.episodes):
            totals = step_loop(env, m0, m1, args.max_steps)
            print(f'Episode {ep+1} totals: agent0={totals["agent0"]:.2f} agent1={totals["agent1"]:.2f}')
    finally:
        try:
            env.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()
