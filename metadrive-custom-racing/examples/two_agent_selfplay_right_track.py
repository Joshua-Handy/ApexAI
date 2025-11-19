import os
import sys
import re
import time
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from metadrive.component.pgblock.first_block import FirstPGBlock
from environments.multi_agent_oval_right_env import MultiAgentOvalEnv


def _unpack_reset(ret):
    if isinstance(ret, tuple) and len(ret) >= 1:
        return ret[0]
    return ret


def _unpack_step(ret):
    if len(ret) == 4:
        return ret[0], ret[1], ret[2], ret[3], None
    else:
        obs, reward, terminated, truncated, info = ret
        done = {k: bool(terminated.get(k, False) or truncated.get(k, False)) for k in terminated.keys()}
        return obs, reward, done, info, (terminated, truncated)


def find_latest_checkpoint(dir_path: str) -> str | None:
    if not os.path.isdir(dir_path):
        return None
    candidates = [f for f in os.listdir(dir_path) if f.endswith('.zip')]
    if not candidates:
        return None
    def steps_in_name(name: str) -> int:
        m = re.search(r"(\d+)[-_]?steps", name)
        return int(m.group(1)) if m else -1
    candidates.sort(key=lambda n: steps_in_name(n))
    return os.path.join(dir_path, candidates[-1])


class ThreeAgentSelfPlayEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, base_env: MultiAgentOvalEnv, agent_ids=("agent0", "agent1", "agent2"), force_barrier: bool = True):
        super().__init__()
        self.base_env = base_env
        self.agent_ids = list(agent_ids)
        self.active_agent = self.agent_ids[0]
        self.opponent_agents = self.agent_ids[1:]  # Now supports multiple opponents
        self.opponent_models = {}  # Dictionary to store multiple opponent models
        self.force_barrier = force_barrier
        # Selective collision system configuration
        self.lane_collision_penalties = {
            0: 1.0,    # Inner lane: Full collision penalty
            1: 0.2,    # Middle lane: Reduced collision penalty (80% reduction)
            2: 1.0     # Outer lane: Full collision penalty
        }
        self.lane_crash_done_override = {
            0: True,   # Inner lane: Crash terminates episode
            1: False,  # Middle lane: Crash does NOT terminate episode
            2: True    # Outer lane: Crash terminates episode
        }
        if hasattr(base_env, 'observation_space') and hasattr(base_env.observation_space, 'spaces'):
            self._base_single_obs_space = list(base_env.observation_space.spaces.values())[0]
            self.action_space = list(base_env.action_space.spaces.values())[0]
        else:
            self._base_single_obs_space = base_env.observation_space
            self.action_space = base_env.action_space
        self.observation_space = self._base_single_obs_space
        self._last_obs_dict = None
        self._ep_len = 0
        self._ep_ret = 0.0
        self.active_obs_dim: int | None = None
        self.opponent_obs_dims = {}  # Store obs dims for each opponent

    def set_active(self, active_agent: str, opponent_models: dict, active_model: PPO | None = None):
        self.active_agent = active_agent
        self.opponent_agents = [aid for aid in self.agent_ids if aid != active_agent]
        self.opponent_models = opponent_models
        # Store observation dimensions for each opponent
        for opp_agent, opp_model in opponent_models.items():
            try:
                self.opponent_obs_dims[opp_agent] = int(np.prod(opp_model.observation_space.shape))
            except Exception:
                self.opponent_obs_dims[opp_agent] = None
        if active_model is not None:
            try:
                self.active_obs_dim = int(np.prod(active_model.observation_space.shape))
                low = float(np.min(active_model.observation_space.low)) if hasattr(active_model.observation_space, 'low') else -0.0
                high = float(np.max(active_model.observation_space.high)) if hasattr(active_model.observation_space, 'high') else 1.0
                self.observation_space = gym.spaces.Box(low=low, high=high, shape=(self.active_obs_dim,), dtype=np.float32)
            except Exception:
                self.active_obs_dim = None
                self.observation_space = self._base_single_obs_space

    def _adapt_obs(self, obs: np.ndarray, dim: int | None) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if dim is None:
            return arr
        if arr.shape[0] == dim:
            return arr
        if arr.shape[0] > dim:
            return arr[:dim]
        out = np.zeros((dim,), dtype=np.float32)
        out[:arr.shape[0]] = arr
        return out

    def reset(self, **kwargs):
        od = _unpack_reset(self.base_env.reset(**kwargs))
        if self.force_barrier:
            try:
                any_agent = next(iter(self.base_env.agents.values())) if getattr(self.base_env, 'agents', None) else None
                if any_agent is not None:
                    lane = getattr(any_agent, 'lane', None)
                    pos = getattr(any_agent, 'position', None)
                    if lane is not None and pos is not None:
                        s, _ = lane.local_coordinates(pos)
                        s_target = float(s + 80.0)
                        om = getattr(self.base_env.engine, 'object_manager', None)
                        if om is not None:
                            om.barrier_scene(lane, s_target)
            except Exception:
                pass
        self._last_obs_dict = od
        self._ep_len = 0
        self._ep_ret = 0.0
        obs = od[self.active_agent]
        obs = self._adapt_obs(obs, self.active_obs_dim)
        return obs, {}

    def _get_agent_lane(self, agent_id: str) -> int:
        """Get the current lane index of an agent"""
        try:
            if hasattr(self.base_env, 'agents') and agent_id in self.base_env.agents:
                agent = self.base_env.agents[agent_id]
                if hasattr(agent, 'lane') and agent.lane is not None:
                    # Get lane index from the lane object
                    lane = agent.lane
                    if hasattr(lane, 'index'):
                        return int(lane.index[-1]) if isinstance(lane.index, (list, tuple)) else int(lane.index)
                    # Fallback: try to determine lane from position
                    if hasattr(agent, 'position'):
                        # Estimate lane based on lateral position
                        pos = agent.position
                        if hasattr(lane, 'local_coordinates'):
                            _, lateral = lane.local_coordinates(pos)
                            # Approximate lane based on lateral position (assuming lane_width = 25.0)
                            return max(0, min(2, int((lateral + 37.5) / 25.0)))
        except Exception:
            pass
        return 1  # Default to middle lane if detection fails

    def _apply_selective_collision_penalties(self, reward: float, done: bool, info: dict, agent_id: str) -> tuple:
        """Apply selective collision penalties based on lane position"""
        try:
            # Check if this was a collision-related penalty
            crashed = info.get('crash', False) or info.get('crash_vehicle', False) or info.get('crash_object', False)
            
            if crashed:
                current_lane = self._get_agent_lane(agent_id)
                lane_penalty_multiplier = self.lane_collision_penalties.get(current_lane, 1.0)
                override_crash_done = self.lane_crash_done_override.get(current_lane, True)
                
                # If this was a crash penalty, modify it based on lane
                if reward < -10.0:  # Likely a crash penalty (crash_vehicle_penalty=15.0, crash_object_penalty=20.0)
                    original_penalty = reward
                    modified_penalty = original_penalty * lane_penalty_multiplier
                    reward = modified_penalty
                    
                    # Override done status based on lane rules
                    if not override_crash_done:
                        done = False
                    
                    # Add lane info to the info dict
                    info['selective_collision'] = {
                        'lane': current_lane,
                        'original_penalty': original_penalty,
                        'modified_penalty': modified_penalty,
                        'penalty_multiplier': lane_penalty_multiplier,
                        'crash_done_override': override_crash_done
                    }
                    
        except Exception as e:
            # If lane detection fails, use default behavior
            pass
            
        return reward, done, info

    def step(self, action):
        actions = {}
        if isinstance(action, (list, tuple, np.ndarray)) and len(action) >= 2:
            a0 = [float(np.clip(action[0], -1.0, 1.0)), float(np.clip(action[1], 0.0, 1.0))]
        else:
            a0 = action
        actions[self.active_agent] = a0
        
        # Handle multiple opponent actions
        for opp_agent in self.opponent_agents:
            if (opp_agent in self.opponent_models and 
                self.opponent_models[opp_agent] is not None and 
                self._last_obs_dict is not None and 
                opp_agent in self._last_obs_dict):
                opp_obs = self._last_obs_dict[opp_agent]
                opp_obs = self._adapt_obs(opp_obs, self.opponent_obs_dims[opp_agent])
                opp_action, _ = self.opponent_models[opp_agent].predict(opp_obs, deterministic=False)
                if isinstance(opp_action, (list, tuple, np.ndarray)) and len(opp_action) >= 2:
                    opp_a = [float(np.clip(opp_action[0], -1.0, 1.0)), float(np.clip(opp_action[1], 0.0, 1.0))]
                else:
                    opp_a = opp_action
                actions[opp_agent] = opp_a
            else:
                actions[opp_agent] = [0.0, 0.0]
        step_ret = self.base_env.step(actions)
        obs_dict, rew_dict, done_dict, info_dict, td = _unpack_step(step_ret)
        self._last_obs_dict = obs_dict
        r = float(rew_dict.get(self.active_agent, 0.0))
        done = bool(done_dict.get(self.active_agent, False)) or bool(done_dict.get("__all__", False))
        info = info_dict.get(self.active_agent, {}) if isinstance(info_dict, dict) else {}
        
        # Apply selective collision penalties
        r, done, info = self._apply_selective_collision_penalties(r, done, info, self.active_agent)
        
        self._ep_len += 1
        self._ep_ret += r
        if done:
            info = dict(info)
            info.setdefault('episode', {})
            info['episode']['r'] = self._ep_ret
            info['episode']['l'] = self._ep_len
        next_obs = obs_dict[self.active_agent]
        next_obs = self._adapt_obs(next_obs, self.active_obs_dim)
        return next_obs, r, done, done, info

    def render(self):
        try:
            return self.base_env.render(mode='rgb_array')
        except Exception:
            return None

    def close(self):
        try:
            self.base_env.close()
        except Exception:
            pass


def build_env(render: bool = True) -> MultiAgentOvalEnv:
    agent_ids = ["agent0", "agent1", "agent2"]
    spawn_cfgs = {}
    for i, aid in enumerate(agent_ids):
        spawn_cfgs[aid] = {
            "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, i),  # Each agent gets their own lane (0, 1, 2)
            "spawn_longitude": 4.0,  # All agents start at same longitudinal position
            "spawn_lateral": 0.0,
        }
    cfg = {
        "num_agents": 3,
        "traffic_density": 0.0,
        "use_render": render,
        "window_size": (800, 600),
        "crash_done": True,
        "out_of_road_done": True,
        "allow_respawn": False,
        "horizon": 3000,
        "success_reward": 20.0,
        "driving_reward": 2.0,
        "speed_reward": 1.0,
        "out_of_road_penalty": 5.0,
        "crash_vehicle_penalty": 15.0,
        "crash_object_penalty": 20.0,
        "accident_prob": 0.0,
        "static_traffic_object": False,
        "map_config": {
            "lane_num": 3,  # Increased from 2 to 3 lanes
            "lane_width": 25.0,  # Slightly reduced width to fit 3 lanes better
        },
        "vehicle_config": {
            "show_lidar": False,
            "show_lane_line_detector": False,
            "show_side_detector": False,
            "enable_reverse": False,
        },
        "agent_configs": spawn_cfgs,
    }
    return MultiAgentOvalEnv(cfg)


def continue_training(agent1_dir: str, agent2_dir: str, agent3_dir: str, timesteps_per_round: int = 200_000, rounds: int = 3,
                      render: bool = True, results_dir: str = "self_play_three_agents_results"):
    os.makedirs(results_dir, exist_ok=True)
    m1_path = find_latest_checkpoint(agent1_dir)
    m2_path = find_latest_checkpoint(agent2_dir)
    m3_path = find_latest_checkpoint(agent3_dir)
    if m1_path is None or m2_path is None or m3_path is None:
        raise FileNotFoundError(f"Missing checkpoints in {agent1_dir}, {agent2_dir}, or {agent3_dir}")
    base_env = build_env(render=render)
    wrapper = ThreeAgentSelfPlayEnv(base_env)
    model1 = PPO.load(m1_path)
    model2 = PPO.load(m2_path)
    model3 = PPO.load(m3_path)
    ckpt1 = CheckpointCallback(save_freq=50_000, save_path=os.path.join(results_dir, 'agent1'), name_prefix='agent1')
    ckpt2 = CheckpointCallback(save_freq=50_000, save_path=os.path.join(results_dir, 'agent2'), name_prefix='agent2')
    ckpt3 = CheckpointCallback(save_freq=50_000, save_path=os.path.join(results_dir, 'agent3'), name_prefix='agent3')
    for r in range(rounds):
        # Train agent0 against agent1 and agent2
        wrapper.set_active("agent0", opponent_models={"agent1": model2, "agent2": model3}, active_model=model1)
        model1.set_env(wrapper)
        model1.learn(total_timesteps=timesteps_per_round, reset_num_timesteps=False, callback=ckpt1)
        a1_out = os.path.join(results_dir, f"agent1_round{r+1}.zip")
        model1.save(a1_out)
        
        # Train agent1 against agent0 and agent2
        wrapper.set_active("agent1", opponent_models={"agent0": model1, "agent2": model3}, active_model=model2)
        model2.set_env(wrapper)
        model2.learn(total_timesteps=timesteps_per_round, reset_num_timesteps=False, callback=ckpt2)
        a2_out = os.path.join(results_dir, f"agent2_round{r+1}.zip")
        model2.save(a2_out)
        
        # Train agent2 against agent0 and agent1
        wrapper.set_active("agent2", opponent_models={"agent0": model1, "agent1": model2}, active_model=model3)
        model3.set_env(wrapper)
        model3.learn(total_timesteps=timesteps_per_round, reset_num_timesteps=False, callback=ckpt3)
        a3_out = os.path.join(results_dir, f"agent3_round{r+1}.zip")
        model3.save(a3_out)
    final1 = os.path.join(results_dir, "agent1_final.zip")
    final2 = os.path.join(results_dir, "agent2_final.zip")
    final3 = os.path.join(results_dir, "agent3_final.zip")
    model1.save(final1)
    model2.save(final2)
    model3.save(final3)
    try:
        wrapper.close()
    except Exception:
        pass
    return final1, final2, final3


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--agent1-dir', type=str, default='results_agent_1')
    p.add_argument('--agent2-dir', type=str, default='results_agent_2')
    p.add_argument('--agent3-dir', type=str, default='results_agent_3')
    p.add_argument('--timesteps', type=int, default=200_000)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--no-render', action='store_true')
    p.add_argument('--results-dir', type=str, default='self_play_three_agents_results')
    args = p.parse_args()
    f1, f2, f3 = continue_training(
        agent1_dir=args.agent1_dir,
        agent2_dir=args.agent2_dir,
        agent3_dir=args.agent3_dir,
        timesteps_per_round=args.timesteps,
        rounds=args.rounds,
        render=not args.no_render,
        results_dir=args.results_dir,
    )
    print('Saved:', f1)
    print('Saved:', f2)
    print('Saved:', f3)


if __name__ == '__main__':
    main()
