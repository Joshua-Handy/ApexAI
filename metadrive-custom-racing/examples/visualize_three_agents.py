import os
import sys
import argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from environments.multi_agent_oval_right_env import MultiAgentOvalEnv
from metadrive.component.pgblock.first_block import FirstPGBlock
import pickle

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
        "agent2": {"spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 2), "spawn_longitude": 4.0, "spawn_lateral": 0.0},
    }
    cfg = {
        "num_agents": 3,
        "traffic_density": 0.0,
        "use_render": render,
        "window_size": (1200, 800),  # Larger window for 3 agents
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
        "map_config": {"lane_num": 3, "lane_width": 25.0},  # 3 lanes for 3 agents
        "vehicle_config": {
            "show_lidar": True, 
            "show_lane_line_detector": True, 
            "show_side_detector": True, 
            "enable_reverse": False
        },
        "agent_configs": spawn_cfgs,
    }
    return MultiAgentOvalEnv(cfg)

def _adapt_obs(arr: np.ndarray, dim: int | None) -> np.ndarray:
    """Adapt observation to match model's expected input dimension"""
    try:
        v = np.asarray(arr, dtype=np.float32).reshape(-1)
        if dim is None:
            return v
        if v.shape[0] == dim:
            return v
        if v.shape[0] > dim:
            # Truncate if observation is larger than expected
            return v[:dim]
        # Pad with zeros if observation is smaller than expected
        out = np.zeros((dim,), dtype=np.float32)
        out[:v.shape[0]] = v
        return out
    except Exception as e:
        print(f"Warning: Failed to adapt observation: {e}")
        # Return a safe default observation
        return np.zeros((dim if dim else 100,), dtype=np.float32)

def step_loop(env: MultiAgentOvalEnv, m0: PPO, m1: PPO, m2: PPO, v0, v1, v2, max_steps: int):
    ret = env.reset()
    obs = ret[0] if isinstance(ret, tuple) else ret
    done = {"agent0": False, "agent1": False, "agent2": False}
    total = {"agent0": 0.0, "agent1": 0.0, "agent2": 0.0}
    
    # Expected dims from models
    dims = {}
    for i, model in enumerate([m0, m1, m2]):
        try:
            dims[f"agent{i}"] = int(np.prod(model.observation_space.shape))
        except Exception:
            dims[f"agent{i}"] = None
    
    for i in range(max_steps):
        actions = {}
        for aid, model, vecnorm in [("agent0", m0, v0), ("agent1", m1, v1), ("agent2", m2, v2)]:
            if aid in obs and obs[aid] is not None:
                dim = dims[aid]
                o_in = _adapt_obs(obs[aid], dim)
                
                # Apply VecNormalize if available
                if vecnorm is not None:
                    try:
                        # VecNormalize expects shape (1, obs_dim) for single observation
                        o_normalized = vecnorm.normalize_obs(o_in.reshape(1, -1))
                        o_in = o_normalized.flatten()
                    except Exception as e:
                        print(f"Warning: VecNorm failed for {aid}: {e}")
                
                try:
                    # Use deterministic=False for more natural behavior, or True for consistent testing
                    a, _ = model.predict(o_in, deterministic=False)
                    
                    # Debug: Print actions for first few steps
                    if i < 5:
                        print(f"Step {i}, {aid}: Raw action = {a}")
                    
                    # Check if action is in the expected format
                    if isinstance(a, (list, tuple, np.ndarray)) and len(a) >= 2:
                        # Use the agent's raw actions directly - no artificial modifications
                        steer = float(a[0])  # Use raw steering value
                        throttle = float(a[1])  # Use raw throttle value
                        actions[aid] = [steer, throttle]
                        
                        # Debug: Print processed actions for first few steps
                        if i < 5:
                            print(f"  -> Processed: steer={steer:.3f}, throttle={throttle:.3f}")
                    else:
                        # Fallback for unexpected action format
                        print(f"Warning: Unexpected action format for {aid}: {a}")
                        actions[aid] = [0.0, 0.5]  # Default: no steering, half throttle
                except Exception as e:
                    print(f"Error predicting action for {aid}: {e}")
                    actions[aid] = [0.0, 0.5]  # Safe fallback
            else:
                # Default action when agent not in obs: slight forward movement
                actions[aid] = [0.0, 0.3]
        
        sr = env.step(actions)
        if len(sr) == 4:
            obs, rew, dn, info = sr
        else:
            obs, rew, term, trunc, info = sr
            dn = {k: bool(term.get(k, False) or trunc.get(k, False)) for k in term.keys()}
        
        for aid in ("agent0", "agent1", "agent2"):
            total[aid] += float(rew.get(aid, 0.0)) if isinstance(rew, dict) else 0.0
            crashed = False
            if isinstance(info, dict) and aid in info:
                crashed = bool(info[aid].get('crashed', False))
            if crashed:
                done[aid] = True
        
        if any(done.values()):
            break
    
    return total, done, i + 1

def main():
    p = argparse.ArgumentParser(description="Visualize three trained racing agents competing")
    p.add_argument('--agent1', type=str, default=None, help='Path to agent1 model file')
    p.add_argument('--agent2', type=str, default=None, help='Path to agent2 model file')
    p.add_argument('--agent3', type=str, default=None, help='Path to agent3 model file')
    p.add_argument('--agent1-dir', type=str, default='results_agent_0', help='Directory containing agent1 checkpoints')
    p.add_argument('--agent2-dir', type=str, default='results_agent_1', help='Directory containing agent2 checkpoints')
    p.add_argument('--agent3-dir', type=str, default='results_agent_2', help='Directory containing agent3 checkpoints')
    p.add_argument('--agent1-vecnorm', type=str, default=None, help='Path to agent1 VecNormalize file (.pkl)')
    p.add_argument('--agent2-vecnorm', type=str, default=None, help='Path to agent2 VecNormalize file (.pkl)')
    p.add_argument('--agent3-vecnorm', type=str, default=None, help='Path to agent3 VecNormalize file (.pkl)')
    p.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    p.add_argument('--max-steps', type=int, default=2000, help='Maximum steps per episode')
    p.add_argument('--no-render', action='store_true', help='Disable rendering')
    args = p.parse_args()

    # Load agent models
    a1_path = args.agent1 or latest(args.agent1_dir)
    a2_path = args.agent2 or latest(args.agent2_dir)
    a3_path = args.agent3 or latest(args.agent3_dir)
    
    missing_agents = []
    if not a1_path or not os.path.exists(a1_path):
        missing_agents.append('Agent1')
    if not a2_path or not os.path.exists(a2_path):
        missing_agents.append('Agent2')
    if not a3_path or not os.path.exists(a3_path):
        missing_agents.append('Agent3')
    
    if missing_agents:
        print(f'Missing models for: {", ".join(missing_agents)}')
        return
    
    print(f"Loading models:")
    print(f"  Agent1: {a1_path}")
    print(f"  Agent2: {a2_path}")
    print(f"  Agent3: {a3_path}")
    
    # Load models and VecNormalize files
    models = []
    vecnorms = []
    
    for i, (model_path, vecnorm_path) in enumerate([(a1_path, args.agent1_vecnorm), 
                                                    (a2_path, args.agent2_vecnorm), 
                                                    (a3_path, args.agent3_vecnorm)]):
        model = PPO.load(model_path)
        models.append(model)
        
        # Load VecNormalize if provided
        if vecnorm_path and os.path.exists(vecnorm_path):
            try:
                with open(vecnorm_path, 'rb') as f:
                    vecnorm = pickle.load(f)
                vecnorms.append(vecnorm)
                print(f"  Agent{i+1}: Loaded VecNormalize from {vecnorm_path}")
            except Exception as e:
                print(f"  Agent{i+1}: Failed to load VecNormalize: {e}")
                vecnorms.append(None)
        else:
            vecnorms.append(None)
    
    m0, m1, m2 = models
    v0, v1, v2 = vecnorms
    
    # Debug: Print model information
    print(f"\nModel Information:")
    for i, model in enumerate([m0, m1, m2]):
        obs_space = model.observation_space
        act_space = model.action_space
        has_vecnorm = vecnorms[i] is not None
        print(f"  Agent{i}: Obs shape={obs_space.shape}, Action shape={act_space.shape}")
        print(f"           Obs range=[{obs_space.low.min():.2f}, {obs_space.high.max():.2f}]")
        print(f"           Action range=[{act_space.low.min():.2f}, {act_space.high.max():.2f}]")
        print(f"           VecNormalize: {'✅' if has_vecnorm else '❌'}")
    env = build_env(render=not args.no_render)
    
    try:
        print(f"\n🏁 Starting {args.episodes} episode(s) with 3-agent racing...")
        print("=" * 60)
        
        episode_results = []
        for ep in range(args.episodes):
            totals, done_status, steps = step_loop(env, m0, m1, m2, v0, v1, v2, args.max_steps)
            episode_results.append(totals)
            
            # Determine winner
            winner = max(totals, key=totals.get)
            winner_score = totals[winner]
            
            print(f'Episode {ep+1:2d} ({steps:4d} steps):')
            print(f'  🥇 Winner: {winner} (Score: {winner_score:.2f})')
            print(f'  📊 Scores: Agent0={totals["agent0"]:6.2f} | Agent1={totals["agent1"]:6.2f} | Agent2={totals["agent2"]:6.2f}')
            
            # Show crash status
            crashed_agents = [agent for agent, crashed in done_status.items() if crashed]
            if crashed_agents:
                print(f'  💥 Crashed: {", ".join(crashed_agents)}')
            print()
        
        # Summary statistics
        if args.episodes > 1:
            print("=" * 60)
            print("📈 SUMMARY STATISTICS:")
            
            # Calculate wins
            wins = {"agent0": 0, "agent1": 0, "agent2": 0}
            avg_scores = {"agent0": 0.0, "agent1": 0.0, "agent2": 0.0}
            
            for result in episode_results:
                winner = max(result, key=result.get)
                wins[winner] += 1
                for agent in avg_scores:
                    avg_scores[agent] += result[agent]
            
            for agent in avg_scores:
                avg_scores[agent] /= args.episodes
            
            print(f"Win Counts: Agent0={wins['agent0']} | Agent1={wins['agent1']} | Agent2={wins['agent2']}")
            print(f"Avg Scores: Agent0={avg_scores['agent0']:.2f} | Agent1={avg_scores['agent1']:.2f} | Agent2={avg_scores['agent2']:.2f}")
            
            overall_winner = max(wins, key=wins.get)
            print(f"🏆 Overall Champion: {overall_winner} ({wins[overall_winner]}/{args.episodes} wins)")
            
    finally:
        try:
            env.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()