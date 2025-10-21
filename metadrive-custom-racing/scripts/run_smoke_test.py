import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stable_baselines3 import PPO
from environments.multi_agent_oval_right_env import MultiAgentOvalEnv
from metadrive.component.pgblock.first_block import FirstPGBlock

def main(models_dir='multi_agent_racing_results/final_models', steps=200):
    agent_ids = [f'agent{i}' for i in range(4)]
    models = {}
    for a in agent_ids:
        path = os.path.join(models_dir, f'{a}_final.zip')
        if os.path.exists(path):
            models[a] = PPO.load(path)
        else:
            print(f'Model missing: {path}')
            return 1
    lane_width = 12.0
    racing_grid_configs = {}
    for i, agent_id in enumerate(agent_ids):
        racing_grid_configs[agent_id] = {
            'spawn_lane_index': (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, i % len(agent_ids)),
            'spawn_longitude': 4.0,
            'spawn_lateral': 0.0,
        }
    env_config = {
        'num_agents': len(agent_ids),
        'traffic_density': 0.0,
        'use_render': False,
        'crash_done': True,
        'out_of_road_done': True,
        'on_continuous_line_done': True,
        'on_broken_line_done': True,
        'allow_respawn': False,
        'horizon': 4000,
        'success_reward': 10.0,
        'driving_reward': 2.0,
        'speed_reward': 1.5,
        'use_lateral_reward': True,
        'out_of_road_penalty': 1.0,
        'crash_vehicle_penalty': 2.5,
        'crash_object_penalty': 2.0,
        'map_config': {
            'lane_num': len(agent_ids),
            'lane_width': lane_width,
        },
        'vehicle_config': {
            'show_lidar': True,
            'show_lane_line_detector': True,
            'show_side_detector': True,
            'enable_reverse': False,
            'lidar': {'num_others': 4, 'distance': 50, 'num_lasers': 72},
        },
        'agent_configs': racing_grid_configs,
    }
    env = MultiAgentOvalEnv(env_config)
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs_dict = obs[0]
        else:
            obs_dict = obs
        agent_done = {a: False for a in agent_ids}
        avg_abs_steer = {a: 0.0 for a in agent_ids}
        total_steps = {a: 0 for a in agent_ids}
        active = len(agent_ids)
        for step in range(steps):
            if active == 0:
                break
            actions = {}
            for a in agent_ids:
                if agent_done[a] or a not in obs_dict:
                    actions[a] = [0.0, 0.0]
                    continue
                action, _ = models[a].predict(obs_dict[a], deterministic=True)
                transformed_throttle = 0.3 + ((action[1] + 1.0) / 2.0) * 0.7
                transformed_steering = float(np.clip(action[0], -0.3, 0.3))
                actions[a] = [transformed_steering, float(transformed_throttle)]
                avg_abs_steer[a] += abs(transformed_steering)
                total_steps[a] += 1
            step_result = env.step(actions)
            if len(step_result) == 4:
                obs_dict, reward_dict, done_dict, info_dict = step_result
            else:
                obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = step_result
                done_dict = {k: terminated_dict.get(k, False) or truncated_dict.get(k, False) for k in terminated_dict.keys()}
            for a in agent_ids:
                if not agent_done[a] and (done_dict.get(a, False) or info_dict.get(a, {}).get('crashed', False)):
                    agent_done[a] = True
                    active -= 1
        print('Smoke Test Summary:')
        for a in agent_ids:
            steer_avg = (avg_abs_steer[a] / max(1, total_steps[a]))
            print(f'  {a}: steps={total_steps[a]}, avg|steer|={steer_avg:.3f}, done={agent_done[a]}')
        return 0
    finally:
        env.close()

if __name__ == '__main__':
    sys.exit(main())
