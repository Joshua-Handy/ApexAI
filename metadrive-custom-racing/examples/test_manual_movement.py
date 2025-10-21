#!/usr/bin/env python3
"""
Quick test to manually make cars move with fixed actions
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from environments.multi_agent_oval_right_env import MultiAgentOvalEnv
from metadrive.component.pgblock.first_block import FirstPGBlock

def test_manual_movement():
    """Test with completely manual, forced actions to verify cars can move"""
    
    agent_ids = ["agent0", "agent1", "agent2", "agent3"]
    
    def get_horizontal_lane_position(agent_index: int, total_agents: int, lane_width: float):
        center = (total_agents - 1) / 2.0
        lateral = (agent_index - center) * lane_width
        longitude = -20.0
        return longitude, lateral
    
    racing_grid_configs = {}
    lane_width = 12.0
    for i, agent_id in enumerate(agent_ids):
        longitude, lateral = get_horizontal_lane_position(i, len(agent_ids), lane_width)
        racing_grid_configs[agent_id] = {
            "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, i % len(agent_ids)),
            "spawn_longitude": 4.0,
            "spawn_lateral": 0.0,
        }
    
    # Environment configuration
    env_config = {
        "num_agents": len(agent_ids),
        "traffic_density": 0.0,
        "use_render": True,  # Visual rendering
        "crash_done": False,
        "out_of_road_done": False,
        "on_continuous_line_done": False,
        "on_broken_line_done": False,
        "allow_respawn": False,
        "horizon": 1000,
        "agent_configs": racing_grid_configs,
        "map_config": {
            "lane_num": len(agent_ids),
            "lane_width": lane_width,
        }
    }
    
    print("Creating environment with manual control...")
    env = MultiAgentOvalEnv(env_config)
    
    try:
        print("Starting manual movement test...")
        obs = env.reset()
        if isinstance(obs, tuple):
            obs_dict = obs[0]
        else:
            obs_dict = obs
        print(f"Environment reset. Agents: {list(obs_dict.keys())}")
        
        # Manual actions - FORCE STRONG MOVEMENT
        for step in range(1000):  # Run for 1000 steps
            
            # FIXED ACTIONS - STRONG THROTTLE AND STEERING
            actions = {}
            for agent_id in agent_ids:
                if agent_id in obs_dict:
                    actions[agent_id] = [0.1, 0.8]
            
            print(f"Step {step}: Sending actions {actions}")
            
            # Step environment
            step_result = env.step(actions)
            
            if len(step_result) == 4:
                obs_dict, reward_dict, done_dict, info_dict = step_result
                agent_done = done_dict
            else:
                obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = step_result
                agent_done = {k: terminated_dict.get(k, False) or truncated_dict.get(k, False) 
                            for k in obs_dict.keys()}
            
            # Print positions every 100 steps
            if step % 100 == 0:
                print(f"\n   Step {step}:")
                for agent_id in agent_ids:
                    if agent_id in env.agents:
                        vehicle = env.agents[agent_id]
                        pos = vehicle.position
                        speed = vehicle.speed_km_h
                        print(f"      {agent_id}: pos=({pos[0]:.1f}, {pos[1]:.1f}), speed={speed:.1f}km/h")
                    else:
                        print(f"      {agent_id}: NOT IN env.agents")
            
            # Stop if all agents are done
            if all(agent_done.values()):
                print("All agents done!")
                break
                
        print("Manual movement test completed!")
        
    finally:
        env.close()

if __name__ == "__main__":
    test_manual_movement()
