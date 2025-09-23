"""
Simple custom agent for MetaDrive example.
Provides a policy class with a get_action(obs) method and an SB3 wrapper.
"""
import numpy as np
from typing import Any, Optional


class MyAgent:
    """Improved heuristic agent using lidar for steering and a throttle controller.

    Expected observation layout (MetaDrive default vector obs) varies; this agent
    attempts to read lidar slices if present. Lidar typically appears as a tail
    section of the observation vector.
    """

    def __init__(self, target_speed_kmh: float = 80.0, min_speed_kmh: float = 5.0):
        self.target_speed = target_speed_kmh
        self.min_speed = min_speed_kmh
        self.prev_steering = 0.0
        self.stall_counter = 0

    def _extract_lidar(self, obs: Any) -> Optional[np.ndarray]:
        """Try to extract lidar array from observation vector.

        Returns None if lidar not found.
        """
        if obs is None:
            return None
        # Heuristic: lidar is often the trailing N values and length is >= 20
        if len(obs) >= 30:
            # Assume last 24 values are lidar distances
            lidar = np.array(obs[-24:], dtype=float)
            # Basic sanity check
            if np.all(lidar >= 0):
                return lidar
        return None

    def _lidar_steering(self, lidar: np.ndarray) -> float:
        """Compute steering from lidar by dividing into left/center/right sectors."""
        n = len(lidar)
        left = np.mean(lidar[: n // 3])
        center = np.mean(lidar[n // 3 : 2 * n // 3])
        right = np.mean(lidar[2 * n // 3 :])

        # Steer away from the closest side: if left space > right space, steer left (<0)
        steer = 0.0
        if center < max(left, right) * 0.7:
            # Obstacle ahead, choose direction with more space
            steer = -0.8 if left > right else 0.8
        else:
            # Gentle centering
            steer = np.clip((right - left) * 0.2, -0.6, 0.6)

        return float(steer)

    def get_action(self, obs: Any) -> np.ndarray:
        # Speed at obs[0] (m/s) when present
        speed = float(obs[0]) if len(obs) > 0 else 0.0
        speed_kmh = speed * 3.6

        # Throttle controller: proportional with gentle scaling and anti-stall
        speed_error = self.target_speed - speed_kmh
        throttle = 0.02 * speed_error

        # Prevent tiny oscillations
        if abs(throttle) < 0.05:
            throttle = 0.0

        # Strong braking if far above target
        if speed_kmh > self.target_speed + 15:
            throttle = -0.6

        # Anti-stall: if nearly zero speed but throttle small, increase throttle
        if speed_kmh < self.min_speed and throttle <= 0.1:
            self.stall_counter += 1
            if self.stall_counter > 10:
                throttle = 0.8
        else:
            self.stall_counter = 0

        # Steering: prefer lidar when available
        lidar = self._extract_lidar(obs)
        if lidar is not None:
            steering = self._lidar_steering(lidar)
        else:
            # Fallback: try to use lane offset if available at obs[5]
            steering = 0.0
            if len(obs) > 5:
                try:
                    lane_offset = float(obs[5])
                    steering = np.clip(-lane_offset * 0.6, -1.0, 1.0)
                except Exception:
                    steering = 0.0

        # Smooth steering
        steering = 0.7 * self.prev_steering + 0.3 * steering
        self.prev_steering = steering

        # Clip outputs
        throttle = float(np.clip(throttle, -1.0, 1.0))
        steering = float(np.clip(steering, -1.0, 1.0))

        return np.array([throttle, steering], dtype=float)



# Minimal SB3 wrapper: provides predict(observation)
class SB3PolicyWrapper:
    def __init__(self, agent: MyAgent):
        self.agent = agent

    def predict(self, observation, deterministic=True):
        action = self.agent.get_action(observation)
        return action, None
