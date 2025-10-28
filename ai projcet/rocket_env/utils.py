# rocket_utils.py

import numpy as np
from gymnasium import spaces

def sample_random_action(action_space: spaces.Box) -> np.ndarray:
    """
    Sample a random action within the action space.

    Parameters
    ----------
    action_space : gymnasium.spaces.Box
        Gymnasium action space object

    Returns
    -------
    np.ndarray
        Random action within valid bounds
    """
    return action_space.sample()


def compute_reward(state: np.ndarray, mission: dict) -> float:
    """
    Compute a simplified reward based on altitude and fuel efficiency.

    Parameters
    ----------
    state : np.ndarray
        Rocket state [x, y, z, altitude, fuel]
    mission : dict
        Mission configuration including reward weights

    Returns
    -------
    float
        Reward value
    """
    altitude = state[3]       # altitude is at index 3
    fuel_remaining = state[4] # fuel is at index 4

    altitude_weight = mission.get("reward_shaping", {}).get("altitude_reward_weight", 1.0)
    fuel_weight = mission.get("reward_shaping", {}).get("fuel_efficiency_weight", 0.1)

    reward = altitude * altitude_weight
    reward += fuel_remaining * fuel_weight
    return reward


def check_done(state: np.ndarray, steps: int, mission: dict) -> bool:
    """
    Check if the episode should terminate.

    Parameters
    ----------
    state : np.ndarray
        Rocket state [x, y, z, altitude, fuel]
    steps : int
        Current timestep
    mission : dict
        Mission configuration including max_duration

    Returns
    -------
    bool
        True if episode should terminate, False otherwise
    """
    altitude = state[3]
    fuel_remaining = state[4]
    max_duration = mission.get("max_duration", 1000)

    if altitude <= 0 or fuel_remaining <= 0 or steps >= max_duration:
        return True
    return False
