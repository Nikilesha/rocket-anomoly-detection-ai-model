# rocket_physics.py

import numpy as np
from rocket_env.rocket_dynamics import (
    compute_acceleration,
    update_velocity_and_altitude,
    update_pitch_and_deviation,
    update_fuel
)

def compute_next_state(
    state: np.ndarray,
    action: np.ndarray,
    rocket,
    physics: dict,
    dt: float = 1.0
) -> np.ndarray:
    """
    Compute the next state of the rocket.

    Parameters
    ----------
    state : np.ndarray
        Current state [x, y, z, altitude, fuel]
    action : np.ndarray
        Action inputs [angle_adjustment, throttle]
    rocket : Rocket
        Rocket object containing dry_mass, fuel_mass, max_thrust, burn_rate, max_angle_change
    physics : dict
        Dictionary containing physics parameters (gravity, air_density, etc.)
    dt : float
        Simulation timestep in seconds

    Returns
    -------
    np.ndarray
        Updated state [x, y, z, altitude, fuel]
    """

    # Unpack state and action
    x, y, z, altitude, fuel = state
    angle_adjust, throttle = action

    # Clamp throttle between 0 and 1
    throttle = np.clip(throttle, 0.0, 1.0)

    # Compute thrust
    thrust = throttle * rocket.max_thrust

    # Compute net acceleration
    mass = rocket.dry_mass + fuel
    gravity = physics.get("gravity", 9.81)
    acceleration = compute_acceleration(thrust, mass, gravity)

    # Update velocity and altitude (placeholder velocity=0.0)
    velocity, altitude = update_velocity_and_altitude(
        velocity=0.0,
        altitude=altitude,
        acceleration=acceleration,
        dt=dt
    )

    # Update fuel
    fuel = update_fuel(fuel, throttle, rocket.burn_rate, dt=dt)

    # Update pitch and deviation (currently placeholder, can extend later)
    pitch, deviation = update_pitch_and_deviation(
        pitch=0.0,
        deviation=0.0,
        angle_adjust=angle_adjust,
        max_angle_change=rocket.max_angle_change
    )

    # Return updated state
    return np.array([x, y, z, altitude, fuel], dtype=np.float32)
