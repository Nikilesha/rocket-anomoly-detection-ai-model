# rocket_dynamics.py

def compute_acceleration(thrust: float, mass: float, gravity: float = 9.81) -> float:
    """
    Compute net acceleration of the rocket.
    
    Formula:
        a = F/m - g

    Parameters
    ----------
    thrust : float
        Thrust force applied by the engine (N)
    mass : float
        Current mass of the rocket (kg)
    gravity : float
        Gravitational acceleration (m/s^2)

    Returns
    -------
    float
        Net acceleration (m/s^2)
    """
    if mass <= 0:
        raise ValueError("Mass must be positive to compute acceleration.")
    return thrust / mass - gravity


def update_velocity_and_altitude(
    velocity: float, altitude: float, acceleration: float, dt: float = 0.1
) -> tuple[float, float]:
    """
    Update velocity and altitude based on acceleration using simple Euler integration.

    Parameters
    ----------
    velocity : float
        Current velocity (m/s)
    altitude : float
        Current altitude (m)
    acceleration : float
        Current acceleration (m/s^2)
    dt : float
        Time step (s)

    Returns
    -------
    tuple
        Updated velocity, altitude
    """
    velocity += acceleration * dt
    altitude += velocity * dt
    return velocity, max(altitude, 0.0)  # Prevent negative altitude


def update_pitch_and_deviation(
    pitch: float, deviation: float, angle_adjust: float, max_angle_change: float
) -> tuple[float, float]:
    """
    Update the rocket's pitch and lateral deviation.

    Parameters
    ----------
    pitch : float
        Current pitch angle (degrees)
    deviation : float
        Current lateral deviation
    angle_adjust : float
        Angle adjustment input
    max_angle_change : float
        Maximum allowed pitch change per step

    Returns
    -------
    tuple
        Updated pitch, deviation
    """
    # Clamp the angle adjustment
    angle_adjust = max(-max_angle_change, min(angle_adjust, max_angle_change))
    pitch += angle_adjust

    # Placeholder for deviation (can implement wind or perturbations later)
    deviation += 0.0
    return pitch, deviation


def update_fuel(fuel: float, throttle: float, burn_rate: float, dt: float = 1.0) -> float:
    """
    Update remaining fuel based on throttle and burn rate.

    Parameters
    ----------
    fuel : float
        Current fuel mass (kg)
    throttle : float
        Throttle value between 0 and 1
    burn_rate : float
        Maximum fuel consumption rate (kg/s)
    dt : float
        Time step (s)

    Returns
    -------
    float
        Remaining fuel (kg, >= 0)
    """
    throttle = max(0.0, min(throttle, 1.0))  # Clamp throttle
    fuel -= throttle * burn_rate * dt
    return max(fuel, 0.0)
