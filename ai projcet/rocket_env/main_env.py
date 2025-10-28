import gymnasium as gym
from gymnasium import spaces
import numpy as np

GRAVITY = 9.81  # m/s²


class Rocket:
    def __init__(self, dry_mass, fuel_mass, max_thrust, burn_rate):
        self.dry_mass = dry_mass
        self.initial_fuel = fuel_mass
        self.max_thrust = max_thrust
        self.burn_rate = burn_rate
        self.reset()

    def reset(self):
        self.altitude = 0.0
        self.velocity = 0.0
        self.fuel_mass = float(self.initial_fuel)
        self.time = 0.0

    @property
    def total_mass(self):
        return self.dry_mass + self.fuel_mass

    def step(self, throttle, dt=0.1):
        throttle = np.clip(throttle, 0.0, 1.0)
        thrust = throttle * self.max_thrust

        # Fuel consumption
        fuel_used = self.burn_rate * throttle * dt
        self.fuel_mass = max(0.0, self.fuel_mass - fuel_used)

        # Acceleration (Thrust - Weight)
        accel = (thrust / self.total_mass) - GRAVITY

        # Integrate motion
        self.velocity += accel * dt
        self.altitude += max(0.0, self.velocity * dt)  # only positive climb
        self.time += dt

        # Stop below ground (if physics glitch)
        if self.altitude < 0:
            self.altitude = 0
            self.velocity = 0

        return accel


class RocketLaunchEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.dt = 0.1

        self.rocket = Rocket(
            dry_mass=800.0,
            fuel_mass=1200.0,
            max_thrust=40000.0,
            burn_rate=25.0,
        )

        self.target_altitude = 10000.0
        self.max_velocity = 2500.0
        self.total_reward = 0.0

        # Actions: throttle (0–1)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observations: [altitude, velocity, fuel_mass]
        low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([self.target_altitude, self.max_velocity, 2000.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rocket.reset()
        self.total_reward = 0.0
        self.time = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(
            [self.rocket.altitude, self.rocket.velocity, self.rocket.fuel_mass],
            dtype=np.float32,
        )

    def step(self, action):
        throttle = float(np.clip(action[0], 0.0, 1.0))
        accel = self.rocket.step(throttle, self.dt)

        # Reward encourages higher altitude, smooth control, and fuel efficiency
        reward = (
            + (self.rocket.altitude * 0.05)       # reward altitude
            + (self.rocket.velocity * 0.01)       # reward upward velocity
            - (throttle * 0.02)                   # penalize high throttle (fuel use)
            - (abs(accel) * 0.001)                # smoother acceleration
        )

        terminated = False
        if self.rocket.altitude >= self.target_altitude:
            reward += 1000
            terminated = True
        elif self.rocket.fuel_mass <= 0:
            reward -= 200
            terminated = True

        truncated = self.rocket.time >= 600

        obs = self._get_obs()
        self.total_reward += reward

        info = {
            "Alt": self.rocket.altitude,
            "Vel": self.rocket.velocity,
            "Fuel": self.rocket.fuel_mass,
            "Reward": reward,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        print(
            f"Alt: {self.rocket.altitude:8.2f} m | "
            f"Vel: {self.rocket.velocity:8.2f} m/s | "
            f"Fuel: {self.rocket.fuel_mass:8.2f} kg"
        )


if __name__ == "__main__":
    env = RocketLaunchEnv()
    obs, _ = env.reset()
    for _ in range(300):
        action = np.array([1.0])
        obs, reward, done, _, _ = env.step(action)
        env.render()
        if done:
            print("\n🚀 Launch successful!")
            break
