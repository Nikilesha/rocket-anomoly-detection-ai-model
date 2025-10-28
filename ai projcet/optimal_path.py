import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from rocket_env.main_env import RocketLaunchEnv


def compute_and_visualize_optimal_path(model_path="rocket_ppo_model.zip", max_steps=2000):
    """
    Simulates and visualizes the rocket's optimal launch trajectory
    using the trained PPO model.
    """
    print("🧠 Loading trained PPO model for optimal trajectory visualization...\n")
    model = PPO.load(model_path)
    env = RocketLaunchEnv()

    obs, _ = env.reset()
    path_data = []

    # 🧩 Run simulation using the learned PPO policy
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        path_data.append({
            "step": step,
            "altitude": info["Alt"],
            "velocity": info["Vel"],
            "fuel": info["Fuel"],
            "reward": info["Reward"],
            "action": float(action[0])
        })

        # Optional live terminal feedback
        print(
            f"Step {step:04d} | Alt: {info['Alt']:8.2f} m | "
            f"Vel: {info['Vel']:7.2f} m/s | Fuel: {info['Fuel']:8.2f} kg | "
            f"Thrust: {action[0]:.2f}"
        )

        if terminated or truncated:
            print("\n🎯 Optimal trajectory complete.")
            break

    env.close()

    # ✅ Extract data for plotting
    steps = [d["step"] for d in path_data]
    altitude = [d["altitude"] for d in path_data]
    velocity = [d["velocity"] for d in path_data]
    fuel = [d["fuel"] for d in path_data]
    thrust = [d["action"] for d in path_data]

    # 🪄 Visualization
    plt.figure(figsize=(12, 9))
    plt.suptitle("🚀 Rocket Launch Trajectory (Learned Optimal Path)", fontsize=16, weight="bold")

    # Altitude
    plt.subplot(4, 1, 1)
    plt.plot(steps, altitude, linewidth=2)
    plt.ylabel("Altitude (m)")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Velocity
    plt.subplot(4, 1, 2)
    plt.plot(steps, velocity, linewidth=2)
    plt.ylabel("Velocity (m/s)")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Fuel
    plt.subplot(4, 1, 3)
    plt.plot(steps, fuel, linewidth=2, color="orange")
    plt.ylabel("Fuel (kg)")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Thrust
    plt.subplot(4, 1, 4)
    plt.plot(steps, thrust, linewidth=2, color="red")
    plt.ylabel("Throttle (0–1)")
    plt.xlabel("Simulation Steps")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    final = path_data[-1]
    print(
        f"\n🏁 Final Altitude: {final['altitude']:.2f} m | "
        f"Velocity: {final['velocity']:.2f} m/s | "
        f"Fuel Remaining: {final['fuel']:.2f} kg | "
        f"Total Steps: {len(path_data)}"
    )


if __name__ == "__main__":
    compute_and_visualize_optimal_path()
