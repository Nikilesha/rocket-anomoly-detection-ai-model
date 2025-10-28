import numpy as np
from stable_baselines3 import PPO
from rocket_env.main_env import RocketLaunchEnv


def train_model():
    print("🚀 Initializing RocketLaunchEnv for PPO training...")
    env = RocketLaunchEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,
        tensorboard_log="./rocket_logs/",
    )

    print("\n🧠 Training PPO model...\n")
    model.learn(total_timesteps=150_000)
    model.save("rocket_ppo_model")
    print("\n✅ Model training complete and saved as 'rocket_ppo_model.zip'.")
    env.close()


def evaluate_model():
    print("\n🧠 Running trained model simulation...\n")
    model = PPO.load("rocket_ppo_model")
    env = RocketLaunchEnv()

    obs, _ = env.reset()
    total_reward = 0
    for step in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        if terminated or truncated:
            print("\n🎯 Episode finished.")
            break

    print(
        f"\nFinal Altitude: {info['Alt']:.2f} m | "
        f"Velocity: {info['Vel']:.2f} m/s | "
        f"Remaining Fuel: {info['Fuel']:.2f} kg | "
        f"Total Reward: {total_reward:.2f}"
    )


if __name__ == "__main__":
    train_model()
    evaluate_model()
