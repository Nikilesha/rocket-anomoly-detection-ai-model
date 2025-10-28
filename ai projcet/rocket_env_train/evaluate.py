import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rocket_env.main_env import RocketEnv

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_CONFIG_PATH = os.path.join(BASE_DIR, "config", "env_config.yaml")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "trained_models", "ppo_rocket_final.zip")
TELEMETRY_CSV = os.path.join(BASE_DIR, "sample_tel", "telemetry_sample.csv")

# Environment creation
def make_env():
    return RocketEnv(config_path=ENV_CONFIG_PATH, telemetry_csv=TELEMETRY_CSV)

env = DummyVecEnv([make_env])

# Load model
model = PPO.load(MODEL_SAVE_PATH, env=env)

# Evaluate
num_episodes = 5
for ep in range(num_episodes):
    obs = env.reset()  # only returns observation for VecEnv
    done = [False]
    cumulative_reward = 0
    step_num = 0

    print(f"\n=== Starting Episode {ep+1} ===")

    while not done[0]:
        step_num += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)  # VecEnv returns 4 values
        cumulative_reward += reward[0]

        print(f"Step {step_num}:")
        print(f"  Action: {action}")
        print(f"  Observation: {obs}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}\n")

    print(f"✅ Episode {ep+1} finished! Total Reward: {cumulative_reward}")

env.close()
