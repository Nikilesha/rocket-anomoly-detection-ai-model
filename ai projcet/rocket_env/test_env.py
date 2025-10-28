from rocket_env.main_env import RocketEnv
import numpy as np

env = RocketEnv()
obs = env.reset()

for step in range(100):
    action = np.array([0.0, 0.6], dtype=np.float32)  # angle_adjust=0, throttle=0.6

    obs, reward, done, _ = env.step(action)
    print(f"Step {step+1}: obs={obs}, reward={reward}, done={done}")

    if done:
        print("Rocket ended at step", step+1)
        obs = env.reset()
        print("Environment reset!")
