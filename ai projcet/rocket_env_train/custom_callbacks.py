import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RocketLoggingCallback(BaseCallback):
    """
    Custom callback for logging rocket-specific metrics during training.
    Logs: altitude, velocity, fuel, deviation, and episode rewards.
    """

    def __init__(self, log_dir="logs", verbose=1):
        super(RocketLoggingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_episode_reward = 0
        self._current_episode_length = 0

    def _on_step(self) -> bool:
        # Called at every environment step
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", None)
        if rewards is not None:
            self._current_episode_reward += np.sum(rewards)
            self._current_episode_length += 1

        # Check for episode end
        for info in infos:
            if info.get("episode_end", False) or info.get("cumulative_reward", None):
                self.episode_rewards.append(self._current_episode_reward)
                self.episode_lengths.append(self._current_episode_length)
                self._current_episode_reward = 0
                self._current_episode_length = 0

        return True

    def _on_training_end(self):
        # Save metrics at the end of training
        if self.episode_rewards:
            rewards_file = os.path.join(self.log_dir, "episode_rewards.npy")
            lengths_file = os.path.join(self.log_dir, "episode_lengths.npy")
            np.save(rewards_file, np.array(self.episode_rewards))
            np.save(lengths_file, np.array(self.episode_lengths))
            if self.verbose > 0:
                print(f"✅ Saved episode rewards to {rewards_file}")
                print(f"✅ Saved episode lengths to {lengths_file}")

    def log_episode(self, reward, length):
        # Can be used manually to log an episode
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
