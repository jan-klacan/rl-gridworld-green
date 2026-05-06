import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None, size: int = 5) -> None:
        self.size = size
        self.render_mode = render_mode

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32),
                "goal": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32),
            }
        )
        self.action_space = spaces.Discrete(4)

        self._agent_location = np.zeros(2, dtype=np.int32)
        self._goal_location = np.array([size - 1, size - 1], dtype=np.int32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "agent": self._agent_location.copy(),
            "goal": self._goal_location.copy(),
        }

    def _get_info(self) -> dict[str, int]:
        distance = int(np.abs(self._agent_location - self._goal_location).sum())
        return {"distance": distance}

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        super().reset(seed=seed)
        self._agent_location = np.array([0, 0], dtype=np.int32)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, int]]:
        direction_map = {
            0: np.array([-1, 0], dtype=np.int32),
            1: np.array([1, 0], dtype=np.int32),
            2: np.array([0, -1], dtype=np.int32),
            3: np.array([0, 1], dtype=np.int32),
        }
        direction = direction_map[action]
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        terminated = np.array_equal(self._agent_location, self._goal_location)
        reward = 1.0 if terminated else 0.0
        truncated = False

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info
