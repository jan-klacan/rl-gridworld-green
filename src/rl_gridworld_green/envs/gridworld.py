import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    """Implement a simple fully observable GridWorld environment.

    The agent moves on a square grid and must reach a goal cell. This
    baseline version uses sparse rewards, full observability, and a fixed
    time limit per episode.
    """

    # Gymnasium uses 'metadata' to describe supported render modes and
    # playback speed. Rendering is not implemented yet, but this is defined
    # now so the environment already follows the expected structure.
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        size: int = 5,
        max_episode_steps: int = 50,
    ) -> None:
        """Initialize the GridWorld environment and its spaces.

        Args:
            render_mode: Optional render mode for future visualization support.
            size: Side length of the square grid.
            max_episode_steps: Maximum number of steps before truncation.
        """
        # Validate the configuration early so invalid environments fail fast
        # and clearly instead of behaving strangely later.
        if size < 2:
            raise ValueError("size must be at least 2")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")

        # Store the key task settings.
        # - 'size' controls the grid dimensions (e.g., size=5 means a 5x5 grid).
        # - 'max_episode_steps' gives the episode a time limit.
        # - 'episode_step' tracks how far we are into the current episode.
        self.size = size
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.episode_step = 0

        # Define the observation space.
        #
        # For the baseline version of the project, the environment is fully
        # observable: the agent sees both its own position and the goal
        # position. Each position is a 2D coordinate [row, col].
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32),
                "goal": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32),
            }
        )

        # Define the action space.
        #
        # The agent can choose from four discrete actions:
        # 0 = up, 1 = down, 2 = left, 3 = right.
        self.action_space = spaces.Discrete(4)

        # Translate action IDs into movement vectors on the grid.
        # This lets the policy output simple integers while the environment
        # handles the actual movement dynamics.
        self._action_to_direction = {
            0: np.array([-1, 0], dtype=np.int32),  # up
            1: np.array([1, 0], dtype=np.int32),   # down
            2: np.array([0, -1], dtype=np.int32),  # left
            3: np.array([0, 1], dtype=np.int32),   # right
        }

        # Initialize internal state placeholders.
        #
        # These are overwritten in reset(), but they are defined here so that
        # the environment always has these attributes.
        self._agent_location = np.zeros(2, dtype=np.int32)
        self._goal_location = np.zeros(2, dtype=np.int32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Return the current observation for the agent.

        Returns:
            A dictionary containing the agent and goal positions.
        """
        # Return the current observation seen by the agent.
        #
        # Return copies instead of the raw arrays so outside code cannot
        # accidentally modify the environment's internal state.
        return {
            "agent": self._agent_location.copy(),
            "goal": self._goal_location.copy(),
        }

    def _get_info(self) -> dict[str, int]:
        """Return auxiliary diagnostic information about the state.

        Returns:
            A dictionary with the Manhattan distance to the goal and the
            current episode step count.
        """
        # Return extra diagnostic information.
        #
        # 'info' is useful for debugging, logging, and evaluation, but it is
        # not part of the formal observation seen by the policy.
        distance = int(np.abs(self._agent_location - self._goal_location).sum())
        return {
            "distance": distance,  # Manhattan distance between the agent and the goal
            "episode_step": self.episode_step,
        }

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        """Start a new episode and return the initial state.

        Args:
            seed: Optional random seed for reproducibility.
            options: Optional extra configuration, unused in this environment.

        Returns:
            A tuple of (observation, info) for the new episode.
        """
        # Let Gymnasium initialize its internal RNG.
        # This enables reproducibility when reset() is called with a seed.
        super().reset(seed=seed)

        # A new episode starts at step zero.
        self.episode_step = 0

        # Sample a random starting location for the agent.
        # Use Gymnasium's seeded RNG (self.np_random) and not the global
        # NumPy randomness so experiments are reproducible.
        self._agent_location = self.np_random.integers(
            0, self.size, size=2, dtype=np.int32
        )

        # Sample a random goal location.
        # The goal must not be identical to the agent's start position, so
        # keep sampling until the two positions differ.
        self._goal_location = self._agent_location.copy()
        while np.array_equal(self._goal_location, self._agent_location):
            self._goal_location = self.np_random.integers(
                0, self.size, size=2, dtype=np.int32
            )

        # Return the initial observation and info for the new episode.
        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, int]]:
        """Advance the environment by one action.

        Args:
            action: The discrete action chosen by the agent.

        Returns:
            A tuple of (observation, reward, terminated, truncated, info).
        """
        # Reject invalid actions explicitly.
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Take one more step inside the current episode.
        self.episode_step += 1

        # Convert the chosen action into a movement direction and update the
        # agent's position.
        #
        # np.clip keeps the agent inside the grid boundaries. If the agent
        # tries to move off the map, it simply stays on the border.
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction,
            0,
            self.size - 1,
        )

        # Check whether the task itself has been completed.
        #
        # 'terminated' means the episode ended because of the environment's
        # actual success condition: the agent reached the goal.
        terminated = np.array_equal(self._agent_location, self._goal_location)

        # Check whether the episode hit the external time limit.
        #
        # 'truncated' is different from 'terminated': the task was not
        # necessarily solved, but the episode is forced to stop because it has
        # become too long.
        truncated = self.episode_step >= self.max_episode_steps and not terminated

        # Define the reward function.
        #
        # This baseline environment uses a sparse reward:
        # +1 when the goal is reached
        #  0 otherwise
        reward = 1.0 if terminated else 0.0

        # Return the full transition tuple required by Gymnasium.
        return self._get_obs(), reward, terminated, truncated, self._get_info()