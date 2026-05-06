import numpy as np

from rl_gridworld_green.envs.gridworld import GridWorldEnv


def test_reset_returns_valid_observation() -> None:
    """Check that reset() returns a correctly structured initial state.

    This test verifies the most basic contract of the environment:
    - the observation has the expected keys,
    - each position has shape (2,),
    - the info dictionary contains useful diagnostics,
    - the episode starts at step 0,
    - and the agent does not start directly on the goal.
    """
    # Create a small environment for testing.
    env = GridWorldEnv(size=5, max_episode_steps=10)

    # Start a new episode with a fixed seed so the setup is reproducible.
    obs, info = env.reset(seed=123)

    # The observation should contain both the agent position and goal position.
    assert "agent" in obs
    assert "goal" in obs

    # Each position should be a 2D coordinate [row, col].
    assert obs["agent"].shape == (2,)
    assert obs["goal"].shape == (2,)

    # The info dictionary should expose the episode step and Manhattan distance.
    assert "distance" in info
    assert "episode_step" in info
    assert isinstance(info["distance"], int)
    assert isinstance(info["episode_step"], int)

    # A fresh episode should always begin at step 0.
    assert info["episode_step"] == 0

    # The environment should guarantee that the start state is not already solved.
    assert not np.array_equal(obs["agent"], obs["goal"])


def test_reset_is_reproducible_with_seed() -> None:
    """Check that seeding reset() produces reproducible initial states.

    If two identical environments are reset with the same seed, they should
    produce the same initial agent position and goal position.
    """
    env1 = GridWorldEnv(size=5, max_episode_steps=10)
    env2 = GridWorldEnv(size=5, max_episode_steps=10)

    obs1, info1 = env1.reset(seed=42)
    obs2, info2 = env2.reset(seed=42)

    assert np.array_equal(obs1["agent"], obs2["agent"])
    assert np.array_equal(obs1["goal"], obs2["goal"])
    assert info1["distance"] == info2["distance"]


def test_step_returns_five_values() -> None:
    """Check that step() follows the Gymnasium five-value API.

    Gymnasium environments must return:
    (observation, reward, terminated, truncated, info)

    This test protects from accidentally using an outdated
    API or returning the wrong number of values.
    """
    env = GridWorldEnv(size=5, max_episode_steps=10)
    env.reset(seed=123)

    result = env.step(0)

    assert len(result) == 5


def test_episode_truncates_at_step_limit() -> None:
    """Check that the environment ends when the time limit is reached.

    In this test, a one-step episode horizon is used so that after a single
    action the episode must be over, either because:
    - the agent reached the goal (terminated=True), or
    - the time limit was reached (truncated=True).

    This test helps confirm that the environment handles episode boundaries
    correctly under Gymnasiums terminated/truncated API.
    """
    env = GridWorldEnv(size=5, max_episode_steps=1)
    env.reset(seed=123)

    _, reward, terminated, truncated, info = env.step(0)

    # In this environment, rewards are sparse, so a step reward is either 0 or 1.
    assert reward in (0.0, 1.0)

    # After one step in a one-step episode, the internal counter should be 1.
    assert info["episode_step"] == 1

    # The episode must now be over for one reason or the other.
    assert truncated or terminated


def test_invalid_action_raises_error() -> None:
    """Check that invalid actions are rejected explicitly.

    This protects the environment from silently accepting illegal actions,
    which would make debugging training code harder later.
    """
    env = GridWorldEnv(size=5, max_episode_steps=10)
    env.reset(seed=123)

    try:
        env.step(99)
        assert False, "Expected ValueError for invalid action"
    except ValueError:
        pass