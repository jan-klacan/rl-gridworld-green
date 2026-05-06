from rl_gridworld_green.envs.gridworld import GridWorldEnv


def test_reset_returns_valid_observation() -> None:
    env = GridWorldEnv(size=5)
    obs, info = env.reset()

    assert obs["agent"].shape == (2,)
    assert obs["goal"].shape == (2,)
    assert isinstance(info["distance"], int)


def test_step_returns_five_values() -> None:
    env = GridWorldEnv(size=5)
    env.reset()
    result = env.step(0)

    assert len(result) == 5
