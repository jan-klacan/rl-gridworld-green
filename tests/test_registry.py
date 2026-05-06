import gymnasium as gym

from rl_gridworld_green.registry import register_envs


def test_env_can_be_created() -> None:
    register_envs()
    env = gym.make("rl_gridworld_green/GridWorld-v0")
    obs, info = env.reset()

    assert "agent" in obs
    assert "goal" in obs
    assert "distance" in info
