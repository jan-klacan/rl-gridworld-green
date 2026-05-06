from gymnasium.envs.registration import register


def register_envs() -> None:
    register(
        id="rl_gridworld_green/GridWorld-v0",
        entry_point="rl_gridworld_green.envs.gridworld:GridWorldEnv",
    )
