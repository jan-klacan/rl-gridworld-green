from stable_baselines3.common.env_checker import check_env

from rl_gridworld_green.envs.gridworld import GridWorldEnv


def test_env_passes_sb3_checker() -> None:
    """Check that the custom environment follows the expected Gymnasium API.

    Stable-Baselines3 provides check_env() specifically for validating
    custom environments before training. This helps catch API mistakes
    early, such as wrong return shapes, invalid spaces, or reset/step
    contract issues.
    """
    # Create a normal instance of the environment.
    # Keep the settings simple because this test is about API
    # correctness, not training performance.
    env = GridWorldEnv(size=5, max_episode_steps=10)

    # Run Stable-Baselines3's environment checker.
    #
    # warn=True keeps helpful warnings visible.
    # skip_render_check=True avoids testing rendering behavior, which
    # has not been implemented yet.
    check_env(env, warn=True, skip_render_check=True)