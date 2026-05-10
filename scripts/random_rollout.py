"""Run random-policy rollouts in the registered GridWorld environment.

This is the first post-testing sanity check script.
It does not learn anything yet. It only interacts with the environment using
random actions so that the following can be inspected:
- episode returns
- episode lengths
- ending reasons (terminated vs truncated)
"""

import argparse

import gymnasium as gym

from rl_gridworld_green.registry import register_envs

ENV_ID = "rl_gridworld_green/GridWorld-v0"


def run_random_episode(env: gym.Env, trace: bool = False) -> tuple[float, int, str]:
    """Run one episode with random actions and return a short summary."""

    # Reset the environment to get the initial observation and info dict.
    obs, info = env.reset()

    total_reward = 0.0
    steps = 0

    # If trace is enabled, print the initial observation and info dict.
    if trace:
        print(f"reset -> obs={obs}, info={info}")

    # Keep sampling random actions until the episode ends.
    while True:
        action = env.action_space.sample()  # Sample a random action from the action space.
        obs, reward, terminated, truncated, info = env.step(action)  # Take a step in the environment using the sampled action.

        total_reward += reward
        steps += 1

        # If trace is enabled, print the details of the transition.
        if trace:
            print(
                f"step={steps} action={action} reward={reward} "
                f"terminated={terminated} truncated={truncated} info={info}"
            )


        # Check if the episode has ended due to termination or truncation and return the summary.

        if terminated:
            return total_reward, steps, "terminated"

        if truncated:
            return total_reward, steps, "truncated"


def main() -> None:
    """Create the environment and run a few random episodes."""
    parser = argparse.ArgumentParser(
        description="Run random-policy rollouts in GridWorld."
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=10)
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print every transition instead of only episode summaries.",
    )
    args = parser.parse_args()

    # Register the environment so Gymnasium can create it by ID.
    register_envs()

    env = gym.make(
        ENV_ID,
        size=args.size,
        max_episode_steps=args.max_episode_steps,
    )

    try:
        # Run several episodes and print a compact summary for each one.
        for episode in range(1, args.episodes + 1):
            total_reward, steps, end_reason = run_random_episode(
                env, trace=args.trace
            )
            print(
                f"episode={episode} return={total_reward:.1f} "
                f"steps={steps} end={end_reason}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()