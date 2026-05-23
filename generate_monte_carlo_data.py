from __future__ import annotations

import argparse
import csv
from pathlib import Path

from logic.action_encoding import action_to_id
from logic.env import Action, EuchreEnv
from logic.monte_carlo_policy import OracleMonteCarloPolicy
from logic.observation_encoding import (
    encode_observation_without_action_mask,
    observation_action_mask,
)
from logic.policies import Policy, SimpleBotPolicy


OUTPUT_PATH = Path("data/imitation_oracle_mc.csv")


def generate_dataset(
    n_games: int,
    output_path: Path = OUTPUT_PATH,
    seed: int = 123,
    rollouts_per_action: int = 1,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # These policies are mostly placeholders for constructing EuchreEnv and for
    # any old-style environment methods. The actual labels are chosen by the
    # OracleMonteCarloPolicy below.
    policies: list[Policy] = [
        SimpleBotPolicy(),
        SimpleBotPolicy(),
        SimpleBotPolicy(),
        SimpleBotPolicy(),
    ]

    env = EuchreEnv(policies=policies, winning_score=10, seed=seed)
    teacher = OracleMonteCarloPolicy(rollouts_per_action=rollouts_per_action)

    examples_written = 0

    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)

        # Header: x_0, ..., x_n, mask_0, ..., mask_67, action_id
        dummy_obs = env.reset()
        state_size = len(encode_observation_without_action_mask(dummy_obs))
        mask_size = len(observation_action_mask(dummy_obs))

        header = (
            [f"x_{i}" for i in range(state_size)]
            + [f"mask_{i}" for i in range(mask_size)]
            + ["action_id"]
        )
        writer.writerow(header)

        for game_index in range(n_games):
            obs = env.reset()
            done = False
            steps = 0

            while not done:
                # This is the key difference from SimpleBot imitation data:
                # the teacher evaluates legal actions by Monte Carlo rollout.
                action: Action = teacher.choose_action(env)

                if action not in obs.legal_actions:
                    raise ValueError(
                        f"Monte Carlo teacher chose illegal action {action}; "
                        f"legal actions were {obs.legal_actions}."
                    )

                state_vector = encode_observation_without_action_mask(obs)
                mask = observation_action_mask(obs)
                action_id = action_to_id(action)

                writer.writerow(state_vector + mask + [action_id])
                examples_written += 1

                # Advance using the Monte Carlo-improved action, so the dataset
                # comes from the Monte Carlo policy's own state distribution.
                result = env.step(action)
                obs = result.observation
                done = result.done
                steps += 1

                if steps > 1000:
                    raise RuntimeError(
                        "Game exceeded 1000 steps; possible environment loop bug."
                    )

            if (game_index + 1) % 10 == 0:
                print(
                    f"Generated {game_index + 1}/{n_games} games, "
                    f"{examples_written} examples"
                )

    print(f"Wrote {examples_written} examples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate oracle Monte Carlo imitation data."
    )
    parser.add_argument(
        "--games",
        type=int,
        default=500,
        help="Number of games to generate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed.",
    )
    parser.add_argument(
        "--rollouts-per-action",
        type=int,
        default=1,
        help="Number of rollouts per legal action.",
    )

    args = parser.parse_args()

    generate_dataset(
        n_games=args.games,
        output_path=args.output,
        seed=args.seed,
        rollouts_per_action=args.rollouts_per_action,
    )


if __name__ == "__main__":
    main()