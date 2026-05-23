from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import cast

from logic.action_encoding import action_to_id
from logic.env import (
    Action,
    CallTrumpAction,
    DiscardAction,
    EuchreEnv,
    Observation,
    OrderUpAction,
    PlayCardAction,
)
from logic.monte_carlo_policy import OracleMonteCarloPolicy
from logic.observation_encoding import (
    encode_observation_without_action_mask,
    observation_action_mask,
)
from logic.policies import EuchreGame, Policy, SimpleBotPolicy


OUTPUT_PATH = Path("data/imitation_hybrid_mc.csv")


def choose_simple_bot_action(
    env: EuchreEnv,
    obs: Observation,
    policy: SimpleBotPolicy,
) -> Action:
    """
    Ask SimpleBotPolicy what action it would take in the current observation.

    Used for bidding and discarding, where SimpleBot imitation was already
    strong and oracle Monte Carlo can be too dependent on hidden information.
    """
    player = obs.player
    game_view = cast(EuchreGame, env)

    if obs.phase == "bidding_round_1":
        assert obs.upcard is not None
        return OrderUpAction(
            order_up=policy.choose_order_up(
                game_view,
                player,
                obs.upcard,
                is_dealer=(player == obs.dealer),
            )
        )

    if obs.phase == "bidding_round_2":
        assert obs.upcard is not None
        return CallTrumpAction(
            suit=policy.choose_trump(
                game_view,
                player,
                forbidden_suit=obs.upcard.suit,
                is_dealer=(player == obs.dealer),
            )
        )

    if obs.phase == "discard":
        return DiscardAction(card=policy.choose_discard(game_view, player))

    if obs.phase == "play_card":
        legal_cards = [
            action.card
            for action in obs.legal_actions
            if isinstance(action, PlayCardAction)
        ]
        return PlayCardAction(
            card=policy.choose_card(
                game_view,
                player,
                legal_cards,
                list(obs.trick),
            )
        )

    raise ValueError(f"Cannot choose a SimpleBot action during phase {obs.phase!r}.")


def choose_hybrid_action(
    env: EuchreEnv,
    obs: Observation,
    simple_teacher: SimpleBotPolicy,
    mc_teacher: OracleMonteCarloPolicy,
) -> Action:
    """
    Hybrid teacher:

    - bidding_round_1: SimpleBotPolicy
    - bidding_round_2: SimpleBotPolicy
    - discard:         SimpleBotPolicy
    - play_card:       OracleMonteCarloPolicy

    This keeps the stable SimpleBot bidding/discard behavior while using
    Monte Carlo policy improvement for the weakest phase: card play.
    """
    if obs.phase == "play_card":
        return mc_teacher.choose_action(env)

    return choose_simple_bot_action(env, obs, simple_teacher)


def generate_dataset(
    n_games: int,
    output_path: Path = OUTPUT_PATH,
    seed: int = 123,
    rollouts_per_action: int = 1,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Placeholder old-style policies for constructing EuchreEnv. Labels are
    # chosen explicitly by choose_hybrid_action().
    policies: list[Policy] = [
        SimpleBotPolicy(),
        SimpleBotPolicy(),
        SimpleBotPolicy(),
        SimpleBotPolicy(),
    ]
    env = EuchreEnv(policies=policies, winning_score=10, seed=seed)

    simple_teacher = SimpleBotPolicy()
    mc_teacher = OracleMonteCarloPolicy(rollouts_per_action=rollouts_per_action)

    examples_written = 0
    examples_by_phase: dict[str, int] = {}

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
                action = choose_hybrid_action(
                    env=env,
                    obs=obs,
                    simple_teacher=simple_teacher,
                    mc_teacher=mc_teacher,
                )

                if action not in obs.legal_actions:
                    raise ValueError(
                        f"Hybrid teacher chose illegal action {action}; "
                        f"legal actions were {obs.legal_actions}."
                    )

                state_vector = encode_observation_without_action_mask(obs)
                mask = observation_action_mask(obs)
                action_id = action_to_id(action)

                writer.writerow(state_vector + mask + [action_id])
                examples_written += 1
                examples_by_phase[obs.phase] = examples_by_phase.get(obs.phase, 0) + 1

                # Roll in using the hybrid action. This means the generated
                # state distribution follows the hybrid policy itself.
                result = env.step(action)
                obs = result.observation
                done = result.done
                steps += 1

                if steps > 1000:
                    raise RuntimeError(
                        "Game exceeded 1000 steps; possible environment loop bug."
                    )

            if (game_index + 1) % 10 == 0:
                phase_summary = ", ".join(
                    f"{phase}={count}"
                    for phase, count in sorted(examples_by_phase.items())
                )
                print(
                    f"Generated {game_index + 1}/{n_games} games, "
                    f"{examples_written} examples"
                )
                print(f"  Phase counts: {phase_summary}")

    print(f"Wrote {examples_written} examples to {output_path}")
    print("Examples by phase:")
    for phase, count in sorted(examples_by_phase.items()):
        print(f"  {phase}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hybrid SimpleBot/Monte Carlo imitation data.")
    parser.add_argument(
        "--games",
        type=int,
        default=1000,
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
        help="Number of Monte Carlo rollouts per legal play-card action.",
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
