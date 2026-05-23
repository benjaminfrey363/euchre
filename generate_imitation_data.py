from __future__ import annotations

import csv
from pathlib import Path
from typing import cast

import argparse

from logic.action_encoding import action_to_id
from logic.env import Action, EuchreEnv, Observation, PlayCardAction
from logic.observation_encoding import encode_observation_without_action_mask, observation_action_mask
from logic.policies import EuchreGame, Policy, SimpleBotPolicy


OUTPUT_PATH = Path("data/imitation_simple_bot.csv")


def choose_simple_bot_action(env: EuchreEnv, obs: Observation, policy: SimpleBotPolicy) -> Action:
    """
    Use the existing method-based SimpleBotPolicy to choose one symbolic Action.

    The environment is now action-based, but SimpleBotPolicy still has the old
    interface:
        choose_order_up(...)
        choose_trump(...)
        choose_discard(...)
        choose_card(...)

    This adapter lets us generate supervised examples from SimpleBotPolicy.
    """
    player = obs.player
    game_view = cast(EuchreGame, env)

    if obs.phase == "bidding_round_1":
        assert obs.upcard is not None
        order_up = policy.choose_order_up(
            game_view,
            player,
            obs.upcard,
            is_dealer=(player == obs.dealer),
        )
        from logic.env import OrderUpAction

        return OrderUpAction(order_up=order_up)

    if obs.phase == "bidding_round_2":
        assert obs.upcard is not None
        suit = policy.choose_trump(
            game_view,
            player,
            forbidden_suit=obs.upcard.suit,
            is_dealer=(player == obs.dealer),
        )
        from logic.env import CallTrumpAction

        return CallTrumpAction(suit=suit)

    if obs.phase == "discard":
        card = policy.choose_discard(game_view, player)
        from logic.env import DiscardAction

        return DiscardAction(card=card)

    if obs.phase == "play_card":
        legal_cards = [
            action.card
            for action in obs.legal_actions
            if isinstance(action, PlayCardAction)
        ]

        card = policy.choose_card(game_view, player, legal_cards, list(obs.trick))

        return PlayCardAction(card=card)

    raise ValueError(f"Cannot choose an action during phase {obs.phase!r}.")


def generate_dataset(n_games: int, output_path: Path = OUTPUT_PATH, seed: int = 123) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    policies: list[Policy] = [
        SimpleBotPolicy(),
        SimpleBotPolicy(),
        SimpleBotPolicy(),
        SimpleBotPolicy(),
    ]
    env = EuchreEnv(policies=policies, winning_score=10, seed=seed)

    # One action-selection policy object is enough because SimpleBotPolicy has
    # no internal state.
    teacher = SimpleBotPolicy()

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
                action = choose_simple_bot_action(env, obs, teacher)

                if action not in obs.legal_actions:
                    raise ValueError(
                        f"Teacher chose illegal action {action}; legal actions were {obs.legal_actions}."
                    )

                state_vector = encode_observation_without_action_mask(obs)
                mask = observation_action_mask(obs)
                action_id = action_to_id(action)

                writer.writerow(state_vector + mask + [action_id])
                examples_written += 1

                result = env.step(action)
                obs = result.observation
                done = result.done
                steps += 1

                if steps > 1000:
                    raise RuntimeError("Game exceeded 1000 steps; possible environment loop bug.")

            if (game_index + 1) % 100 == 0:
                print(f"Generated {game_index + 1}/{n_games} games, {examples_written} examples")

    print(f"Wrote {examples_written} examples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SimpleBot imitation data.")
    parser.add_argument(
        "--games",
        type=int,
        default=10_000,
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

    args = parser.parse_args()

    generate_dataset(
        n_games=args.games,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

