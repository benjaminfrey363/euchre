from __future__ import annotations

import argparse
from typing import Any, cast

from logic.cards import Card, Suit
from logic.env import EuchreEnv, Observation
from logic.monte_carlo_policy import OracleMonteCarloPolicy
from logic.policies import EuchreGame, Policy, RandomPolicy, SimpleBotPolicy


class MonteCarloPolicyAdapter:
    """
    Adapter from OracleMonteCarloPolicy to the old method-based Policy protocol.

    EuchreEnv.run_many_games still expects Policy objects, so this adapter lets
    the Monte Carlo policy be used in the existing evaluator.
    """

    def __init__(self, policy: OracleMonteCarloPolicy):
        self.policy = policy

    def _env(self, game: EuchreGame) -> EuchreEnv:
        return cast(Any, game)

    def choose_order_up(
        self,
        game: EuchreGame,
        player: int,
        upcard: Card,
        is_dealer: bool,
    ) -> bool:
        from logic.env import OrderUpAction

        action = self.policy.choose_action(self._env(game))
        if not isinstance(action, OrderUpAction):
            raise TypeError(f"Expected OrderUpAction, got {action!r}")
        return action.order_up

    def choose_trump(
        self,
        game: EuchreGame,
        player: int,
        forbidden_suit: Suit,
        is_dealer: bool,
    ) -> Suit | None:
        from logic.env import CallTrumpAction

        action = self.policy.choose_action(self._env(game))
        if not isinstance(action, CallTrumpAction):
            raise TypeError(f"Expected CallTrumpAction, got {action!r}")
        return action.suit

    def choose_discard(self, game: EuchreGame, player: int) -> Card:
        from logic.env import DiscardAction

        action = self.policy.choose_action(self._env(game))
        if not isinstance(action, DiscardAction):
            raise TypeError(f"Expected DiscardAction, got {action!r}")
        return action.card

    def choose_card(
        self,
        game: EuchreGame,
        player: int,
        legal: list[Card],
        trick: list[tuple[int, Card]],
    ) -> Card:
        from logic.env import PlayCardAction

        action = self.policy.choose_action(self._env(game))
        if not isinstance(action, PlayCardAction):
            raise TypeError(f"Expected PlayCardAction, got {action!r}")
        return action.card


def make_mc_policy(rollouts_per_action: int) -> Policy:
    return MonteCarloPolicyAdapter(
        OracleMonteCarloPolicy(rollouts_per_action=rollouts_per_action)
    )


def print_stats(name: str, stats) -> None:
    team_0_wins, team_1_wins = stats.team_wins
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Games: {stats.games_played}")
    print(f"Team 0 wins: {team_0_wins} ({100.0 * team_0_wins / stats.games_played:.2f}%)")
    print(f"Team 1 wins: {team_1_wins} ({100.0 * team_1_wins / stats.games_played:.2f}%)")
    print(f"Average score: {stats.average_score[0]:.3f} - {stats.average_score[1]:.3f}")
    print(f"Average hands/game: {stats.average_hands_per_game:.3f}")


def run_matchup(
    name: str,
    policies: list[Policy],
    games: int,
    seed: int,
) -> None:
    env = EuchreEnv(policies=policies, winning_score=10, seed=seed)
    stats = env.run_many_games(games)
    print_stats(name, stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate oracle Monte Carlo Euchre policy.")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rollouts-per-action", type=int, default=1)
    args = parser.parse_args()

    # This is intentionally small by default. Oracle Monte Carlo uses deep copies
    # and rollouts at every decision, so it is much slower than SimpleBotPolicy.
    run_matchup(
        "Oracle MC team vs Simple team",
        [
            make_mc_policy(args.rollouts_per_action),
            SimpleBotPolicy(),
            make_mc_policy(args.rollouts_per_action),
            SimpleBotPolicy(),
        ],
        games=args.games,
        seed=args.seed,
    )

    run_matchup(
        "Simple team vs Oracle MC team",
        [
            SimpleBotPolicy(),
            make_mc_policy(args.rollouts_per_action),
            SimpleBotPolicy(),
            make_mc_policy(args.rollouts_per_action),
        ],
        games=args.games,
        seed=args.seed,
    )

    run_matchup(
        "Oracle MC team vs Random team",
        [
            make_mc_policy(args.rollouts_per_action),
            RandomPolicy(),
            make_mc_policy(args.rollouts_per_action),
            RandomPolicy(),
        ],
        games=args.games,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

