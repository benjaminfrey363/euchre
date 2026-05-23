from __future__ import annotations

import argparse
from typing import Any, cast

from logic.cards import Card, Suit
from logic.env import EuchreEnv
from logic.information_set_monte_carlo_policy import InformationSetMonteCarloPolicy
from logic.policies import EuchreGame, Policy, RandomPolicy, SimpleBotPolicy


class InformationSetMonteCarloPolicyAdapter:
    """
    Adapter from InformationSetMonteCarloPolicy to the old method-based Policy
    protocol used by EuchreEnv.run_many_games().
    """

    def __init__(self, policy: InformationSetMonteCarloPolicy):
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


def make_ismc_policy(samples_per_action: int, seed: int) -> Policy:
    return InformationSetMonteCarloPolicyAdapter(
        InformationSetMonteCarloPolicy(
            samples_per_action=samples_per_action,
            seed=seed,
        )
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


def run_matchup(name: str, policies: list[Policy], games: int, seed: int) -> None:
    env = EuchreEnv(policies=policies, winning_score=10, seed=seed)
    stats = env.run_many_games(games)
    print_stats(name, stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate information-set Monte Carlo Euchre policy.")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--samples-per-action", type=int, default=10)
    args = parser.parse_args()

    run_matchup(
        "Information-set MC team vs Simple team",
        [
            make_ismc_policy(args.samples_per_action, args.seed + 1),
            SimpleBotPolicy(),
            make_ismc_policy(args.samples_per_action, args.seed + 2),
            SimpleBotPolicy(),
        ],
        games=args.games,
        seed=args.seed,
    )

    run_matchup(
        "Simple team vs Information-set MC team",
        [
            SimpleBotPolicy(),
            make_ismc_policy(args.samples_per_action, args.seed + 3),
            SimpleBotPolicy(),
            make_ismc_policy(args.samples_per_action, args.seed + 4),
        ],
        games=args.games,
        seed=args.seed,
    )

    run_matchup(
        "Information-set MC team vs Random team",
        [
            make_ismc_policy(args.samples_per_action, args.seed + 5),
            RandomPolicy(),
            make_ismc_policy(args.samples_per_action, args.seed + 6),
            RandomPolicy(),
        ],
        games=args.games,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

