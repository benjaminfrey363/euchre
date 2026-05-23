from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from logic.cards import Card, Suit
from logic.env import EuchreEnv, Observation, SimulationStats
from logic.model_policy import ModelActionPolicy
from logic.policies import EuchreGame, Policy, RandomPolicy, SimpleBotPolicy

import argparse

DEFAULT_N_GAMES = 10_000
DEFAULT_SEEDS = [123, 456, 789]


class ActionPolicyAdapter:
    """
    Adapter from the new ActionPolicy interface to the older Policy interface.

    EuchreEnv.run_many_games currently expects old method-based Policy objects.
    This adapter lets a model policy participate in those simulations.
    """

    def __init__(self, action_policy: ModelActionPolicy):
        self.action_policy = action_policy

    def _observation(self, game: EuchreGame, player: int) -> Observation:
        # The old Policy protocol is typed against EuchreGame, but at runtime
        # this adapter is used with EuchreEnv, which has observation_for_player().
        env = cast(Any, game)
        return env.observation_for_player(player)

    def choose_order_up(
        self,
        game: EuchreGame,
        player: int,
        upcard: Card,
        is_dealer: bool,
    ) -> bool:
        obs = self._observation(game, player)
        action = self.action_policy.choose_action(obs)
        from logic.env import OrderUpAction

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
        obs = self._observation(game, player)
        action = self.action_policy.choose_action(obs)
        from logic.env import CallTrumpAction

        if not isinstance(action, CallTrumpAction):
            raise TypeError(f"Expected CallTrumpAction, got {action!r}")
        return action.suit

    def choose_discard(self, game: EuchreGame, player: int) -> Card:
        obs = self._observation(game, player)
        action = self.action_policy.choose_action(obs)
        from logic.env import DiscardAction

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
        obs = self._observation(game, player)
        action = self.action_policy.choose_action(obs)
        from logic.env import PlayCardAction

        if not isinstance(action, PlayCardAction):
            raise TypeError(f"Expected PlayCardAction, got {action!r}")
        return action.card


@dataclass(frozen=True)
class MatchupResult:
    name: str
    seed: int
    stats: SimulationStats

    @property
    def team_0_win_rate(self) -> float:
        return self.stats.team_wins[0] / self.stats.games_played

    @property
    def team_1_win_rate(self) -> float:
        return self.stats.team_wins[1] / self.stats.games_played


def make_model_policy(model_path: Path) -> Policy:
    return ActionPolicyAdapter(ModelActionPolicy(model_path))


def format_stats(stats: SimulationStats) -> str:
    team_0_wins, team_1_wins = stats.team_wins
    team_0_win_rate = 100.0 * team_0_wins / stats.games_played
    team_1_win_rate = 100.0 * team_1_wins / stats.games_played

    return (
        f"Games: {stats.games_played}\n"
        f"Team 0 wins: {team_0_wins} ({team_0_win_rate:.2f}%)\n"
        f"Team 1 wins: {team_1_wins} ({team_1_win_rate:.2f}%)\n"
        f"Average score: {stats.average_score[0]:.3f} - {stats.average_score[1]:.3f}\n"
        f"Average hands/game: {stats.average_hands_per_game:.3f}"
    )


def summarize_across_seeds(name: str, results: list[MatchupResult]) -> None:
    if len(results) <= 1:
        return

    games = sum(result.stats.games_played for result in results)
    team_0_wins = sum(result.stats.team_wins[0] for result in results)
    team_1_wins = sum(result.stats.team_wins[1] for result in results)

    weighted_score_0 = sum(
        result.stats.average_score[0] * result.stats.games_played
        for result in results
    ) / games
    weighted_score_1 = sum(
        result.stats.average_score[1] * result.stats.games_played
        for result in results
    ) / games
    weighted_hands = sum(
        result.stats.average_hands_per_game * result.stats.games_played
        for result in results
    ) / games

    print(f"\n{name} — combined across {len(results)} seeds")
    print("-" * (len(name) + 31))
    print(f"Games: {games}")
    print(f"Team 0 wins: {team_0_wins} ({100.0 * team_0_wins / games:.2f}%)")
    print(f"Team 1 wins: {team_1_wins} ({100.0 * team_1_wins / games:.2f}%)")
    print(f"Average score: {weighted_score_0:.3f} - {weighted_score_1:.3f}")
    print(f"Average hands/game: {weighted_hands:.3f}")


def run_matchup(
    name: str,
    policies: list[Policy],
    n_games: int = DEFAULT_N_GAMES,
    seed: int = 123,
) -> MatchupResult:
    env = EuchreEnv(policies=policies, winning_score=10, seed=seed)
    stats = env.run_many_games(n_games)

    print(f"\n{name} — seed {seed}")
    print("-" * (len(name) + len(str(seed)) + 8))
    print(format_stats(stats))

    return MatchupResult(name=name, seed=seed, stats=stats)


def evaluate_all_matchups(
    model_path: Path,
    n_games: int = DEFAULT_N_GAMES,
    seeds: list[int] | None = None,
) -> None:
    if seeds is None:
        seeds = DEFAULT_SEEDS

    matchup_results: dict[str, list[MatchupResult]] = {}

    for seed in seeds:
        # Create fresh model adapters for each matchup. This avoids sharing any
        # accidental mutable state if the policy later becomes stochastic.
        '''
        matchups: list[tuple[str, list[Policy]]] = [
            (
                "Model team vs Random team",
                [
                    make_model_policy(model_path),
                    RandomPolicy(),
                    make_model_policy(model_path),
                    RandomPolicy(),
                ],
            ),
            (
                "Random team vs Model team",
                [
                    RandomPolicy(),
                    make_model_policy(model_path),
                    RandomPolicy(),
                    make_model_policy(model_path),
                ],
            ),
            (
                "Model team vs Simple team",
                [
                    make_model_policy(model_path),
                    SimpleBotPolicy(),
                    make_model_policy(model_path),
                    SimpleBotPolicy(),
                ],
            ),
            (
                "Simple team vs Model team",
                [
                    SimpleBotPolicy(),
                    make_model_policy(model_path),
                    SimpleBotPolicy(),
                    make_model_policy(model_path),
                ],
            ),
        ]
        '''
        matchups: list[tuple[str, list[Policy]]] = [
            (
                "Model team vs Random team",
                [
                    make_model_policy(model_path),
                    RandomPolicy(),
                    make_model_policy(model_path),
                    RandomPolicy(),
                ],
            )
        ]
        

        for name, policies in matchups:
            result = run_matchup(name=name, policies=policies, n_games=n_games, seed=seed)
            matchup_results.setdefault(name, []).append(result)

    for name, results in matchup_results.items():
        summarize_across_seeds(name, results)



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Euchre model policy.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/imitation_simple_bot.pt"),
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=DEFAULT_N_GAMES,
        help="Number of games per matchup.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Evaluation seeds.",
    )

    args = parser.parse_args()

    evaluate_all_matchups(
        model_path=args.model,
        n_games=args.games,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()

