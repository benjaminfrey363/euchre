from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from logic.cards import Card, Suit
from logic.env import EuchreEnv, SimulationStats
from logic.information_set_monte_carlo_policy import InformationSetMonteCarloPolicy
from logic.monte_carlo_policy import OracleMonteCarloPolicy
from logic.policies import EuchreGame, Policy, RandomPolicy, SimpleBotPolicy


DEFAULT_MODEL_PATH = Path("models/best_mixed_simple3k_ismc500.pt")
DEFAULT_SEEDS = [123, 456, 789]


class EnvActionPolicyAdapter:
    """
    Adapter for policies whose choose_action() needs the full EuchreEnv.

    This covers search policies such as information-set MC and oracle MC. The
    wrapped policy is responsible for respecting the information constraints it
    claims to use.
    """

    def __init__(self, policy: Any):
        self.policy = policy

    def _env(self, game: EuchreGame) -> EuchreEnv:
        return cast(EuchreEnv, game)

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


@dataclass(frozen=True)
class MatchupRun:
    policy_name: str
    orientation: str
    challenger_team: int
    seed: int
    stats: SimulationStats

    @property
    def challenger_wins(self) -> int:
        return self.stats.team_wins[self.challenger_team]

    @property
    def challenger_win_rate(self) -> float:
        return self.challenger_wins / self.stats.games_played


PolicyFactory = Callable[[int], Policy]


def make_model_policy_factory(model_path: Path) -> PolicyFactory:
    def factory(seed: int) -> Policy:
        del seed
        from evaluate_model_policy import make_model_policy

        return make_model_policy(model_path)

    return factory


def make_ismc_policy_factory(samples_per_action: int) -> PolicyFactory:
    def factory(seed: int) -> Policy:
        return EnvActionPolicyAdapter(
            InformationSetMonteCarloPolicy(
                samples_per_action=samples_per_action,
                seed=seed,
            )
        )

    return factory


def make_oracle_policy_factory(rollouts_per_action: int) -> PolicyFactory:
    def factory(seed: int) -> Policy:
        del seed
        return EnvActionPolicyAdapter(
            OracleMonteCarloPolicy(rollouts_per_action=rollouts_per_action)
        )

    return factory


def make_simple_policy_factory() -> PolicyFactory:
    def factory(seed: int) -> Policy:
        del seed
        return SimpleBotPolicy()

    return factory


def make_random_policy_factory() -> PolicyFactory:
    def factory(seed: int) -> Policy:
        del seed
        return RandomPolicy()

    return factory


def confidence_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval for a binomial win rate."""
    if games <= 0:
        raise ValueError("games must be positive.")

    p = wins / games
    denominator = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / denominator
    half_width = (
        z
        * math.sqrt((p * (1.0 - p) + z * z / (4.0 * games)) / games)
        / denominator
    )
    return center - half_width, center + half_width


def combined_average_score(
    runs: list[MatchupRun],
    challenger_team: int,
) -> tuple[float, float]:
    games = sum(run.stats.games_played for run in runs)
    challenger_score = sum(
        run.stats.average_score[challenger_team] * run.stats.games_played
        for run in runs
    )
    baseline_score = sum(
        run.stats.average_score[1 - challenger_team] * run.stats.games_played
        for run in runs
    )
    return challenger_score / games, baseline_score / games


def build_matchup(
    challenger_factory: PolicyFactory,
    baseline_factory: PolicyFactory,
    challenger_team: int,
    seed: int,
) -> list[Policy]:
    if challenger_team == 0:
        return [
            challenger_factory(seed + 1),
            baseline_factory(seed + 2),
            challenger_factory(seed + 3),
            baseline_factory(seed + 4),
        ]

    return [
        baseline_factory(seed + 1),
        challenger_factory(seed + 2),
        baseline_factory(seed + 3),
        challenger_factory(seed + 4),
    ]


def run_matchup(
    policy_name: str,
    challenger_factory: PolicyFactory,
    baseline_factory: PolicyFactory,
    challenger_team: int,
    games: int,
    seed: int,
) -> MatchupRun:
    orientation = "challenger as Team 0" if challenger_team == 0 else "challenger as Team 1"
    env = EuchreEnv(
        policies=build_matchup(
            challenger_factory=challenger_factory,
            baseline_factory=baseline_factory,
            challenger_team=challenger_team,
            seed=seed,
        ),
        winning_score=10,
        seed=seed,
    )
    stats = env.run_many_games(games)
    win_rate = 100.0 * stats.team_wins[challenger_team] / stats.games_played

    print(
        f"{policy_name:16} | {orientation:20} | seed {seed:<6} | "
        f"{stats.team_wins[challenger_team]:>5}/{stats.games_played:<5} "
        f"({win_rate:5.2f}%)"
    )

    return MatchupRun(
        policy_name=policy_name,
        orientation=orientation,
        challenger_team=challenger_team,
        seed=seed,
        stats=stats,
    )


def summarize_policy(policy_name: str, runs: list[MatchupRun]) -> None:
    total_games = sum(run.stats.games_played for run in runs)
    total_wins = sum(run.challenger_wins for run in runs)
    low, high = confidence_interval(total_wins, total_games)
    win_rate = total_wins / total_games

    team_0_runs = [run for run in runs if run.challenger_team == 0]
    team_1_runs = [run for run in runs if run.challenger_team == 1]
    team_0_score = combined_average_score(team_0_runs, challenger_team=0)
    team_1_score = combined_average_score(team_1_runs, challenger_team=1)

    print(f"\n{policy_name} vs SimpleBot")
    print("-" * (len(policy_name) + 13))
    print(f"Games: {total_games}")
    print(
        f"Challenger wins: {total_wins} "
        f"({100.0 * win_rate:.2f}%, 95% CI {100.0 * low:.2f}-{100.0 * high:.2f}%)"
    )
    print(f"Average score as Team 0: {team_0_score[0]:.3f} - {team_0_score[1]:.3f}")
    print(f"Average score as Team 1: {team_1_score[0]:.3f} - {team_1_score[1]:.3f}")


def make_factories(args: argparse.Namespace) -> dict[str, PolicyFactory]:
    factories: dict[str, PolicyFactory] = {
        "random": make_random_policy_factory(),
        "simple": make_simple_policy_factory(),
        "ismc": make_ismc_policy_factory(args.ismc_samples_per_action),
        "oracle": make_oracle_policy_factory(args.oracle_rollouts_per_action),
    }

    if args.model is not None:
        factories["model"] = make_model_policy_factory(args.model)

    return factories


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Euchre policies against SimpleBot from both seating "
            "orientations. Oracle MC is a hidden-information upper-bound probe, "
            "not a fair deployable policy."
        )
    )
    parser.add_argument("--games", type=int, default=1000, help="Games per seed and orientation.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["model", "random", "simple", "ismc", "oracle"],
        default=["model", "ismc", "oracle"],
        help="Policies to benchmark against SimpleBot.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Model checkpoint for the model policy. Use --policies without model to avoid torch.",
    )
    parser.add_argument(
        "--ismc-samples-per-action",
        type=int,
        default=20,
        help="Hidden-world samples per legal action for fair information-set MC.",
    )
    parser.add_argument(
        "--oracle-rollouts-per-action",
        type=int,
        default=1,
        help="Rollouts per legal action for hidden-information oracle MC.",
    )
    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError("--games must be positive.")

    factories = make_factories(args)
    baseline_factory = make_simple_policy_factory()

    print("Policy frontier vs SimpleBot")
    print("===========================")
    print(
        "Oracle MC sees hidden cards and is best read as an upper-bound probe, "
        "not perfect fair play."
    )
    print()

    all_runs: dict[str, list[MatchupRun]] = {}
    for policy_name in args.policies:
        factory = factories[policy_name]
        for seed in args.seeds:
            for challenger_team in (0, 1):
                run = run_matchup(
                    policy_name=policy_name,
                    challenger_factory=factory,
                    baseline_factory=baseline_factory,
                    challenger_team=challenger_team,
                    games=args.games,
                    seed=seed,
                )
                all_runs.setdefault(policy_name, []).append(run)

    for policy_name, runs in all_runs.items():
        summarize_policy(policy_name, runs)


if __name__ == "__main__":
    main()
