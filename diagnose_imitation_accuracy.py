from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
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
from logic.model_policy import ModelActionPolicy
from logic.policies import EuchreGame, Policy, SimpleBotPolicy


@dataclass
class PhaseStats:
    total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


@dataclass
class DiagnosisResult:
    total: int
    correct: int
    by_phase: dict[str, PhaseStats]
    confusion_by_phase: dict[str, Counter[tuple[int, int]]]

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


def choose_simple_bot_action(env: EuchreEnv, obs: Observation, policy: SimpleBotPolicy) -> Action:
    """
    Adapter: ask the old method-based SimpleBotPolicy what it would do in the
    current action-based observation.
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

    raise ValueError(f"No SimpleBot action available for phase {obs.phase!r}")


def diagnose_model(
    model_path: Path,
    n_games: int,
    seed: int,
    max_steps_per_game: int = 1000,
) -> DiagnosisResult:
    policies: list[Policy] = [
        SimpleBotPolicy(),
        SimpleBotPolicy(),
        SimpleBotPolicy(),
        SimpleBotPolicy(),
    ]
    env = EuchreEnv(policies=policies, winning_score=10, seed=seed)

    model_policy = ModelActionPolicy(model_path)
    teacher = SimpleBotPolicy()

    by_phase: dict[str, PhaseStats] = defaultdict(PhaseStats)
    confusion_by_phase: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)

    total = 0
    correct = 0

    for game_index in range(n_games):
        obs = env.reset()
        done = False
        steps = 0

        while not done:
            teacher_action = choose_simple_bot_action(env, obs, teacher)
            model_action = model_policy.choose_action(obs)

            if teacher_action not in obs.legal_actions:
                raise ValueError(
                    f"Teacher chose illegal action {teacher_action}; "
                    f"legal actions were {obs.legal_actions}."
                )
            if model_action not in obs.legal_actions:
                raise ValueError(
                    f"Model chose illegal action {model_action}; "
                    f"legal actions were {obs.legal_actions}."
                )

            teacher_id = action_to_id(teacher_action)
            model_id = action_to_id(model_action)
            is_correct = teacher_id == model_id

            total += 1
            correct += int(is_correct)

            phase_stats = by_phase[obs.phase]
            phase_stats.total += 1
            phase_stats.correct += int(is_correct)

            confusion_by_phase[obs.phase][(teacher_id, model_id)] += 1

            # Advance using the teacher action, so the state distribution matches
            # the SimpleBot data-generation distribution.
            result = env.step(teacher_action)
            obs = result.observation
            done = result.done
            steps += 1

            if steps > max_steps_per_game:
                raise RuntimeError(
                    f"Game {game_index} exceeded {max_steps_per_game} steps; possible loop bug."
                )

        if (game_index + 1) % 100 == 0:
            print(f"Processed {game_index + 1}/{n_games} games")

    return DiagnosisResult(
        total=total,
        correct=correct,
        by_phase=dict(by_phase),
        confusion_by_phase=dict(confusion_by_phase),
    )


def print_diagnosis(result: DiagnosisResult, top_confusions: int) -> None:
    print("\nOverall")
    print("-------")
    print(f"Examples: {result.total}")
    print(f"Correct:  {result.correct}")
    print(f"Accuracy: {100.0 * result.accuracy:.2f}%")

    print("\nAccuracy by phase")
    print("-----------------")
    for phase in sorted(result.by_phase):
        stats = result.by_phase[phase]
        print(
            f"{phase:16s} "
            f"examples={stats.total:7d}  "
            f"correct={stats.correct:7d}  "
            f"accuracy={100.0 * stats.accuracy:6.2f}%"
        )

    if top_confusions <= 0:
        return

    print("\nTop mismatches by phase")
    print("-----------------------")
    for phase in sorted(result.confusion_by_phase):
        mismatches = Counter(
            {
                pair: count
                for pair, count in result.confusion_by_phase[phase].items()
                if pair[0] != pair[1]
            }
        )
        if not mismatches:
            continue

        print(f"\n{phase}")
        for (teacher_id, model_id), count in mismatches.most_common(top_confusions):
            print(
                f"  teacher action {teacher_id:2d} -> model action {model_id:2d}: "
                f"{count} times"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose imitation accuracy by comparing a trained model to SimpleBotPolicy."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/imitation_simple_bot_richer_5k.pt"),
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1000,
        help="Number of teacher-played games to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for deals.",
    )
    parser.add_argument(
        "--top-confusions",
        type=int,
        default=10,
        help="Number of most common mismatches to show per phase.",
    )

    args = parser.parse_args()

    result = diagnose_model(
        model_path=args.model,
        n_games=args.games,
        seed=args.seed,
    )
    print_diagnosis(result, top_confusions=args.top_confusions)


if __name__ == "__main__":
    main()

