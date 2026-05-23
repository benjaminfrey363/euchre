from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, cast

from logic.action_encoding import action_to_id
from logic.env import (
    Action,
    CallTrumpAction,
    DiscardAction,
    EuchreEnv,
    Observation,
    OrderUpAction,
    PlayCardAction,
    StepResult,
)
from logic.policies import EuchreGame, SimpleBotPolicy
from logic.rules import team_of


@dataclass(frozen=True)
class ActionEvaluation:
    action: Action
    average_reward: float
    rollouts: int


@dataclass(frozen=True)
class MonteCarloDecision:
    chosen_action: Action
    evaluations: list[ActionEvaluation]


class SimpleBotActionAdapter:
    """
    Adapter from SimpleBotPolicy's old method-based interface to the new
    action-based interface.
    """

    def __init__(self):
        self.policy = SimpleBotPolicy()

    def choose_action(self, env: EuchreEnv, obs: Observation) -> Action:
        player = obs.player
        game_view = cast(EuchreGame, env)

        if obs.phase == "bidding_round_1":
            assert obs.upcard is not None
            return OrderUpAction(
                order_up=self.policy.choose_order_up(
                    game_view,
                    player,
                    obs.upcard,
                    is_dealer=(player == obs.dealer),
                )
            )

        if obs.phase == "bidding_round_2":
            assert obs.upcard is not None
            return CallTrumpAction(
                suit=self.policy.choose_trump(
                    game_view,
                    player,
                    forbidden_suit=obs.upcard.suit,
                    is_dealer=(player == obs.dealer),
                )
            )

        if obs.phase == "discard":
            return DiscardAction(card=self.policy.choose_discard(game_view, player))

        if obs.phase == "play_card":
            legal_cards = [
                action.card
                for action in obs.legal_actions
                if isinstance(action, PlayCardAction)
            ]
            return PlayCardAction(
                card=self.policy.choose_card(
                    game_view,
                    player,
                    legal_cards,
                    list(obs.trick),
                )
            )

        raise ValueError(f"Cannot choose SimpleBot action during phase {obs.phase!r}.")


class OracleMonteCarloPolicy:
    """
    Oracle hand-level Monte Carlo policy.

    This policy evaluates each legal action by cloning the full environment,
    applying that candidate action, and rolling out the rest of the hand.

    Important: this is an ORACLE policy because it has access to the complete
    EuchreEnv state, including hidden cards in other players' hands. This is not
    a fair deployable policy yet. It is a policy-improvement tool for generating
    stronger labels and checking whether rollouts can beat SimpleBotPolicy.

    The reward is hand-level from the acting player's team perspective:

        reward = points_for_own_team - points_for_opponent_team

    For example:
        +1 if own team makes one point
        +2 if own team marches or euchres
        -2 if own team gets euchred
    """

    def __init__(
        self,
        rollouts_per_action: int = 1,
        max_steps_per_rollout: int = 200,
    ):
        if rollouts_per_action <= 0:
            raise ValueError("rollouts_per_action must be positive.")
        self.rollouts_per_action = rollouts_per_action
        self.max_steps_per_rollout = max_steps_per_rollout
        self.rollout_policy = SimpleBotActionAdapter()

    def choose_action(self, env: EuchreEnv) -> Action:
        return self.evaluate_actions(env).chosen_action

    def evaluate_actions(self, env: EuchreEnv) -> MonteCarloDecision:
        obs = env.observation_for_player(env.current_player)
        if not obs.legal_actions:
            raise ValueError(f"No legal actions available during phase {obs.phase!r}.")

        acting_player = obs.player
        acting_team = team_of(acting_player)

        evaluations: list[ActionEvaluation] = []
        for action in obs.legal_actions:
            total_reward = 0.0
            for _ in range(self.rollouts_per_action):
                rollout_env = deepcopy(env)
                total_reward += self.evaluate_action_once(
                    rollout_env,
                    action,
                    acting_team,
                )

            evaluations.append(
                ActionEvaluation(
                    action=action,
                    average_reward=total_reward / self.rollouts_per_action,
                    rollouts=self.rollouts_per_action,
                )
            )

        # Deterministic tie-breaker by action ID keeps results reproducible.
        best = max(
            evaluations,
            key=lambda evaluation: (evaluation.average_reward, -action_to_id(evaluation.action)),
        )
        return MonteCarloDecision(chosen_action=best.action, evaluations=evaluations)

    def evaluate_action_once(
        self,
        env: EuchreEnv,
        first_action: Action,
        acting_team: int,
    ) -> float:
        result = env.step(first_action)
        maybe_reward = self.reward_from_step_result(result, acting_team)
        if maybe_reward is not None:
            return maybe_reward

        steps = 0
        while not result.done and steps < self.max_steps_per_rollout:
            obs = result.observation
            action = self.rollout_policy.choose_action(env, obs)
            result = env.step(action)

            maybe_reward = self.reward_from_step_result(result, acting_team)
            if maybe_reward is not None:
                return maybe_reward

            steps += 1

        raise RuntimeError(
            f"Rollout exceeded {self.max_steps_per_rollout} steps without finishing a hand."
        )

    @staticmethod
    def reward_from_step_result(result: StepResult, acting_team: int) -> Optional[float]:
        hand_result = result.info.get("hand_result")
        if hand_result is None:
            return None

        # HandResult has points_by_team: list[int]. We keep this duck-typed to
        # avoid importing the concrete dataclass from policies.py here.
        points_by_team = hand_result.points_by_team  # type: ignore[attr-defined]
        return float(points_by_team[acting_team] - points_by_team[1 - acting_team])

