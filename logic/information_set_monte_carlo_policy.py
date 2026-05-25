from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Optional

from logic.action_encoding import action_to_id
from logic.cards import Card, make_deck, Suit, effective_suit
from logic.env import Action, EuchreEnv, Observation, PlayCardAction, StepResult
from logic.monte_carlo_policy import SimpleBotActionAdapter
from logic.rules import team_of


@dataclass(frozen=True)
class InformationSetActionEvaluation:
    action: Action
    average_reward: float
    samples: int


@dataclass(frozen=True)
class InformationSetDecision:
    chosen_action: Action
    evaluations: list[InformationSetActionEvaluation]


class InformationSetMonteCarloPolicy:
    """
    Information-set Monte Carlo policy for Euchre.

    This is the non-oracle version of the Monte Carlo idea. Instead of rolling
    out from the true hidden hands, it samples possible hidden hands consistent
    with the acting player's observation, evaluates each legal action across
    those sampled worlds, and chooses the action with best average hand reward.

    Current scope:
    - Intended primarily for play_card decisions.
    - For non-play phases, it falls back to SimpleBotActionAdapter.
    - Rollouts after the candidate action use SimpleBotActionAdapter.
    - Reward is hand-level points from the acting player's team perspective.

    This is still an approximation, but unlike OracleMonteCarloPolicy it does
    not intentionally use the true identities of hidden cards.
    """

    def __init__(
        self,
        samples_per_action: int = 10,
        max_steps_per_rollout: int = 200,
        seed: Optional[int] = None,
    ):
        if samples_per_action <= 0:
            raise ValueError("samples_per_action must be positive.")

        self.samples_per_action = samples_per_action
        self.max_steps_per_rollout = max_steps_per_rollout
        self.random = random.Random(seed)
        self.rollout_policy = SimpleBotActionAdapter()

    def choose_action(self, env: EuchreEnv) -> Action:
        obs = env.observation_for_player(env.current_player)

        # Start conservatively: only use information-set MC for card play.
        # Bidding/discard remain SimpleBot-style until we implement sampled-hand
        # bidding evaluation carefully.
        if obs.phase != "play_card":
            return self.rollout_policy.choose_action(env, obs)

        return self.evaluate_actions(env).chosen_action

    def evaluate_actions(self, env: EuchreEnv) -> InformationSetDecision:
        obs = env.observation_for_player(env.current_player)
        if obs.phase != "play_card":
            raise ValueError(f"Information-set MC evaluation expected play_card, got {obs.phase!r}.")
        if not obs.legal_actions:
            raise ValueError("No legal actions available.")

        acting_team = team_of(obs.player)
        evaluations: list[InformationSetActionEvaluation] = []

        for action in obs.legal_actions:
            total_reward = 0.0
            for _ in range(self.samples_per_action):
                sampled_env = self.sample_compatible_env(env, obs)
                total_reward += self.evaluate_action_once(sampled_env, action, acting_team)

            evaluations.append(
                InformationSetActionEvaluation(
                    action=action,
                    average_reward=total_reward / self.samples_per_action,
                    samples=self.samples_per_action,
                )
            )

        # Deterministic tie-breaker keeps runs more stable.
        best = max(
            evaluations,
            key=lambda evaluation: (evaluation.average_reward, -action_to_id(evaluation.action)),
        )
        return InformationSetDecision(chosen_action=best.action, evaluations=evaluations)



    def sample_hidden_hands_unconstrained(
        self,
        unknown_cards: list[Card],
        hidden_players: list[int],
        hand_sizes: list[int],
    ) -> dict[int, list[Card]]:
        cards = list(unknown_cards)
        self.random.shuffle(cards)

        assignments: dict[int, list[Card]] = {}
        cursor = 0

        for player in hidden_players:
            size = hand_sizes[player]
            assignments[player] = cards[cursor : cursor + size]
            cursor += size

        return assignments



    def sample_hidden_hands_with_void_constraints(
        self,
        unknown_cards: list[Card],
        hidden_players: list[int],
        hand_sizes: list[int],
        void_suits: tuple[tuple[Suit, ...], ...],
        trump: Optional[Suit],
        max_attempts: int = 100,
    ) -> dict[int, list[Card]]:
        """
        Assign unknown cards to hidden players while respecting known void suits.

        If constraints are too tight, fall back to unconstrained random assignment.
        This keeps rollouts robust even if our inferred constraints become
        inconsistent because of a bug or edge case.
        """
        if trump is None:
            return self.sample_hidden_hands_unconstrained(
                unknown_cards=unknown_cards,
                hidden_players=hidden_players,
                hand_sizes=hand_sizes,
            )

        for _ in range(max_attempts):
            cards = list(unknown_cards)
            self.random.shuffle(cards)

            assignments: dict[int, list[Card]] = {player: [] for player in hidden_players}

            success = True
            for player in hidden_players:
                needed = hand_sizes[player]
                forbidden_suits = set(void_suits[player])

                chosen: list[Card] = []
                remaining: list[Card] = []

                for card in cards:
                    if len(chosen) < needed and effective_suit(card, trump) not in forbidden_suits:
                        chosen.append(card)
                    else:
                        remaining.append(card)

                if len(chosen) != needed:
                    success = False
                    break

                assignments[player] = chosen
                cards = remaining

            if success:
                return assignments

        # Fallback: do not crash training/evaluation because of one impossible sample.
        return self.sample_hidden_hands_unconstrained(
            unknown_cards=unknown_cards,
            hidden_players=hidden_players,
            hand_sizes=hand_sizes,
        )


    def sample_compatible_env(self, env: EuchreEnv, obs: Observation) -> EuchreEnv:
        """
        Clone env and resample hidden hands.

        We preserve:
        - acting player's hand
        - public trick state
        - played cards
        - scores, trump, maker, dealer, current player, etc.
        - hand sizes for hidden players
        - void-suit information inferred from earlier plays

        We randomize:
        - card identities in other players' current hands
        - remaining kitty/discard pool

        This avoids using true hidden card identities and respects known voids.
        """
        sampled_env = deepcopy(env)
        acting_player = obs.player

        known_cards = set(obs.hand)
        known_cards.update(obs.played_cards)
        known_cards.update(card for _, card in obs.trick)

        deck = make_deck()
        unknown_cards = [card for card in deck if card not in known_cards]
        self.random.shuffle(unknown_cards)

        hand_sizes = [len(env.hands[player]) for player in range(4)]
        sampled_env.hands[acting_player] = list(obs.hand)

        hidden_players = [player for player in range(4) if player != acting_player]

        hidden_assignments = self.sample_hidden_hands_with_void_constraints(
            unknown_cards=unknown_cards,
            hidden_players=hidden_players,
            hand_sizes=hand_sizes,
            void_suits=obs.void_suits,
            trump=obs.trump,
        )

        used_cards: set[Card] = set()
        for player, hand in hidden_assignments.items():
            sampled_env.hands[player] = hand
            used_cards.update(hand)

        sampled_env.kitty = [card for card in unknown_cards if card not in used_cards]
        return sampled_env

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

        points_by_team = hand_result.points_by_team  # type: ignore[attr-defined]
        return float(points_by_team[acting_team] - points_by_team[1 - acting_team])

