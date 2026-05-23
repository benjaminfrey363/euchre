from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from logic.cards import Card, Suit
from logic.env import EuchreEnv, Observation
from logic.model_policy import ModelActionPolicy
from logic.policies import EuchreGame, Policy, RandomPolicy, SimpleBotPolicy


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

    def choose_order_up(self, game: EuchreGame, player: int, upcard: Card, is_dealer: bool) -> bool:
        obs = self._observation(game, player)
        action = self.action_policy.choose_action(obs)
        from logic.env import OrderUpAction

        if not isinstance(action, OrderUpAction):
            raise TypeError(f"Expected OrderUpAction, got {action!r}")
        return action.order_up

    def choose_trump(self, game: EuchreGame, player: int, forbidden_suit: Suit, is_dealer: bool):
        obs = self._observation(game, player)
        action = self.action_policy.choose_action(obs)
        from logic.env import CallTrumpAction

        if not isinstance(action, CallTrumpAction):
            raise TypeError(f"Expected CallTrumpAction, got {action!r}")
        return action.suit

    def choose_discard(self, game: EuchreGame, player: int):
        obs = self._observation(game, player)
        action = self.action_policy.choose_action(obs)
        from logic.env import DiscardAction

        if not isinstance(action, DiscardAction):
            raise TypeError(f"Expected DiscardAction, got {action!r}")
        return action.card

    def choose_card(self, game: EuchreGame, player: int, legal: list[Card], trick: list[tuple[int, Card]]):
        obs = self._observation(game, player)
        action = self.action_policy.choose_action(obs)
        from logic.env import PlayCardAction

        if not isinstance(action, PlayCardAction):
            raise TypeError(f"Expected PlayCardAction, got {action!r}")
        return action.card


def run_matchup(name: str, policies: list[Policy], n_games: int = 1000) -> None:
    env = EuchreEnv(policies=policies, winning_score=10, seed=123)
    stats = env.run_many_games(n_games)

    print(f"\n{name}")
    print("-" * len(name))
    print(stats)


def main() -> None:
    model_path = Path("models/imitation_simple_bot.pt")
    model_policy: Policy = ActionPolicyAdapter(ModelActionPolicy(model_path))

    run_matchup(
        "Model team vs Random team",
        [
            model_policy,
            RandomPolicy(),
            model_policy,
            RandomPolicy(),
        ],
    )

    run_matchup(
        "Random team vs Model team",
        [
            RandomPolicy(),
            model_policy,
            RandomPolicy(),
            model_policy,
        ],
    )

    run_matchup(
        "Model team vs Simple team",
        [
            model_policy,
            SimpleBotPolicy(),
            model_policy,
            SimpleBotPolicy(),
        ],
    )

    run_matchup(
        "Simple team vs Model team",
        [
            SimpleBotPolicy(),
            model_policy,
            SimpleBotPolicy(),
            model_policy,
        ],
    )


if __name__ == "__main__":
    main()
