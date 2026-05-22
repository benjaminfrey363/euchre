from logic.cards import Suit
from logic.env import (
    CallTrumpAction,
    EuchreEnv,
    OrderUpAction,
    PlayCardAction,
)
from logic.policies import Policy, RandomPolicy, SimpleBotPolicy


def test_random_policies_can_complete_one_game() -> None:
    policies: list[Policy] = [
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
    ]

    env = EuchreEnv(policies=policies, winning_score=10, seed=123)
    result = env.play_game()

    assert result.winner_team in {0, 1}
    assert max(result.final_score) >= 10
    assert result.hands_played > 0


def test_simple_team_beats_random_team_over_many_games() -> None:
    policies: list[Policy] = [
        SimpleBotPolicy(),
        RandomPolicy(),
        SimpleBotPolicy(),
        RandomPolicy(),
    ]

    env = EuchreEnv(policies=policies, winning_score=10, seed=123)
    stats = env.run_many_games(200)

    assert stats.team_wins[0] > stats.team_wins[1]


def test_random_vs_random_is_not_degenerate() -> None:
    policies: list[Policy] = [
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
    ]

    env = EuchreEnv(policies=policies, winning_score=10, seed=123)
    stats = env.run_many_games(200)

    assert stats.team_wins[0] > 0
    assert stats.team_wins[1] > 0


def test_observation_contains_only_current_players_hand() -> None:
    policies: list[Policy] = [
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
    ]

    env = EuchreEnv(policies=policies, winning_score=10, seed=123)
    env.deal()

    obs = env.observation_for_player(0)

    assert obs.player == 0
    assert len(obs.hand) == 5
    assert obs.upcard is not None
    assert obs.phase == "bidding_round_1"


def test_legal_actions_for_bidding_round_1_are_pass_or_order() -> None:
    policies: list[Policy] = [
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
    ]

    env = EuchreEnv(policies=policies, winning_score=10, seed=123)
    env.deal()

    actions = env.legal_actions_for_player(env.current_player)

    assert OrderUpAction(order_up=False) in actions
    assert OrderUpAction(order_up=True) in actions


def test_legal_actions_for_bidding_round_2_exclude_upcard_suit() -> None:
    policies: list[Policy] = [
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
    ]

    env = EuchreEnv(policies=policies, winning_score=10, seed=123)
    env.deal()
    assert env.upcard is not None

    env.phase = "bidding_round_2"
    actions = env.legal_actions_for_player(env.current_player)

    called_suits = {
        action.suit
        for action in actions
        if isinstance(action, CallTrumpAction) and action.suit is not None
    }

    assert env.upcard.suit not in called_suits
    assert called_suits == set(Suit) - {env.upcard.suit}


def test_legal_play_card_actions_are_generated_during_play() -> None:
    policies: list[Policy] = [
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
    ]

    env = EuchreEnv(policies=policies, winning_score=10, seed=123)
    env.deal()
    env.trump = Suit.HEARTS
    env.phase = "play_card"
    env.current_player = 0

    actions = env.legal_actions_for_player(0)

    assert actions
    assert all(isinstance(action, PlayCardAction) for action in actions)

