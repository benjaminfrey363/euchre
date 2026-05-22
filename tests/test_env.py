from logic.env import EuchreEnv
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
    