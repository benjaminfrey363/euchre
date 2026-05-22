from logic.env import EuchreEnv
from logic.policies import Policy, RandomPolicy, SimpleBotPolicy


def run_matchup(name: str, policies: list[Policy], n_games: int = 1000) -> None:
    env = EuchreEnv(policies=policies, winning_score=10, seed=123)
    stats = env.run_many_games(n_games)

    print(f"\n{name}")
    print("-" * len(name))
    print(stats)


def main() -> None:
    run_matchup(
        "Simple vs Simple",
        [
            SimpleBotPolicy(),
            SimpleBotPolicy(),
            SimpleBotPolicy(),
            SimpleBotPolicy(),
        ],
    )

    run_matchup(
        "Simple team vs Random team",
        [
            SimpleBotPolicy(),
            RandomPolicy(),
            SimpleBotPolicy(),
            RandomPolicy(),
        ],
    )

    run_matchup(
        "Random team vs Simple team",
        [
            RandomPolicy(),
            SimpleBotPolicy(),
            RandomPolicy(),
            SimpleBotPolicy(),
        ],
    )

    run_matchup(
        "Random vs Random",
        [
            RandomPolicy(),
            RandomPolicy(),
            RandomPolicy(),
            RandomPolicy(),
        ],
    )


if __name__ == "__main__":
    main()