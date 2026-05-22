from logic.env import EuchreEnv
from logic.policies import Policy, SimpleBotPolicy


def main() -> None:
    policies = [
        SimpleBotPolicy(),
        SimpleBotPolicy(),
        SimpleBotPolicy(),
        SimpleBotPolicy(),
    ]

    env = EuchreEnv(policies=policies, winning_score=10, seed=123)

    one_game = env.play_game()
    print("One game:")
    print(one_game)

    stats = env.run_many_games(1000)
    print("\n1000-game simulation:")
    print(stats)


if __name__ == "__main__":
    main()
