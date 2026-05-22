from logic.env import EuchreEnv, RandomActionPolicy
from logic.policies import Policy, RandomPolicy


def main() -> None:
    policies: list[Policy] = [
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
    ]

    env = EuchreEnv(policies=policies, winning_score=10, seed=123)
    action_policy = RandomActionPolicy(seed=456)

    obs = env.reset()
    done = False
    steps = 0

    while not done:
        action = action_policy.choose_action(obs)
        result = env.step(action)

        if steps < 40:
            print(f"Step {steps}")
            print(f"  Player:        {obs.player}")
            print(f"  Phase:         {obs.phase}")
            print(f"  Action:        {action}")
            print(f"  Score before:  {obs.scores}")
            print(f"  Score after:   {result.observation.scores}")

            if "trick_winner" in result.info:
                print(f"  Trick winner:  {result.info['trick_winner']}")

            if "hand_result" in result.info:
                print(f"  Hand result:   {result.info['hand_result']}")

            print()

        obs = result.observation
        done = result.done
        steps += 1

    print(f"Finished in {steps} steps")
    print(f"Final score: {env.scores}")


if __name__ == "__main__":
    main()
