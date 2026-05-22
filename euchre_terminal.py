
from __future__ import annotations

from logic.policies import Policy, HumanPolicy, SimpleBotPolicy, EuchreGame

def main() -> None:
    policies: list[Policy] = [
        HumanPolicy(),
        SimpleBotPolicy(),
        SimpleBotPolicy(),
        SimpleBotPolicy(),
    ]
    game = EuchreGame(policies=policies, winning_score=10)
    game.play_game()


if __name__ == "__main__":
    main()

