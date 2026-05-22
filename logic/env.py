from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional, cast

from logic.cards import Card, Suit, make_deck, effective_suit
from logic.rules import legal_cards, next_player, left_of, team_of, trick_winner
from logic.policies import HandResult, Policy, EuchreGame

from collections.abc import Sequence

@dataclass(frozen=True)
class GameResult:
    winner_team: int
    final_score: tuple[int, int]
    hands_played: int


@dataclass(frozen=True)
class SimulationStats:
    games_played: int
    team_wins: tuple[int, int]
    average_score: tuple[float, float]
    average_hands_per_game: float


class EuchreEnv:
    """
    Headless Euchre simulator for bot-vs-bot games.

    This class deliberately has no input(), no print(), and no GUI code.
    It is the bridge between the current playable game and future ML training.

    Current limitations:
    - Four-player fixed-partner Euchre.
    - No going alone yet.
    - Policies must choose legal actions themselves, but this environment
      enforces legal card play through legal_cards().
    - If everyone passes both bidding rounds, the hand is redealt with the
      next dealer and no points are awarded.

    Note: Policy methods are currently typed against EuchreGame in
    logic.policies. This environment exposes the same fields the policies use,
    so we cast self to EuchreGame when calling policies. Later, we should replace
    that with a smaller GameView Protocol.
    """

    def __init__(
        self,
        policies: Sequence[Policy],
        winning_score: int = 10,
        seed: Optional[int] = None,
    ):
        if len(policies) != 4:
            raise ValueError("EuchreEnv requires exactly four policies.")

        self.policies = list(policies)
        self.winning_score = winning_score
        self.random = random.Random(seed)

        self.dealer = 0
        self.scores = [0, 0]

        self.hands: list[list[Card]] = [[] for _ in range(4)]
        self.kitty: list[Card] = []
        self.upcard: Optional[Card] = None
        self.trump: Optional[Suit] = None
        self.maker: Optional[int] = None

    def reset_game(self) -> None:
        self.dealer = 0
        self.scores = [0, 0]
        self.hands = [[] for _ in range(4)]
        self.kitty = []
        self.upcard = None
        self.trump = None
        self.maker = None

    def deal(self) -> None:
        deck = make_deck()
        self.random.shuffle(deck)

        self.hands = [deck[i * 5 : (i + 1) * 5] for i in range(4)]
        self.upcard = deck[20]
        self.kitty = deck[21:]
        self.trump = None
        self.maker = None

    def bid_hand(self) -> bool:
        """
        Returns True if trump is chosen, False if all players pass.
        """
        assert self.upcard is not None
        upcard = self.upcard

        # Round 1: order up the upcard suit.
        player = left_of(self.dealer)
        for _ in range(4):
            is_dealer = player == self.dealer
            wants_order = self.policies[player].choose_order_up(
                cast(EuchreGame, self),
                player,
                upcard,
                is_dealer,
            )
            if wants_order:
                self.trump = upcard.suit
                self.maker = player
                self.dealer_pickup()
                return True
            player = next_player(player)

        # Round 2: choose any suit except the upcard suit.
        player = left_of(self.dealer)
        for _ in range(4):
            is_dealer = player == self.dealer
            chosen = self.policies[player].choose_trump(
                cast(EuchreGame, self),
                player,
                upcard.suit,
                is_dealer,
            )
            if chosen is not None:
                self.trump = chosen
                self.maker = player
                return True
            player = next_player(player)

        return False

    def dealer_pickup(self) -> None:
        assert self.upcard is not None
        assert self.trump is not None

        self.hands[self.dealer].append(self.upcard)
        discard = self.policies[self.dealer].choose_discard(cast(EuchreGame, self), self.dealer)

        if discard not in self.hands[self.dealer]:
            raise ValueError(f"Dealer policy tried to discard a card not in hand: {discard}")

        self.hands[self.dealer].remove(discard)

    def play_tricks(self) -> list[int]:
        assert self.trump is not None
        trump = self.trump

        tricks_by_team = [0, 0]
        leader = left_of(self.dealer)

        for _ in range(5):
            trick: list[tuple[int, Card]] = []
            led_suit: Optional[Suit] = None
            player = leader

            for _ in range(4):
                legal = legal_cards(self.hands[player], trump, led_suit)
                card = self.policies[player].choose_card(
                    cast(EuchreGame, self),
                    player,
                    legal,
                    trick,
                )

                if card not in legal:
                    raise ValueError(
                        f"Policy for player {player} tried to play illegal card {card}; "
                        f"legal cards were {legal}."
                    )

                self.hands[player].remove(card)
                trick.append((player, card))

                if led_suit is None:
                    led_suit = effective_suit(card, trump)

                player = next_player(player)

            winner = trick_winner(trick, trump)
            tricks_by_team[team_of(winner)] += 1
            leader = winner

        return tricks_by_team

    def score_hand(self, tricks_by_team: list[int]) -> HandResult:
        assert self.maker is not None
        maker_team = team_of(self.maker)
        maker_tricks = tricks_by_team[maker_team]
        points_by_team = [0, 0]

        if maker_tricks >= 3:
            points_by_team[maker_team] = 2 if maker_tricks == 5 else 1
        else:
            points_by_team[1 - maker_team] = 2

        self.scores[0] += points_by_team[0]
        self.scores[1] += points_by_team[1]

        return HandResult(
            maker_team=maker_team,
            tricks_by_team=tricks_by_team,
            points_by_team=points_by_team,
        )

    def play_hand(self) -> Optional[HandResult]:
        self.deal()

        trump_chosen = self.bid_hand()
        if not trump_chosen:
            self.dealer = next_player(self.dealer)
            return None

        tricks_by_team = self.play_tricks()
        result = self.score_hand(tricks_by_team)
        self.dealer = next_player(self.dealer)
        return result

    def play_game(self) -> GameResult:
        self.reset_game()
        hands_played = 0

        while max(self.scores) < self.winning_score:
            self.play_hand()
            hands_played += 1

        winner_team = 0 if self.scores[0] > self.scores[1] else 1
        return GameResult(
            winner_team=winner_team,
            final_score=(self.scores[0], self.scores[1]),
            hands_played=hands_played,
        )

    def run_many_games(self, n_games: int) -> SimulationStats:
        if n_games <= 0:
            raise ValueError("n_games must be positive.")

        wins = [0, 0]
        total_scores = [0, 0]
        total_hands = 0

        for _ in range(n_games):
            result = self.play_game()
            wins[result.winner_team] += 1
            total_scores[0] += result.final_score[0]
            total_scores[1] += result.final_score[1]
            total_hands += result.hands_played

        return SimulationStats(
            games_played=n_games,
            team_wins=(wins[0], wins[1]),
            average_score=(total_scores[0] / n_games, total_scores[1] / n_games),
            average_hands_per_game=total_hands / n_games,
        )
