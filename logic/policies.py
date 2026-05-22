from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional, Protocol

from logic.rules import *



@dataclass
class HandResult:
    maker_team: int
    tricks_by_team: list[int]
    points_by_team: list[int]


class EuchreGame:
    def __init__(self, policies: list[Policy], winning_score: int = 10, seed: Optional[int] = None):
        if len(policies) != 4:
            raise ValueError("Euchre requires exactly four policies.")
        self.policies = policies
        self.winning_score = winning_score
        self.random = random.Random(seed)
        self.dealer = 0
        self.scores = [0, 0]
        self.hands: list[list[Card]] = [[] for _ in range(4)]
        self.kitty: list[Card] = []
        self.upcard: Optional[Card] = None
        self.trump: Optional[Suit] = None
        self.maker: Optional[int] = None

    def deal(self) -> None:
        deck = make_deck()
        self.random.shuffle(deck)
        self.hands = [sorted(deck[i * 5:(i + 1) * 5], key=lambda c: (c.suit.value, c.rank.value)) for i in range(4)]
        self.upcard = deck[20]
        self.kitty = deck[21:]
        self.trump = None
        self.maker = None

    def bidding(self) -> bool:
        assert self.upcard is not None

        # Round 1: order up the upcard suit.
        player = left_of(self.dealer)
        for _ in range(4):
            is_dealer = player == self.dealer
            if self.policies[player].choose_order_up(self, player, self.upcard, is_dealer):
                self.trump = self.upcard.suit
                self.maker = player
                print(f"Player {player} orders up {self.trump.value}.")
                self.dealer_pickup()
                return True
            print(f"Player {player} passes.")
            player = next_player(player)

        # Round 2: choose any suit except the upcard suit.
        player = left_of(self.dealer)
        for _ in range(4):
            is_dealer = player == self.dealer
            chosen = self.policies[player].choose_trump(self, player, self.upcard.suit, is_dealer)
            if chosen is not None:
                self.trump = chosen
                self.maker = player
                print(f"Player {player} calls {self.trump.value}.")
                return True
            print(f"Player {player} passes.")
            player = next_player(player)

        print("Everyone passed. The hand is redealt with the next dealer.")
        return False

    def dealer_pickup(self) -> None:
        assert self.upcard is not None
        self.hands[self.dealer].append(self.upcard)
        discard = self.policies[self.dealer].choose_discard(self, self.dealer)
        self.hands[self.dealer].remove(discard)
        print(f"Dealer discards one card.")

    def play_hand(self) -> Optional[HandResult]:
        self.deal()
        assert self.upcard is not None
        print("\n" + "=" * 60)
        print(f"Dealer: Player {self.dealer}. Upcard: {self.upcard}")

        if not self.bidding():
            self.dealer = next_player(self.dealer)
            return None

        assert self.trump is not None
        assert self.maker is not None

        print(f"Trump is {self.trump.value}. Maker: Player {self.maker} / Team {team_of(self.maker)}")

        tricks_by_team = [0, 0]
        leader = left_of(self.dealer)

        for trick_number in range(1, 6):
            trick: list[tuple[int, Card]] = []
            print(f"\nTrick {trick_number}. Leader: Player {leader}")

            player = leader
            led_suit: Optional[Suit] = None
            for _ in range(4):
                legal = legal_cards(self.hands[player], self.trump, led_suit)
                card = self.policies[player].choose_card(self, player, legal, trick)
                self.hands[player].remove(card)
                trick.append((player, card))
                if led_suit is None:
                    led_suit = effective_suit(card, self.trump)
                print(f"Player {player} plays {card}.")
                player = next_player(player)

            winner = trick_winner(trick, self.trump)
            tricks_by_team[team_of(winner)] += 1
            leader = winner
            print(f"Player {winner} wins the trick.")

        maker_team = team_of(self.maker)
        maker_tricks = tricks_by_team[maker_team]
        points_by_team = [0, 0]

        if maker_tricks >= 3:
            points_by_team[maker_team] = 2 if maker_tricks == 5 else 1
        else:
            points_by_team[1 - maker_team] = 2

        self.scores[0] += points_by_team[0]
        self.scores[1] += points_by_team[1]

        print("\nHand result:")
        print(f"  Tricks: Team 0 = {tricks_by_team[0]}, Team 1 = {tricks_by_team[1]}")
        print(f"  Points: Team 0 = {points_by_team[0]}, Team 1 = {points_by_team[1]}")
        print(f"  Score:  Team 0 = {self.scores[0]}, Team 1 = {self.scores[1]}")

        self.dealer = next_player(self.dealer)
        return HandResult(maker_team, tricks_by_team, points_by_team)

    def play_game(self) -> None:
        while max(self.scores) < self.winning_score:
            self.play_hand()

        print("\n" + "=" * 60)
        winner = 0 if self.scores[0] > self.scores[1] else 1
        print(f"Team {winner} wins {self.scores[winner]}-{self.scores[1 - winner]}!")





class Policy(Protocol):
    def choose_order_up(
        self,
        game: "EuchreGame",
        player: int,
        upcard: Card,
        is_dealer: bool,
    ) -> bool:
        ...

    def choose_trump(
        self,
        game: "EuchreGame",
        player: int,
        forbidden_suit: Suit,
        is_dealer: bool,
    ) -> Optional[Suit]:
        ...

    def choose_discard(self, game: "EuchreGame", player: int) -> Card:
        ...

    def choose_card(
        self,
        game: "EuchreGame",
        player: int,
        legal: list[Card],
        trick: list[tuple[int, Card]],
    ) -> Card:
        ...


class HumanPolicy:
    def choose_order_up(self, game: "EuchreGame", player: int, upcard: Card, is_dealer: bool) -> bool:
        print(f"\nYour hand: {format_hand(game.hands[player])}")
        dealer_note = " You are the dealer." if is_dealer else ""
        answer = input(f"Upcard is {upcard}.{dealer_note} Order up {upcard.suit.value}? [y/N] ").strip().lower()
        return answer in {"y", "yes"}

    def choose_trump(self, game: "EuchreGame", player: int, forbidden_suit: Suit, is_dealer: bool) -> Optional[Suit]:
        print(f"\nYour hand: {format_hand(game.hands[player])}")
        choices = [suit for suit in SUITS if suit != forbidden_suit]
        print("Choose trump, or pass:")
        for i, suit in enumerate(choices):
            print(f"  [{i}] {suit.value}")
        answer = input("Trump choice [Enter to pass]: ").strip()
        if answer == "":
            return None
        try:
            index = int(answer)
            return choices[index]
        except (ValueError, IndexError):
            print("Invalid choice; passing.")
            return None

    def choose_discard(self, game: "EuchreGame", player: int) -> Card:
        hand = game.hands[player]
        print(f"\nDealer pickup. Your hand is now: {format_hand(hand)}")
        while True:
            answer = input("Choose a card index to discard: ").strip()
            try:
                return hand[int(answer)]
            except (ValueError, IndexError):
                print("Invalid card index.")

    def choose_card(self, game: "EuchreGame", player: int, legal: list[Card], trick: list[tuple[int, Card]]) -> Card:
        print("\nCurrent trick:")
        if trick:
            for p, card in trick:
                print(f"  Player {p}: {card}")
        else:
            print("  You are leading.")

        hand = game.hands[player]
        print(f"Your hand: {format_hand(hand)}")
        print(f"Legal cards: {'  '.join(str(card) for card in legal)}")

        while True:
            answer = input("Choose a card index to play: ").strip()
            try:
                card = hand[int(answer)]
                if card in legal:
                    return card
                print("That card is not legal. You must follow suit if possible.")
            except (ValueError, IndexError):
                print("Invalid card index.")


class SimpleBotPolicy:
    """
    A deliberately basic bot.

    Later we will replace this with stronger heuristics, Monte Carlo search,
    or a learned policy.
    """

    def choose_order_up(self, game: "EuchreGame", player: int, upcard: Card, is_dealer: bool) -> bool:
        hand = game.hands[player]
        candidate_trump = upcard.suit
        trump_count = count_potential_trump(hand, candidate_trump)
        if is_dealer:
            # Dealer will pick up the upcard, so be slightly more willing.
            trump_count += 1
        return trump_count >= 3 or has_bower(hand, candidate_trump) and trump_count >= 2

    def choose_trump(self, game: "EuchreGame", player: int, forbidden_suit: Suit, is_dealer: bool) -> Optional[Suit]:
        hand = game.hands[player]
        options = [suit for suit in SUITS if suit != forbidden_suit]
        scored = []
        for suit in options:
            score = count_potential_trump(hand, suit) + (2 if has_bower(hand, suit) else 0)
            scored.append((score, suit))
        best_score, best_suit = max(scored)
        threshold = 4 if not is_dealer else 3
        return best_suit if best_score >= threshold else None

    def choose_discard(self, game: "EuchreGame", player: int) -> Card:
        trump = game.trump
        assert trump is not None
        hand = game.hands[player]
        # Discard weakest card relative to trump, with no led suit.
        return min(hand, key=lambda card: card_strength(card, trump, effective_suit(card, trump)))

    def choose_card(self, game: "EuchreGame", player: int, legal: list[Card], trick: list[tuple[int, Card]]) -> Card:
        trump = game.trump
        assert trump is not None

        if not trick:
            # Lead strongest legal card.
            return max(legal, key=lambda card: card_strength(card, trump, effective_suit(card, trump)))

        led_suit = effective_suit(trick[0][1], trump)
        current_winner = trick_winner(trick, trump)
        current_best_card = next(card for p, card in trick if p == current_winner)
        current_best_strength = card_strength(current_best_card, trump, led_suit)

        winning_cards = [card for card in legal if card_strength(card, trump, led_suit) > current_best_strength]

        # If partner is currently winning, conserve strength.
        if team_of(current_winner) == team_of(player):
            return min(legal, key=lambda card: card_strength(card, trump, led_suit))

        # Otherwise play the cheapest winning card if possible.
        if winning_cards:
            return min(winning_cards, key=lambda card: card_strength(card, trump, led_suit))

        # Cannot win; dump weakest legal card.
        return min(legal, key=lambda card: card_strength(card, trump, led_suit))



# Random policy used as baseline
class RandomPolicy:
    def choose_order_up(
        self,
        game: EuchreGame,
        player: int,
        upcard: Card,
        is_dealer: bool,
    ) -> bool:
        return game.random.random() < 0.25

    def choose_trump(
        self,
        game: EuchreGame,
        player: int,
        forbidden_suit: Suit,
        is_dealer: bool,
    ) -> Optional[Suit]:
        choices = [suit for suit in SUITS if suit != forbidden_suit]
        if game.random.random() < 0.25:
            return game.random.choice(choices)
        return None

    def choose_discard(self, game: EuchreGame, player: int) -> Card:
        return game.random.choice(game.hands[player])

    def choose_card(
        self,
        game: EuchreGame,
        player: int,
        legal: list[Card],
        trick: list[tuple[int, Card]],
    ) -> Card:
        return game.random.choice(legal)

