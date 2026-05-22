
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional, Protocol

from logic.cards import *


def card_strength(card: Card, trump: Suit, led_suit: Suit) -> int:
    """
    Larger values are stronger. Cards outside trump and the led suit cannot win.
    """
    if is_right_bower(card, trump):
        return 200
    if is_left_bower(card, trump):
        return 199

    non_bower_rank_value = {
        Rank.ACE: 6,
        Rank.KING: 5,
        Rank.QUEEN: 4,
        Rank.JACK: 3,
        Rank.TEN: 2,
        Rank.NINE: 1,
    }

    if is_trump(card, trump):
        return 100 + non_bower_rank_value[card.rank]

    if effective_suit(card, trump) == led_suit:
        return 10 + non_bower_rank_value[card.rank]

    return 0


def team_of(player_index: int) -> int:
    """Players 0 and 2 are partners; players 1 and 3 are partners."""
    return player_index % 2


def next_player(player_index: int) -> int:
    return (player_index + 1) % 4


def left_of(player_index: int) -> int:
    return next_player(player_index)


def legal_cards(hand: list[Card], trump: Suit, led_suit: Optional[Suit]) -> list[Card]:
    if led_suit is None:
        return hand[:]

    following = [card for card in hand if effective_suit(card, trump) == led_suit]
    return following if following else hand[:]


def trick_winner(trick: list[tuple[int, Card]], trump: Suit) -> int:
    if not trick:
        raise ValueError("Cannot determine winner of an empty trick.")

    led_suit = effective_suit(trick[0][1], trump)
    return max(trick, key=lambda played: card_strength(played[1], trump, led_suit))[0]


def format_hand(hand: list[Card]) -> str:
    return "  ".join(f"[{i}] {card}" for i, card in enumerate(hand))


def count_potential_trump(hand: list[Card], trump: Suit) -> int:
    return sum(1 for card in hand if is_trump(card, trump))


def has_bower(hand: list[Card], trump: Suit) -> bool:
    return any(is_right_bower(card, trump) or is_left_bower(card, trump) for card in hand)

