from dataclasses import dataclass
from enum import Enum

class Suit(str, Enum):
    CLUBS = "Clubs"
    DIAMONDS = "Diamonds"
    HEARTS = "Hearts"
    SPADES = "Spades"


class Rank(str, Enum):
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"


RED_SUITS = {Suit.HEARTS, Suit.DIAMONDS}
BLACK_SUITS = {Suit.CLUBS, Suit.SPADES}
RANKS = [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]
SUITS = [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]


@dataclass(frozen=True, order=False)
class Card:
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        symbols = {
            Suit.CLUBS: "♣",
            Suit.DIAMONDS: "♦",
            Suit.HEARTS: "♥",
            Suit.SPADES: "♠",
        }
        return f"{self.rank.value}{symbols[self.suit]}"
    

def same_color_suit(suit: Suit) -> Suit:
    if suit == Suit.HEARTS:
        return Suit.DIAMONDS
    if suit == Suit.DIAMONDS:
        return Suit.HEARTS
    if suit == Suit.CLUBS:
        return Suit.SPADES
    return Suit.CLUBS


def is_right_bower(card: Card, trump: Suit) -> bool:
    return card.rank == Rank.JACK and card.suit == trump


def is_left_bower(card: Card, trump: Suit) -> bool:
    return card.rank == Rank.JACK and card.suit == same_color_suit(trump)


def effective_suit(card: Card, trump: Suit) -> Suit:
    """The left bower is treated as a trump-suit card."""
    if is_left_bower(card, trump):
        return trump
    return card.suit


def is_trump(card: Card, trump: Suit) -> bool:
    return effective_suit(card, trump) == trump

def make_deck() -> list[Card]:
    return [Card(rank, suit) for suit in SUITS for rank in RANKS]
