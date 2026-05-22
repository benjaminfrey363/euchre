
import pytest

from logic.cards import (
    Card,
    Rank,
    Suit,
    effective_suit,
    is_left_bower,
    is_right_bower,
    is_trump,
)

from logic.rules import (
    legal_cards,
    trick_winner,
    team_of,
)


def test_right_bower_is_strongest_trump_card():
    trump = Suit.HEARTS
    trick = [
        (0, Card(Rank.ACE, Suit.HEARTS)),
        (1, Card(Rank.JACK, Suit.HEARTS)),
        (2, Card(Rank.JACK, Suit.DIAMONDS)),
        (3, Card(Rank.NINE, Suit.HEARTS)),
    ]

    assert is_right_bower(Card(Rank.JACK, Suit.HEARTS), trump)
    assert trick_winner(trick, trump) == 1


def test_left_bower_counts_as_trump_suit():
    trump = Suit.HEARTS
    left_bower = Card(Rank.JACK, Suit.DIAMONDS)

    assert is_left_bower(left_bower, trump)
    assert is_trump(left_bower, trump)
    assert effective_suit(left_bower, trump) == Suit.HEARTS


def test_left_bower_beats_ace_of_trump_but_loses_to_right_bower():
    trump = Suit.SPADES
    trick = [
        (0, Card(Rank.ACE, Suit.SPADES)),
        (1, Card(Rank.JACK, Suit.CLUBS)),
        (2, Card(Rank.JACK, Suit.SPADES)),
        (3, Card(Rank.KING, Suit.SPADES)),
    ]

    assert trick_winner(trick, trump) == 2


def test_trump_beats_led_suit():
    trump = Suit.CLUBS
    trick = [
        (0, Card(Rank.ACE, Suit.HEARTS)),
        (1, Card(Rank.NINE, Suit.HEARTS)),
        (2, Card(Rank.NINE, Suit.CLUBS)),
        (3, Card(Rank.KING, Suit.HEARTS)),
    ]

    assert trick_winner(trick, trump) == 2


def test_highest_card_of_led_suit_wins_when_no_trump_is_played():
    trump = Suit.CLUBS
    trick = [
        (0, Card(Rank.TEN, Suit.HEARTS)),
        (1, Card(Rank.ACE, Suit.HEARTS)),
        (2, Card(Rank.KING, Suit.HEARTS)),
        (3, Card(Rank.NINE, Suit.SPADES)),
    ]

    assert trick_winner(trick, trump) == 1


def test_off_suit_non_trump_cannot_win_even_if_high_rank():
    trump = Suit.CLUBS
    trick = [
        (0, Card(Rank.NINE, Suit.HEARTS)),
        (1, Card(Rank.ACE, Suit.SPADES)),
        (2, Card(Rank.TEN, Suit.HEARTS)),
        (3, Card(Rank.KING, Suit.DIAMONDS)),
    ]

    assert trick_winner(trick, trump) == 2


def test_legal_cards_must_follow_led_suit_when_possible():
    trump = Suit.SPADES
    hand = [
        Card(Rank.NINE, Suit.HEARTS),
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.KING, Suit.CLUBS),
        Card(Rank.NINE, Suit.DIAMONDS),
        Card(Rank.ACE, Suit.CLUBS),
    ]

    legal = legal_cards(hand, trump, led_suit=Suit.HEARTS)

    assert legal == [
        Card(Rank.NINE, Suit.HEARTS),
        Card(Rank.ACE, Suit.HEARTS),
    ]


def test_legal_cards_all_cards_legal_when_void_in_led_suit():
    trump = Suit.SPADES
    hand = [
        Card(Rank.NINE, Suit.CLUBS),
        Card(Rank.ACE, Suit.CLUBS),
        Card(Rank.KING, Suit.DIAMONDS),
        Card(Rank.NINE, Suit.DIAMONDS),
        Card(Rank.ACE, Suit.SPADES),
    ]

    legal = legal_cards(hand, trump, led_suit=Suit.HEARTS)

    assert legal == hand


def test_left_bower_must_follow_trump_not_its_printed_suit():
    trump = Suit.HEARTS
    left_bower = Card(Rank.JACK, Suit.DIAMONDS)
    hand = [
        left_bower,
        Card(Rank.ACE, Suit.DIAMONDS),
        Card(Rank.NINE, Suit.SPADES),
        Card(Rank.TEN, Suit.CLUBS),
        Card(Rank.KING, Suit.CLUBS),
    ]

    legal_when_hearts_led = legal_cards(hand, trump, led_suit=Suit.HEARTS)
    legal_when_diamonds_led = legal_cards(hand, trump, led_suit=Suit.DIAMONDS)

    assert legal_when_hearts_led == [left_bower]
    assert legal_when_diamonds_led == [Card(Rank.ACE, Suit.DIAMONDS)]


def test_team_assignment():
    assert team_of(0) == team_of(2)
    assert team_of(1) == team_of(3)
    assert team_of(0) != team_of(1)

