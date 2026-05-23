from __future__ import annotations

from logic.cards import Card, Rank, Suit
from logic.env import (
    Action,
    CallTrumpAction,
    DiscardAction,
    Observation,
    OrderUpAction,
    PlayCardAction,
)


"""
Fixed action encoding for ML agents.

The goal is to convert symbolic actions like

    PlayCardAction(Card(Rank.ACE, Suit.SPADES))

into integer IDs that a model can output.

Current action space:

    0       pass in bidding round 1
    1       order up in bidding round 1

    10      pass in bidding round 2
    11-14   call trump suit

    20-43   discard one of the 24 Euchre cards
    44-67   play one of the 24 Euchre cards

Total fixed action space size: 68.

Not every action ID is legal in every state. Use legal_action_ids(observation)
to get the currently legal IDs.
"""


ACTION_SPACE_SIZE = 68

ORDER_UP_PASS_ID = 0
ORDER_UP_TRUE_ID = 1

CALL_TRUMP_PASS_ID = 10
CALL_TRUMP_OFFSET = 11

DISCARD_OFFSET = 20
PLAY_CARD_OFFSET = 44


SUIT_ORDER: tuple[Suit, ...] = (
    Suit.CLUBS,
    Suit.DIAMONDS,
    Suit.HEARTS,
    Suit.SPADES,
)

RANK_ORDER: tuple[Rank, ...] = (
    Rank.NINE,
    Rank.TEN,
    Rank.JACK,
    Rank.QUEEN,
    Rank.KING,
    Rank.ACE,
)


_CARD_TO_INDEX: dict[Card, int] = {
    Card(rank=rank, suit=suit): suit_index * len(RANK_ORDER) + rank_index
    for suit_index, suit in enumerate(SUIT_ORDER)
    for rank_index, rank in enumerate(RANK_ORDER)
}

_INDEX_TO_CARD: dict[int, Card] = {index: card for card, index in _CARD_TO_INDEX.items()}

_SUIT_TO_INDEX: dict[Suit, int] = {suit: index for index, suit in enumerate(SUIT_ORDER)}
_INDEX_TO_SUIT: dict[int, Suit] = {index: suit for suit, index in _SUIT_TO_INDEX.items()}


def card_to_index(card: Card) -> int:
    return _CARD_TO_INDEX[card]


def index_to_card(index: int) -> Card:
    if index not in _INDEX_TO_CARD:
        raise ValueError(f"Invalid Euchre card index: {index}")
    return _INDEX_TO_CARD[index]


def suit_to_index(suit: Suit) -> int:
    return _SUIT_TO_INDEX[suit]


def index_to_suit(index: int) -> Suit:
    if index not in _INDEX_TO_SUIT:
        raise ValueError(f"Invalid suit index: {index}")
    return _INDEX_TO_SUIT[index]


def action_to_id(action: Action) -> int:
    if isinstance(action, OrderUpAction):
        return ORDER_UP_TRUE_ID if action.order_up else ORDER_UP_PASS_ID

    if isinstance(action, CallTrumpAction):
        if action.suit is None:
            return CALL_TRUMP_PASS_ID
        return CALL_TRUMP_OFFSET + suit_to_index(action.suit)

    if isinstance(action, DiscardAction):
        return DISCARD_OFFSET + card_to_index(action.card)

    if isinstance(action, PlayCardAction):
        return PLAY_CARD_OFFSET + card_to_index(action.card)

    raise TypeError(f"Unknown action type: {action!r}")


def id_to_action(action_id: int, observation: Observation) -> Action:
    """
    Convert an action ID into a symbolic action, using the current observation.

    The observation is included because legality is phase-dependent. For example,
    action ID 1 means order up, but that is only meaningful in bidding round 1.
    """
    if action_id < 0 or action_id >= ACTION_SPACE_SIZE:
        raise ValueError(f"Action ID {action_id} is outside action space size {ACTION_SPACE_SIZE}.")

    if observation.phase == "bidding_round_1":
        if action_id == ORDER_UP_PASS_ID:
            return OrderUpAction(order_up=False)
        if action_id == ORDER_UP_TRUE_ID:
            return OrderUpAction(order_up=True)
        raise ValueError(f"Action ID {action_id} is not valid for bidding round 1.")

    if observation.phase == "bidding_round_2":
        if action_id == CALL_TRUMP_PASS_ID:
            return CallTrumpAction(suit=None)
        if CALL_TRUMP_OFFSET <= action_id < CALL_TRUMP_OFFSET + len(SUIT_ORDER):
            return CallTrumpAction(suit=index_to_suit(action_id - CALL_TRUMP_OFFSET))
        raise ValueError(f"Action ID {action_id} is not valid for bidding round 2.")

    if observation.phase == "discard":
        if DISCARD_OFFSET <= action_id < DISCARD_OFFSET + len(_INDEX_TO_CARD):
            return DiscardAction(card=index_to_card(action_id - DISCARD_OFFSET))
        raise ValueError(f"Action ID {action_id} is not valid for discard phase.")

    if observation.phase == "play_card":
        if PLAY_CARD_OFFSET <= action_id < PLAY_CARD_OFFSET + len(_INDEX_TO_CARD):
            return PlayCardAction(card=index_to_card(action_id - PLAY_CARD_OFFSET))
        raise ValueError(f"Action ID {action_id} is not valid for play-card phase.")

    raise ValueError(f"No actions are valid during phase {observation.phase!r}.")


def legal_action_ids(observation: Observation) -> tuple[int, ...]:
    return tuple(action_to_id(action) for action in observation.legal_actions)


def action_mask(observation: Observation) -> list[int]:
    """
    Return a 0/1 mask over the fixed action space.

    This is useful for ML models: invalid actions can be masked out before
    sampling or taking argmax.
    """
    mask = [0] * ACTION_SPACE_SIZE
    for action_id in legal_action_ids(observation):
        mask[action_id] = 1
    return mask


def id_to_legal_action(action_id: int, observation: Observation) -> Action:
    """
    Convert an action ID to an action and verify that it is currently legal.
    """
    action = id_to_action(action_id, observation)
    if action not in observation.legal_actions:
        raise ValueError(
            f"Action ID {action_id} maps to illegal action {action}; "
            f"legal actions are {observation.legal_actions}."
        )
    return action

