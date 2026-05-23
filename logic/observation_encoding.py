from __future__ import annotations

from dataclasses import dataclass

from logic.action_encoding import ACTION_SPACE_SIZE, action_mask, card_to_index, suit_to_index
from logic.cards import Card, Suit
from logic.env import Observation
from logic.rules import team_of


"""
Numeric observation encoding for ML agents.

This file converts a symbolic Observation into a fixed-length list[float].
The encoding is intentionally simple and explicit rather than compact.

Initial feature layout:

    0-23      current player's hand cards, one-hot over 24 cards
    24-47     current trick cards, one-hot over 24 cards
    48-71     upcard, one-hot over 24 cards, all zero if no upcard
    72-75     trump suit, one-hot over 4 suits, all zero if trump not chosen
    76-79     dealer, one-hot over 4 players
    80-83     maker, one-hot over 4 players, all zero if maker not chosen
    84-87     current observation player, one-hot over 4 players
    88-91     scores normalized by 10: [team0, team1, own_team, opponent_team]
    92-95     tricks this hand normalized by 5: [team0, team1, own_team, opponent_team]
    96-101    phase, one-hot over known phases
    102-169   legal action mask over fixed 68-action space

Total feature length: 170.

This is enough to train a first simple model. Later, we may add richer features:
    - position relative to dealer
    - who played each trick card
    - cards already seen/played
    - partner/opponent identity features
    - bidding history
    - known voids / inferred information
"""


PHASE_ORDER: tuple[str, ...] = (
    "not_started",
    "bidding_round_1",
    "bidding_round_2",
    "discard",
    "play_card",
    "hand_over",
)


HAND_OFFSET = 0
TRICK_OFFSET = HAND_OFFSET + 24
UPCARD_OFFSET = TRICK_OFFSET + 24
TRUMP_OFFSET = UPCARD_OFFSET + 24
DEALER_OFFSET = TRUMP_OFFSET + 4
MAKER_OFFSET = DEALER_OFFSET + 4
PLAYER_OFFSET = MAKER_OFFSET + 4
SCORE_OFFSET = PLAYER_OFFSET + 4
TRICKS_OFFSET = SCORE_OFFSET + 4
PHASE_OFFSET = TRICKS_OFFSET + 4
ACTION_MASK_OFFSET = PHASE_OFFSET + len(PHASE_ORDER)
OBSERVATION_VECTOR_SIZE = ACTION_MASK_OFFSET + ACTION_SPACE_SIZE


@dataclass(frozen=True)
class ObservationEncodingLayout:
    hand_offset: int = HAND_OFFSET
    trick_offset: int = TRICK_OFFSET
    upcard_offset: int = UPCARD_OFFSET
    trump_offset: int = TRUMP_OFFSET
    dealer_offset: int = DEALER_OFFSET
    maker_offset: int = MAKER_OFFSET
    player_offset: int = PLAYER_OFFSET
    score_offset: int = SCORE_OFFSET
    tricks_offset: int = TRICKS_OFFSET
    phase_offset: int = PHASE_OFFSET
    action_mask_offset: int = ACTION_MASK_OFFSET
    size: int = OBSERVATION_VECTOR_SIZE


LAYOUT = ObservationEncodingLayout()


def _set_card_one_hot(vector: list[float], offset: int, card: Card) -> None:
    vector[offset + card_to_index(card)] = 1.0


def _set_suit_one_hot(vector: list[float], offset: int, suit: Suit) -> None:
    vector[offset + suit_to_index(suit)] = 1.0


def _set_player_one_hot(vector: list[float], offset: int, player: int) -> None:
    if player not in {0, 1, 2, 3}:
        raise ValueError(f"Invalid player index: {player}")
    vector[offset + player] = 1.0


def encode_observation(obs: Observation) -> list[float]:
    """
    Encode an Observation as a fixed-length list of floats.

    This deliberately encodes only information available in the Observation.
    It does not include hidden opponent cards.
    """
    vector = [0.0] * OBSERVATION_VECTOR_SIZE

    # Current player's hand.
    for card in obs.hand:
        _set_card_one_hot(vector, HAND_OFFSET, card)

    # Cards currently in the trick. This currently ignores who played them;
    # that can be added later as a richer feature.
    for _, card in obs.trick:
        _set_card_one_hot(vector, TRICK_OFFSET, card)

    # Upcard.
    if obs.upcard is not None:
        _set_card_one_hot(vector, UPCARD_OFFSET, obs.upcard)

    # Trump.
    if obs.trump is not None:
        _set_suit_one_hot(vector, TRUMP_OFFSET, obs.trump)

    # Dealer, maker, and player identity.
    _set_player_one_hot(vector, DEALER_OFFSET, obs.dealer)

    if obs.maker is not None:
        _set_player_one_hot(vector, MAKER_OFFSET, obs.maker)

    _set_player_one_hot(vector, PLAYER_OFFSET, obs.player)

    # Scores. Normalize by standard winning score 10 for now.
    own_team = team_of(obs.player)
    opponent_team = 1 - own_team
    vector[SCORE_OFFSET + 0] = obs.scores[0] / 10.0
    vector[SCORE_OFFSET + 1] = obs.scores[1] / 10.0
    vector[SCORE_OFFSET + 2] = obs.scores[own_team] / 10.0
    vector[SCORE_OFFSET + 3] = obs.scores[opponent_team] / 10.0

    # Trick counts within the current hand. Normalize by 5 tricks.
    vector[TRICKS_OFFSET + 0] = obs.tricks_by_team[0] / 5.0
    vector[TRICKS_OFFSET + 1] = obs.tricks_by_team[1] / 5.0
    vector[TRICKS_OFFSET + 2] = obs.tricks_by_team[own_team] / 5.0
    vector[TRICKS_OFFSET + 3] = obs.tricks_by_team[opponent_team] / 5.0

    # Phase.
    if obs.phase in PHASE_ORDER:
        vector[PHASE_OFFSET + PHASE_ORDER.index(obs.phase)] = 1.0

    # Legal action mask.
    mask = action_mask(obs)
    for i, value in enumerate(mask):
        vector[ACTION_MASK_OFFSET + i] = float(value)

    return vector


def encode_observation_without_action_mask(obs: Observation) -> list[float]:
    """
    Encode only state features, excluding the action mask.

    Some ML code prefers to keep the legal action mask as a separate input.
    """
    return encode_observation(obs)[:ACTION_MASK_OFFSET]


def observation_action_mask(obs: Observation) -> list[int]:
    """Convenience wrapper for getting the legal action mask."""
    return action_mask(obs)

