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
TRICK_BY_PLAYER_OFFSET = HAND_OFFSET + 24          # 24-119
PLAYED_CARDS_OFFSET = TRICK_BY_PLAYER_OFFSET + 96  # 120-143
LEGAL_CARDS_OFFSET = PLAYED_CARDS_OFFSET + 24      # 144-167
UPCARD_OFFSET = LEGAL_CARDS_OFFSET + 24            # 168-191
TRUMP_OFFSET = UPCARD_OFFSET + 24                  # 192-195
LED_SUIT_OFFSET = TRUMP_OFFSET + 4                 # 196-199
DEALER_OFFSET = LED_SUIT_OFFSET + 4                # 200-203
MAKER_OFFSET = DEALER_OFFSET + 4                   # 204-207
PLAYER_OFFSET = MAKER_OFFSET + 4                   # 208-211
CURRENT_WINNER_OFFSET = PLAYER_OFFSET + 4          # 212-215
CURRENT_WINNING_CARD_OFFSET = CURRENT_WINNER_OFFSET + 4  # 216-239
PARTNER_WINNING_OFFSET = CURRENT_WINNING_CARD_OFFSET + 24  # 240
TRICK_SIZE_OFFSET = PARTNER_WINNING_OFFSET + 1     # 241-244
SCORE_OFFSET = TRICK_SIZE_OFFSET + 4               # 245-248
TRICKS_OFFSET = SCORE_OFFSET + 4                   # 249-252
PHASE_OFFSET = TRICKS_OFFSET + 4                   # 253-258
ACTION_MASK_OFFSET = PHASE_OFFSET + len(PHASE_ORDER)
OBSERVATION_VECTOR_SIZE = ACTION_MASK_OFFSET + ACTION_SPACE_SIZE


@dataclass(frozen=True)
class ObservationEncodingLayout:
    hand_offset: int = HAND_OFFSET
    trick_by_player_offset: int = TRICK_BY_PLAYER_OFFSET
    played_cards_offset: int = PLAYED_CARDS_OFFSET
    legal_cards_offset: int = LEGAL_CARDS_OFFSET
    upcard_offset: int = UPCARD_OFFSET
    trump_offset: int = TRUMP_OFFSET
    led_suit_offset: int = LED_SUIT_OFFSET
    dealer_offset: int = DEALER_OFFSET
    maker_offset: int = MAKER_OFFSET
    player_offset: int = PLAYER_OFFSET
    current_winner_offset: int = CURRENT_WINNER_OFFSET
    current_winning_card_offset: int = CURRENT_WINNING_CARD_OFFSET
    partner_winning_offset: int = PARTNER_WINNING_OFFSET
    trick_size_offset: int = TRICK_SIZE_OFFSET
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
    for trick_player, card in obs.trick:
        offset = TRICK_BY_PLAYER_OFFSET + 24 * trick_player
        _set_card_one_hot(vector, offset, card)

    
    for card in obs.played_cards:
        _set_card_one_hot(vector, PLAYED_CARDS_OFFSET, card)


    # Legal cards during play.
    for card in obs.legal_cards:
        _set_card_one_hot(vector, LEGAL_CARDS_OFFSET, card)


    # Upcard.
    if obs.upcard is not None:
        _set_card_one_hot(vector, UPCARD_OFFSET, obs.upcard)

    # Trump.
    if obs.trump is not None:
        _set_suit_one_hot(vector, TRUMP_OFFSET, obs.trump)


    # Led suit.
    if obs.led_suit is not None:
        _set_suit_one_hot(vector, LED_SUIT_OFFSET, obs.led_suit)


    # Dealer, maker, and player identity.
    _set_player_one_hot(vector, DEALER_OFFSET, obs.dealer)

    if obs.maker is not None:
        _set_player_one_hot(vector, MAKER_OFFSET, obs.maker)

    _set_player_one_hot(vector, PLAYER_OFFSET, obs.player)



    # Current trick winner.
    if obs.current_trick_winner is not None:
        _set_player_one_hot(vector, CURRENT_WINNER_OFFSET, obs.current_trick_winner)

    # Current winning card.
    if obs.current_trick_winning_card is not None:
        _set_card_one_hot(
            vector,
            CURRENT_WINNING_CARD_OFFSET,
            obs.current_trick_winning_card,
        )

    # Whether this player's team is currently winning the trick.
    vector[PARTNER_WINNING_OFFSET] = 1.0 if obs.partner_winning_trick else 0.0

    # Trick size, one-hot. Length 0 means player is leading.
    trick_size = len(obs.trick)
    if 0 <= trick_size <= 3:
        vector[TRICK_SIZE_OFFSET + trick_size] = 1.0



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

