from logic.cards import Card, Rank, Suit
from logic.env import EuchreEnv, PlayCardAction
from logic.policies import Policy, RandomPolicy

from logic.information_set_monte_carlo_policy import InformationSetMonteCarloPolicy
from logic.cards import effective_suit

def make_env() -> EuchreEnv:
    policies: list[Policy] = [
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
    ]
    return EuchreEnv(policies=policies, winning_score=10, seed=123)


def test_player_marked_void_when_they_cannot_follow_led_suit() -> None:
    env = make_env()
    env.deal()

    env.phase = "play_card"
    env.trump = Suit.HEARTS
    env.current_player = 0
    env.trick = []
    env.led_suit = None

    lead_card = Card(Rank.ACE, Suit.SPADES)
    off_suit_card = Card(Rank.NINE, Suit.CLUBS)

    env.hands[0] = [lead_card]
    env.hands[1] = [off_suit_card]

    env.apply_play_card_action(0, PlayCardAction(lead_card))
    env.current_player = 1
    env.apply_play_card_action(1, PlayCardAction(off_suit_card))

    assert Suit.SPADES in env.void_suits[1]


def test_player_not_marked_void_when_they_follow_led_suit() -> None:
    env = make_env()
    env.deal()

    env.phase = "play_card"
    env.trump = Suit.HEARTS
    env.current_player = 0
    env.trick = []
    env.led_suit = None

    lead_card = Card(Rank.ACE, Suit.SPADES)
    follow_card = Card(Rank.NINE, Suit.SPADES)

    env.hands[0] = [lead_card]
    env.hands[1] = [follow_card]

    env.apply_play_card_action(0, PlayCardAction(lead_card))
    env.current_player = 1
    env.apply_play_card_action(1, PlayCardAction(follow_card))

    assert Suit.SPADES not in env.void_suits[1]


def test_information_set_sampler_respects_void_suits_during_play() -> None:
    env = make_env()
    env.deal()
    env.trump = Suit.HEARTS
    env.phase = "play_card"
    env.current_player = 0
    env.void_suits[1].add(Suit.SPADES)

    obs = env.observation_for_player(0)
    policy = InformationSetMonteCarloPolicy(samples_per_action=1, seed=123)

    sampled = policy.sample_compatible_env(env, obs)

    for card in sampled.hands[1]:
        assert effective_suit(card, Suit.HEARTS) != Suit.SPADES