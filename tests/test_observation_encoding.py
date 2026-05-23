from logic.action_encoding import ACTION_SPACE_SIZE, legal_action_ids
from logic.env import EuchreEnv
from logic.observation_encoding import (
    ACTION_MASK_OFFSET,
    OBSERVATION_VECTOR_SIZE,
    encode_observation,
    encode_observation_without_action_mask,
    observation_action_mask,
)
from logic.policies import Policy, RandomPolicy
from logic.env import RandomActionPolicy


def make_env() -> EuchreEnv:
    policies: list[Policy] = [
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
    ]
    return EuchreEnv(policies=policies, winning_score=10, seed=123)


def test_encoded_observation_has_expected_length() -> None:
    env = make_env()
    obs = env.reset()

    encoded = encode_observation(obs)

    assert len(encoded) == OBSERVATION_VECTOR_SIZE


def test_state_encoding_without_action_mask_has_expected_length() -> None:
    env = make_env()
    obs = env.reset()

    encoded = encode_observation_without_action_mask(obs)

    assert len(encoded) == ACTION_MASK_OFFSET


def test_action_mask_section_matches_legal_actions() -> None:
    env = make_env()
    obs = env.reset()

    encoded = encode_observation(obs)
    mask_section = encoded[ACTION_MASK_OFFSET:]

    assert len(mask_section) == ACTION_SPACE_SIZE
    assert sum(mask_section) == len(obs.legal_actions)

    for action_id in legal_action_ids(obs):
        assert mask_section[action_id] == 1.0


def test_observation_action_mask_matches_legal_actions() -> None:
    env = make_env()
    obs = env.reset()

    mask = observation_action_mask(obs)

    assert len(mask) == ACTION_SPACE_SIZE
    assert sum(mask) == len(obs.legal_actions)

    for action_id in legal_action_ids(obs):
        assert mask[action_id] == 1



def test_play_context_features_exist_during_play() -> None:
    policies: list[Policy] = [
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
    ]
    env = EuchreEnv(policies=policies, winning_score=10, seed=123)
    obs = env.reset()

    # Advance randomly until play_card with at least one trick card.
    policy = RandomActionPolicy(seed=456)
    for _ in range(100):
        result = env.step(policy.choose_action(obs))
        obs = result.observation

        if obs.phase == "play_card" and len(obs.trick) > 0:
            assert obs.led_suit is not None
            assert obs.current_trick_winner is not None
            assert obs.current_trick_winning_card is not None
            assert len(obs.legal_cards) > 0
            break
    else:
        raise AssertionError("Did not reach a nonempty play_card trick.")
