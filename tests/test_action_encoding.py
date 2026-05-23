from logic.action_encoding import (
    ACTION_SPACE_SIZE,
    action_mask,
    action_to_id,
    id_to_legal_action,
    legal_action_ids,
)
from logic.env import EuchreEnv, OrderUpAction, RandomActionPolicy
from logic.policies import Policy, RandomPolicy


def make_env() -> EuchreEnv:
    policies: list[Policy] = [
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
        RandomPolicy(),
    ]
    return EuchreEnv(policies=policies, winning_score=10, seed=123)


def test_order_up_actions_have_expected_ids() -> None:
    assert action_to_id(OrderUpAction(order_up=False)) == 0
    assert action_to_id(OrderUpAction(order_up=True)) == 1


def test_legal_action_ids_match_observation_actions() -> None:
    env = make_env()
    obs = env.reset()

    ids = legal_action_ids(obs)

    assert set(ids) == {action_to_id(action) for action in obs.legal_actions}


def test_action_mask_marks_legal_actions() -> None:
    env = make_env()
    obs = env.reset()

    mask = action_mask(obs)

    assert len(mask) == ACTION_SPACE_SIZE
    assert sum(mask) == len(obs.legal_actions)

    for action_id in legal_action_ids(obs):
        assert mask[action_id] == 1


def test_legal_action_id_round_trip() -> None:
    env = make_env()
    policy = RandomActionPolicy(seed=123)

    obs = env.reset()

    for _ in range(50):
        action = policy.choose_action(obs)
        action_id = action_to_id(action)

        reconstructed = id_to_legal_action(action_id, obs)

        assert reconstructed == action

        result = env.step(action)
        obs = result.observation

        if result.done:
            break
        