from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from logic.action_encoding import (
    ACTION_SPACE_SIZE,
    action_to_id,
    id_to_legal_action,
    legal_action_ids,
)
from logic.env import Action, ActionPolicy, Observation
from logic.observation_encoding import ACTION_MASK_OFFSET, encode_observation_without_action_mask


class ImitationNet(nn.Module):
    """
    Same architecture used by train_imitation_model.py.

    We duplicate the small definition here so model loading does not need to
    import the training script.
    """

    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, ACTION_SPACE_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelActionPolicy:
    """
    ActionPolicy backed by a trained imitation model.

    The model receives the encoded observation state and outputs logits over the
    fixed 68-action space. Illegal actions are masked out before choosing the
    action with highest logit.
    """

    def __init__(self, model_path: Path | str = Path("models/imitation_simple_bot.pt")):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Could not find model at {self.model_path}. Run train_imitation_model.py first."
            )

        checkpoint = torch.load(self.model_path, map_location="cpu")
        input_size = int(checkpoint.get("input_size", ACTION_MASK_OFFSET))

        self.model = ImitationNet(input_size=input_size)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def choose_action(self, observation: Observation) -> Action:
        state = encode_observation_without_action_mask(observation)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        legal_ids = set(legal_action_ids(observation))
        if not legal_ids:
            raise ValueError(f"No legal actions available for observation phase {observation.phase!r}.")

        with torch.no_grad():
            logits = self.model(state_tensor).squeeze(0)

        # Mask illegal actions.
        masked_logits = torch.full_like(logits, -1e9)
        for action_id in legal_ids:
            masked_logits[action_id] = logits[action_id]

        chosen_id = int(torch.argmax(masked_logits).item())
        return id_to_legal_action(chosen_id, observation)


def action_policy_accuracy(policy: ActionPolicy, observations: list[Observation], targets: list[Action]) -> float:
    """
    Utility for measuring how often an ActionPolicy exactly matches target actions.
    """
    if len(observations) != len(targets):
        raise ValueError("observations and targets must have the same length.")
    if not observations:
        raise ValueError("Need at least one observation to compute accuracy.")

    correct = 0
    for observation, target in zip(observations, targets):
        predicted = policy.choose_action(observation)
        correct += int(action_to_id(predicted) == action_to_id(target))

    return correct / len(observations)

