from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

import argparse

import time

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from logic.action_encoding import ACTION_SPACE_SIZE
from logic.observation_encoding import ACTION_MASK_OFFSET


DEFAULT_DATA_PATH = Path("data/imitation_simple_bot.csv")
DEFAULT_MODEL_PATH = Path("models/imitation_simple_bot.pt")


@dataclass(frozen=True)
class TrainingConfig:
    data_path: Path = DEFAULT_DATA_PATH
    model_path: Path = DEFAULT_MODEL_PATH
    batch_size: int = 256
    epochs: int = 20
    learning_rate: float = 1e-3
    validation_fraction: float = 0.2
    seed: int = 123


class ImitationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Dataset of rows:

        state_features..., action_mask..., action_id

    state_features excludes the action mask. The mask is kept separately so the
    model can be trained/evaluated with invalid actions suppressed.
    """

    def __init__(self, csv_path: Path):
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Could not find {csv_path}. Run generate_imitation_data.py first."
            )

        states: list[list[float]] = []
        masks: list[list[float]] = []
        actions: list[int] = []

        print(f"Loading dataset from {csv_path}...")
        start_time = time.perf_counter()
        with csv_path.open(newline="") as file:
            reader = csv.reader(file)
            header = next(reader)

            expected_min_columns = ACTION_MASK_OFFSET + ACTION_SPACE_SIZE + 1
            if len(header) != expected_min_columns:
                raise ValueError(
                    f"Unexpected CSV column count. Got {len(header)}, "
                    f"expected {expected_min_columns}."
                )

            row_iterator = reader
            if tqdm is not None:
                row_iterator = tqdm(reader, desc="Reading CSV rows", unit="rows")

            for row in row_iterator:
                if not row:
                    continue

                values = [float(value) for value in row]
                state = values[:ACTION_MASK_OFFSET]
                mask = values[ACTION_MASK_OFFSET : ACTION_MASK_OFFSET + ACTION_SPACE_SIZE]
                action_id = int(values[-1])

                if not 0 <= action_id < ACTION_SPACE_SIZE:
                    raise ValueError(f"Invalid action id in dataset: {action_id}")

                if mask[action_id] != 1.0:
                    raise ValueError(
                        f"Dataset row has illegal target action {action_id}; "
                        f"mask value is {mask[action_id]}."
                    )

                states.append(state)
                masks.append(mask)
                actions.append(action_id)

        self.states = torch.tensor(states, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)
        elapsed = time.perf_counter() - start_time
        print(f"Loaded {len(self.actions)} examples in {elapsed:.2f}s")

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[index], self.masks[index], self.actions[index]


class ImitationNet(nn.Module):
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


def mask_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Set illegal action logits to a very negative value.

    This makes argmax ignore illegal moves. For the cross-entropy loss, the
    target action is always legal, so this is safe.
    """
    return logits.masked_fill(masks <= 0.0, -1e9)


def accuracy(logits: torch.Tensor, masks: torch.Tensor, targets: torch.Tensor) -> float:
    masked = mask_logits(logits, masks)
    predictions = masked.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def run_epoch(
    model: ImitationNet,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: Optional[torch.optim.Optimizer] = None,
    description: str = "epoch",
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_accuracy = 0.0
    total_examples = 0

    loss_fn = nn.CrossEntropyLoss()

    batch_iterator = loader
    if tqdm is not None:
        batch_iterator = tqdm(loader, desc=description, unit="batch", leave=False)

    for states, masks, targets in batch_iterator:
        logits = model(states)
        masked_logits = mask_logits(logits, masks)
        loss = loss_fn(masked_logits, targets)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = states.shape[0]
        total_loss += loss.item() * batch_size
        total_accuracy += accuracy(logits, masks, targets) * batch_size
        total_examples += batch_size

    return total_loss / total_examples, total_accuracy / total_examples


def train(config: TrainingConfig = TrainingConfig()) -> None:
    torch.manual_seed(config.seed)

    dataset = ImitationDataset(config.data_path)
    if len(dataset) < 10:
        raise ValueError("Dataset is too small to train. Generate more games first.")

    validation_size = max(1, int(len(dataset) * config.validation_fraction))
    train_size = len(dataset) - validation_size

    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)

    input_size = ACTION_MASK_OFFSET
    model = ImitationNet(input_size=input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print(f"Loaded {len(dataset)} examples")
    print(f"Train examples: {train_size}")
    print(f"Validation examples: {validation_size}")
    print(f"Input size: {input_size}")
    print(f"Action space size: {ACTION_SPACE_SIZE}")
    print()

    best_validation_accuracy = -1.0
    config.model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            optimizer,
            description=f"Epoch {epoch:02d} train",
        )

        with torch.no_grad():
            validation_loss, validation_accuracy = run_epoch(
                model,
                validation_loader,
                description=f"Epoch {epoch:02d} val",
            )

        epoch_elapsed = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f}, train acc {train_accuracy:.3f} | "
            f"val loss {validation_loss:.4f}, val acc {validation_accuracy:.3f} | "
            f"time {epoch_elapsed:.1f}s"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": input_size,
                    "action_space_size": ACTION_SPACE_SIZE,
                    "validation_accuracy": validation_accuracy,
                    "config": {
                        **config.__dict__,
                        "data_path": str(config.data_path),
                        "model_path": str(config.model_path),
                    },
                },
                config.model_path,
            )

    print()
    print(f"Best validation accuracy: {best_validation_accuracy:.3f}")
    print(f"Saved model to {config.model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an imitation model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to imitation CSV dataset.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to save trained model.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    train(
        TrainingConfig(
            data_path=args.data,
            model_path=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()

