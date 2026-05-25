# Euchre

A Python Euchre simulator with a terminal/GUI frontend, bot policies, a headless simulation environment, and an early machine learning pipeline for training Euchre agents.

## Project Structure

```text
euchre/
├── logic/
│   ├── cards.py                   # Card, Rank, Suit, deck helpers, bower logic
│   ├── rules.py                   # Trick winner, legal cards, scoring helpers
│   ├── policies.py                # HumanPolicy, RandomPolicy, SimpleBotPolicy, legacy game runner
│   ├── env.py                     # Headless Euchre environment and step API
│   ├── action_encoding.py         # Symbolic action <-> fixed integer action ID
│   ├── observation_encoding.py    # Observation <-> numeric feature vector
│   └── model_policy.py            # Load trained PyTorch model as an action policy
├── euchre_gui.py                  # Tkinter GUI version
├── euchre_terminal.py             # Terminal version
├── simulate.py                    # Bot-vs-bot simulation
├── simulate_steps.py              # Step API debugging script
├── generate_imitation_data.py     # Generate SimpleBot imitation data
├── train_imitation_model.py       # Train imitation model
├── evaluate_model_policy.py       # Evaluate trained model against bots
├── diagnose_imitation_accuracy.py # Accuracy by phase against SimpleBot teacher
└── tests/
```

## Setup

Python 3.11 is recommended.

```bash
python3.11 -m pip install pytest torch tqdm
```

On macOS, if `tkinter` is missing for the GUI:

```bash
brew install python-tk@3.11
```

## Run the GUI

```bash
python3.11 euchre_gui.py
```

## Run Tests

Use `python -m pytest` so the project root is on the import path:

```bash
python3.11 -m pytest
```

## Run Basic Simulations

```bash
python3.11 simulate.py
```

This compares policies such as:

```text
SimpleBotPolicy vs SimpleBotPolicy
SimpleBotPolicy vs RandomPolicy
RandomPolicy vs RandomPolicy
```

## Generate Imitation Data

Generate supervised training examples from `SimpleBotPolicy`:

```bash
python3.11 generate_imitation_data.py \
  --games 5000 \
  --output data/imitation_simple_bot_5k.csv
```

The generated CSV contains:

```text
observation features, legal action mask, chosen action ID
```

## Train an Imitation Model

```bash
python3.11 train_imitation_model.py \
  --data data/imitation_simple_bot_5k.csv \
  --model models/imitation_simple_bot_5k.pt \
  --epochs 10 \
  --batch-size 1024
```

The model learns to imitate `SimpleBotPolicy` by predicting a legal action ID from an encoded observation.

## Evaluate a Trained Model

```bash
python3.11 evaluate_model_policy.py
```

This evaluates a trained model against:

```text
RandomPolicy
SimpleBotPolicy
```

Typical comparisons:

```text
Model team vs Random team
Random team vs Model team
Model team vs Simple team
Simple team vs Model team
```

## Diagnose Imitation Accuracy

To see where the model differs from `SimpleBotPolicy`:

```bash
python3.11 diagnose_imitation_accuracy.py \
  --model models/imitation_simple_bot_5k.pt \
  --games 1000 \
  --seed 123
```

This reports accuracy by phase:

```text
bidding_round_1
bidding_round_2
discard
play_card
```

Recent diagnostics showed that bidding is learned almost perfectly, while `play_card` is the weakest phase.

## Estimate the Policy Frontier

To compare the current model against stronger search policies from both seating
orientations:

```bash
python3.11 evaluate_policy_frontier.py \
  --model models/best_mixed_simple3k_ismc500.pt \
  --games 1000 \
  --seeds 123 456 789
```

This benchmarks:

```text
model   current trained checkpoint
ismc    fair information-set Monte Carlo, no hidden-card access
oracle  hidden-information Monte Carlo upper-bound probe
```

The oracle policy is not a legal deployable Euchre player because it sees hidden
hands. Treat it as a ceiling/probe for “how much could perfect information help”
rather than as fair perfect play.

## Current ML Pipeline

The current pipeline is:

```text
EuchreEnv
→ Observation
→ numeric observation encoding
→ legal action mask
→ model outputs action ID
→ action ID maps to symbolic action
→ env.step(action)
```

So far, the model is trained by imitation learning from `SimpleBotPolicy`.

## Git-Ignored Outputs

Generated data and models should not be committed:

```gitignore
data/
models/
*.pt
*.pth
```
