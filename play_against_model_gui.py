from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from typing import Optional

from logic.cards import Card, Suit
from logic.env import (
    Action,
    CallTrumpAction,
    DiscardAction,
    EuchreEnv,
    Observation,
    OrderUpAction,
    PlayCardAction,
)
from logic.model_policy import ModelActionPolicy
from logic.policies import Policy, RandomPolicy
from logic.rules import team_of


CARD_WIDTH = 62
CARD_HEIGHT = 86
TABLE_WIDTH = 900
TABLE_HEIGHT = 620

SEAT_NAMES: dict[int, str] = {
    0: "You",
    1: "Left Bot",
    2: "Partner Bot",
    3: "Right Bot",
}


class PlayAgainstModelGUI:
    """Play as seat 0 while the trained model controls seats 1, 2, and 3."""

    def __init__(self, root: tk.Tk, model_path: Path, seed: Optional[int] = None):
        self.root = root
        self.root.title("Euchre vs Model")

        placeholder_policies: list[Policy] = [
            RandomPolicy(),
            RandomPolicy(),
            RandomPolicy(),
            RandomPolicy(),
        ]
        self.env = EuchreEnv(
            policies=placeholder_policies,
            winning_score=10,
            seed=seed,
        )
        self.model_policy = ModelActionPolicy(model_path)

        self.status_var = tk.StringVar(value="Welcome to Euchre.")
        self.score_var = tk.StringVar(value="")

        self.last_action_text = ""
        self.last_info_text = ""
        self.waiting_for_continue = False
        self.done = False

        self.continue_button: Optional[tk.Button] = None
        self.continue_window_id: Optional[int] = None

        self._build_widgets()
        self.new_game()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=10, pady=8)

        self.score_label = tk.Label(
            top,
            textvariable=self.score_var,
            font=("Helvetica", 14, "bold"),
        )
        self.score_label.pack(side=tk.LEFT)

        self.new_game_button = tk.Button(top, text="New Game", command=self.new_game)
        self.new_game_button.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(
            self.root,
            width=TABLE_WIDTH,
            height=TABLE_HEIGHT,
            bg="#1f7a3a",
        )
        self.canvas.pack(padx=10, pady=5)

        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Helvetica", 13),
            wraplength=850,
            justify=tk.LEFT,
        )
        self.status_label.pack(fill=tk.X, padx=10, pady=(6, 2))

        self.action_frame = tk.Frame(self.root)
        self.action_frame.pack(fill=tk.X, padx=10, pady=5)

        self.hand_frame = tk.Frame(self.root)
        self.hand_frame.pack(fill=tk.X, padx=10, pady=(5, 12))

    def clear_actions(self) -> None:
        for widget in self.action_frame.winfo_children():
            widget.destroy()

    def clear_hand_buttons(self) -> None:
        for widget in self.hand_frame.winfo_children():
            widget.destroy()

    # ------------------------------------------------------------------
    # Game flow
    # ------------------------------------------------------------------

    def new_game(self) -> None:
        self.clear_continue_button()
        self.done = False
        self.waiting_for_continue = False
        self.last_action_text = ""
        self.last_info_text = ""
        self.env.reset()
        self.status_var.set("New game started.")
        self.render()
        self.root.after(350, self.advance_bots_if_needed)

    def current_observation(self) -> Observation:
        return self.env.observation_for_player(self.env.current_player)

    def advance_bots_if_needed(self) -> None:
        if self.done or self.waiting_for_continue:
            self.render()
            return

        while self.env.current_player != 0 and not self.done and not self.waiting_for_continue:
            obs = self.current_observation()
            action = self.model_policy.choose_action(obs)
            result = self.env.step(action)

            self.last_action_text = f"{SEAT_NAMES[obs.player]}: {self.describe_action(action)}"
            self.last_info_text = self.describe_step_info(result.info)
            self.done = result.done

            if self.is_important_step(result.info):
                self.waiting_for_continue = True
                self.render()
                self.show_continue_button()
                return

            self.render()
            self.root.update_idletasks()

        self.render()
        if not self.done and not self.waiting_for_continue:
            self.prompt_human_action()

    def handle_human_action(self, action: Action) -> None:
        if self.done or self.waiting_for_continue:
            return

        obs = self.current_observation()
        if obs.player != 0:
            return

        result = self.env.step(action)
        self.last_action_text = f"You: {self.describe_action(action)}"
        self.last_info_text = self.describe_step_info(result.info)
        self.done = result.done

        self.clear_actions()
        self.clear_hand_buttons()
        self.render()

        if self.done:
            return

        if self.is_important_step(result.info):
            self.waiting_for_continue = True
            self.show_continue_button()
            return

        self.root.after(450, self.advance_bots_if_needed)

    @staticmethod
    def is_important_step(info: dict[str, object]) -> bool:
        return "trick_winner" in info or "hand_result" in info or "final_score" in info

    def clear_continue_button(self) -> None:
        if self.continue_window_id is not None:
            self.canvas.delete(self.continue_window_id)

        self.continue_window_id = None
        self.continue_button = None

    def show_continue_button(self) -> None:
        self.clear_actions()
        self.clear_continue_button()

        self.continue_button = tk.Button(
            self.canvas,
            text="Continue",
            command=self.continue_after_pause,
            width=16,
            height=2,
            font=("Helvetica", 13, "bold"),
        )
        self.continue_window_id = self.canvas.create_window(
            TABLE_WIDTH // 2,
            TABLE_HEIGHT // 2 + 170,
            window=self.continue_button,
        )

    def continue_after_pause(self) -> None:
        self.clear_continue_button()
        self.waiting_for_continue = False
        self.clear_actions()
        self.clear_hand_buttons()
        self.render()

        if self.done:
            return

        self.root.after(250, self.advance_bots_if_needed)

    # ------------------------------------------------------------------
    # Human prompts
    # ------------------------------------------------------------------

    def prompt_human_action(self) -> None:
        self.clear_actions()
        self.clear_hand_buttons()

        obs = self.current_observation()
        if obs.player != 0:
            return

        if obs.phase == "bidding_round_1":
            self.prompt_order_up(obs)
        elif obs.phase == "bidding_round_2":
            self.prompt_call_trump(obs)
        elif obs.phase == "discard":
            self.prompt_discard(obs)
        elif obs.phase == "play_card":
            self.prompt_play_card(obs)
        else:
            self.status_var.set(f"Waiting during phase {obs.phase}.")

    def prompt_order_up(self, obs: Observation) -> None:
        assert obs.upcard is not None
        self.status_var.set(f"Upcard is {obs.upcard}. Order up {obs.upcard.suit.value}?")

        for action in obs.legal_actions:
            if not isinstance(action, OrderUpAction):
                continue
            label = f"Order up {obs.upcard.suit.value}" if action.order_up else "Pass"
            tk.Button(
                self.action_frame,
                text=label,
                command=lambda chosen=action: self.handle_human_action(chosen),
            ).pack(side=tk.LEFT, padx=4)

        self.render_hand_buttons(obs, disabled=True)

    def prompt_call_trump(self, obs: Observation) -> None:
        assert obs.upcard is not None
        self.status_var.set(f"Choose trump, or pass. Cannot choose {obs.upcard.suit.value}.")

        for action in obs.legal_actions:
            if not isinstance(action, CallTrumpAction):
                continue
            label = "Pass" if action.suit is None else f"Call {action.suit.value}"
            tk.Button(
                self.action_frame,
                text=label,
                command=lambda chosen=action: self.handle_human_action(chosen),
            ).pack(side=tk.LEFT, padx=4)

        self.render_hand_buttons(obs, disabled=True)

    def prompt_discard(self, obs: Observation) -> None:
        self.status_var.set("You picked up the upcard. Choose one card to discard.")
        discard_actions = [action for action in obs.legal_actions if isinstance(action, DiscardAction)]
        self.render_hand_buttons(obs, discard_actions=discard_actions)

    def prompt_play_card(self, obs: Observation) -> None:
        if len(obs.trick) == 0:
            self.status_var.set("Your turn. You are leading this trick.")
        else:
            self.status_var.set("Your turn. Follow suit if possible.")

        play_actions = [action for action in obs.legal_actions if isinstance(action, PlayCardAction)]
        self.render_hand_buttons(obs, play_actions=play_actions)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> None:
        self.score_var.set(
            f"Game: {self.env.scores[0]}-{self.env.scores[1]}    "
            f"Tricks: {self.env.tricks_by_team[0]}-{self.env.tricks_by_team[1]}    "
            f"Dealer: {SEAT_NAMES[self.env.dealer]}"
        )

        self.canvas.delete("all")
        self.continue_window_id = None
        self.continue_button = None
        self.draw_upcard_and_trump()
        self.draw_trick_score_panel()
        self.draw_bot_hands()
        self.draw_trick()
        self.draw_table_labels()

        if self.done:
            winner = 0 if self.env.scores[0] > self.env.scores[1] else 1
            self.status_var.set(
                f"Game over. Team {winner} wins {self.env.scores[winner]}-"
                f"{self.env.scores[1 - winner]}."
            )
            self.clear_actions()
            self.clear_hand_buttons()
            tk.Button(self.action_frame, text="New Game", command=self.new_game).pack(
                side=tk.LEFT,
                padx=4,
            )
            return

        if self.env.current_player != 0 or self.waiting_for_continue:
            self.clear_hand_buttons()
            self.render_hand_buttons(self.env.observation_for_player(0), disabled=True)

        status_lines = []
        if self.last_action_text:
            status_lines.append(self.last_action_text)
        if self.last_info_text:
            status_lines.append(self.last_info_text)
        if status_lines and (self.env.current_player != 0 or self.waiting_for_continue):
            self.status_var.set("\n".join(status_lines))

    def draw_trick_score_panel(self) -> None:
        x = 25
        y = 62
        w = 210
        h = 92

        self.canvas.create_rectangle(
            x,
            y,
            x + w,
            y + h,
            fill="#145c2c",
            outline="#d6f5d6",
            width=2,
        )

        self.canvas.create_text(
            x + w // 2,
            y + 18,
            text="Current Hand",
            fill="white",
            font=("Helvetica", 12, "bold"),
        )

        self.canvas.create_text(
            x + w // 2,
            y + 44,
            text=f"Tricks: Team 0 = {self.env.tricks_by_team[0]}    Team 1 = {self.env.tricks_by_team[1]}",
            fill="white",
            font=("Helvetica", 12, "bold"),
        )

        self.canvas.create_text(
            x + w // 2,
            y + 70,
            text=f"Game: Team 0 = {self.env.scores[0]}    Team 1 = {self.env.scores[1]}",
            fill="white",
            font=("Helvetica", 10, "bold"),
        )

    def draw_table_labels(self) -> None:
        positions = {
            0: (TABLE_WIDTH // 2, TABLE_HEIGHT - 92),
            1: (115, 178),
            2: (TABLE_WIDTH // 2, 26),
            3: (TABLE_WIDTH - 115, 178),
        }

        for player, (x, y) in positions.items():
            dealer_mark = " (D)" if player == self.env.dealer else ""
            turn_mark = " ←" if player == self.env.current_player and not self.done else ""
            label = f"{SEAT_NAMES[player]}{dealer_mark}{turn_mark}\nTeam {team_of(player)}"

            self.canvas.create_rectangle(
                x - 72,
                y - 24,
                x + 72,
                y + 24,
                fill="#145c2c",
                outline="#d6f5d6",
                width=1,
            )
            self.canvas.create_text(
                x,
                y,
                text=label,
                fill="white",
                font=("Helvetica", 13, "bold"),
                justify=tk.CENTER,
            )

    def draw_upcard_and_trump(self) -> None:
        panel_x = TABLE_WIDTH - 145
        panel_y = TABLE_HEIGHT - 180
        panel_w = 125
        panel_h = 160

        self.canvas.create_rectangle(
            panel_x - 15,
            panel_y - 35,
            panel_x + panel_w,
            panel_y + panel_h,
            fill="#145c2c",
            outline="#d6f5d6",
            width=2,
        )

        card_x = panel_x + 31
        card_y = panel_y + 1
        self.canvas.create_text(
            card_x + CARD_WIDTH // 2,
            panel_y - 14,
            text="Upcard",
            fill="white",
            font=("Helvetica", 11, "bold"),
        )

        if self.env.upcard is not None:
            self.draw_card(self.env.upcard, card_x, card_y, faded=self.env.trump is not None)

        trump_text = f"Trump: {self.env.trump.value}" if self.env.trump else "Trump: not chosen"
        maker_text = "Maker: none" if self.env.maker is None else f"Maker: {SEAT_NAMES[self.env.maker]}"

        self.canvas.create_text(
            card_x + CARD_WIDTH // 2,
            card_y + CARD_HEIGHT + 30,
            text=f"{trump_text}\n{maker_text}",
            fill="white",
            font=("Helvetica", 10, "bold"),
            justify=tk.CENTER,
        )

    def draw_bot_hands(self) -> None:
        self.draw_card_backs(center_x=TABLE_WIDTH // 2, y=64, count=len(self.env.hands[2]), horizontal=True)
        self.draw_card_backs(center_x=72, y=230, count=len(self.env.hands[1]), horizontal=False)
        self.draw_card_backs(center_x=TABLE_WIDTH - 72, y=230, count=len(self.env.hands[3]), horizontal=False)

    def draw_trick(self) -> None:
        positions = {
            0: (TABLE_WIDTH // 2 - CARD_WIDTH // 2, TABLE_HEIGHT // 2 + 72),
            1: (TABLE_WIDTH // 2 - 165, TABLE_HEIGHT // 2 - CARD_HEIGHT // 2),
            2: (TABLE_WIDTH // 2 - CARD_WIDTH // 2, TABLE_HEIGHT // 2 - 118),
            3: (TABLE_WIDTH // 2 + 100, TABLE_HEIGHT // 2 - CARD_HEIGHT // 2),
        }

        for player, card in self.env.trick:
            x, y = positions[player]
            self.draw_card(card, x, y)
            self.canvas.create_text(
                x + CARD_WIDTH // 2,
                y + CARD_HEIGHT + 14,
                text=SEAT_NAMES[player],
                fill="white",
                font=("Helvetica", 10, "bold"),
            )

    def draw_card(self, card: Card, x: int, y: int, faded: bool = False) -> None:
        fill = "white"
        outline = "#aaaaaa" if faded else "black"
        self.canvas.create_rectangle(
            x,
            y,
            x + CARD_WIDTH,
            y + CARD_HEIGHT,
            fill=fill,
            outline=outline,
            width=2,
        )

        color = "red" if card.suit in {Suit.HEARTS, Suit.DIAMONDS} else "black"
        self.canvas.create_text(
            x + CARD_WIDTH // 2,
            y + CARD_HEIGHT // 2,
            text=str(card),
            fill=color,
            font=("Helvetica", 18, "bold"),
        )

    def draw_card_back(self, x: int, y: int) -> None:
        self.canvas.create_rectangle(
            x,
            y,
            x + CARD_WIDTH,
            y + CARD_HEIGHT,
            fill="#0b3d91",
            outline="white",
            width=2,
        )
        self.canvas.create_rectangle(
            x + 8,
            y + 8,
            x + CARD_WIDTH - 8,
            y + CARD_HEIGHT - 8,
            outline="white",
        )
        self.canvas.create_text(
            x + CARD_WIDTH // 2,
            y + CARD_HEIGHT // 2,
            text="★",
            fill="white",
            font=("Helvetica", 22, "bold"),
        )

    def draw_card_backs(self, center_x: int, y: int, count: int, horizontal: bool) -> None:
        if count <= 0:
            return

        spacing = 20
        if horizontal:
            total_width = CARD_WIDTH + spacing * (count - 1)
            start_x = center_x - total_width // 2
            for i in range(count):
                self.draw_card_back(start_x + i * spacing, y)
        else:
            for i in range(count):
                self.draw_card_back(center_x - CARD_WIDTH // 2, y + i * 18)

    def render_hand_buttons(
        self,
        obs: Observation,
        play_actions: Optional[list[PlayCardAction]] = None,
        discard_actions: Optional[list[DiscardAction]] = None,
        disabled: bool = False,
    ) -> None:
        self.clear_hand_buttons()

        play_by_card = {action.card: action for action in play_actions or []}
        discard_by_card = {action.card: action for action in discard_actions or []}

        for card in self.env.hands[0]:
            command = None
            is_illegal = False

            if disabled:
                state = tk.DISABLED
            elif card in play_by_card:
                state = tk.NORMAL
                command = lambda chosen=play_by_card[card]: self.handle_human_action(chosen)
            elif card in discard_by_card:
                state = tk.NORMAL
                command = lambda chosen=discard_by_card[card]: self.handle_human_action(chosen)
            else:
                state = tk.DISABLED
                is_illegal = bool(play_actions or discard_actions)

            suit_color = "red" if card.suit in {Suit.HEARTS, Suit.DIAMONDS} else "black"
            foreground = "#888888" if is_illegal else suit_color
            background = "#f2f2f2" if is_illegal else "white"

            kwargs = {
                "text": str(card),
                "width": 10,
                "height": 2,
                "state": state,
                "bg": background,
                "fg": foreground,
                "activebackground": "#eeeeee",
                "activeforeground": suit_color,
                "disabledforeground": foreground,
                "relief": tk.RAISED,
                "font": ("Helvetica", 13, "bold"),
            }
            if command is not None:
                kwargs["command"] = command

            tk.Button(self.hand_frame, **kwargs).pack(side=tk.LEFT, padx=4)

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    @staticmethod
    def describe_action(action: Action) -> str:
        if isinstance(action, OrderUpAction):
            return "orders up" if action.order_up else "passes"
        if isinstance(action, CallTrumpAction):
            return "passes" if action.suit is None else f"calls {action.suit.value}"
        if isinstance(action, DiscardAction):
            return f"discards {action.card}"
        if isinstance(action, PlayCardAction):
            return f"plays {action.card}"
        return repr(action)

    @staticmethod
    def describe_step_info(info: dict[str, object]) -> str:
        pieces: list[str] = []
        if "trick_winner" in info:
            winner = info["trick_winner"]
            if isinstance(winner, int):
                verb = "win" if winner == 0 else "wins"
                pieces.append(f"{SEAT_NAMES[winner]} {verb} the trick.")
                pieces.append("Press Continue for the next trick.")

        if "hand_result" in info:
            result = info["hand_result"]
            pieces.append(f"Hand result: {result}")
            pieces.append("Press Continue for the next hand.")

        if "final_score" in info:
            pieces.append(f"Final score: {info['final_score']}")

        return "\n".join(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Euchre against a trained model.")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained .pt model checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible deals.",
    )
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry("940x760")
    root.update_idletasks()
    root.lift()
    root.attributes("-topmost", True)
    root.after(250, lambda: root.attributes("-topmost", False))

    PlayAgainstModelGUI(root, model_path=args.model, seed=args.seed)
    root.mainloop()


if __name__ == "__main__":
    main()
