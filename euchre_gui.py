from __future__ import annotations

import random
import tkinter as tk
from typing import Callable, Optional, cast

from logic.cards import (
    Card,
    Rank,
    Suit,
    SUITS,
    effective_suit,
    is_left_bower,
    is_right_bower,
    make_deck,
)

from logic.rules import (
    legal_cards,
    next_player,
    left_of,
    team_of,
    trick_winner,
)

from logic.policies import SimpleBotPolicy, Protocol, EuchreGame


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


class BotGameView(Protocol):
    """The small slice of game state SimpleBotPolicy needs.

    For now SimpleBotPolicy is still typed against EuchreGame, so calls below
    use cast(EuchreGame, self). This protocol documents the intended future
    refactor: policies should depend on a small game-state interface, not the
    terminal game class.
    """

    hands: list[list[Card]]
    trump: Optional[Suit]


class EuchreGUI:
    """First playable graphical Euchre prototype using tkinter.

    Current limitations:
    - Human is always Player 0.
    - Three bots use SimpleBotPolicy.
    - No going alone yet.
    - If everyone passes, the hand is redealt with the next dealer.
    """

    def __init__(self, root: tk.Tk, seed: Optional[int] = None):
        self.root = root
        self.root.title("Euchre Simulator")
        self.random = random.Random(seed)

        self.bot_policy = SimpleBotPolicy()
        self.winning_score = 10

        self.dealer = 0
        self.scores = [0, 0]

        self.hands: list[list[Card]] = [[] for _ in range(4)]
        self.kitty: list[Card] = []
        self.upcard: Optional[Card] = None
        self.trump: Optional[Suit] = None
        self.maker: Optional[int] = None

        self.bid_round = 1
        self.bid_player = 0
        self.bid_passes = 0
        self.phase = "setup"

        self.trick: list[tuple[int, Card]] = []
        self.tricks_by_team = [0, 0]
        self.leader = 0
        self.current_player = 0
        self.trick_number = 1
        self.led_suit: Optional[Suit] = None

        self.status_var = tk.StringVar(value="Welcome to Euchre.")
        self.score_var = tk.StringVar(value="")

        self._build_widgets()
        self.start_new_hand()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=10, pady=8)

        self.score_label = tk.Label(top, textvariable=self.score_var, font=("Helvetica", 14, "bold"))
        self.score_label.pack(side=tk.LEFT)

        self.new_hand_button = tk.Button(top, text="Redeal Hand", command=self.force_redeal)
        self.new_hand_button.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(self.root, width=TABLE_WIDTH, height=TABLE_HEIGHT, bg="#1f7a3a")
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
    # Game lifecycle
    # ------------------------------------------------------------------

    def force_redeal(self) -> None:
        self.status_var.set("Redealt the hand.")
        self.start_new_hand(advance_dealer=False)

    def start_new_hand(self, advance_dealer: bool = False) -> None:
        if advance_dealer:
            self.dealer = next_player(self.dealer)

        deck = make_deck()
        self.random.shuffle(deck)

        self.hands = [deck[i * 5 : (i + 1) * 5] for i in range(4)]
        for hand in self.hands:
            self.sort_hand(hand)

        self.upcard = deck[20]
        self.kitty = deck[21:]
        self.trump = None
        self.maker = None

        self.bid_round = 1
        self.bid_player = left_of(self.dealer)
        self.bid_passes = 0
        self.phase = "bidding_round_1"

        self.trick = []
        self.tricks_by_team = [0, 0]
        self.leader = left_of(self.dealer)
        self.current_player = self.leader
        self.trick_number = 1
        self.led_suit = None

        self.render()
        self.advance_bidding()

    @staticmethod
    def sort_hand(hand: list[Card]) -> None:
        suit_order = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3}
        rank_order = {
            Rank.NINE: 0,
            Rank.TEN: 1,
            Rank.JACK: 2,
            Rank.QUEEN: 3,
            Rank.KING: 4,
            Rank.ACE: 5,
        }
        hand.sort(key=lambda card: (suit_order[card.suit], rank_order[card.rank]))

    # ------------------------------------------------------------------
    # Bidding
    # ------------------------------------------------------------------

    def advance_bidding(self) -> None:
        assert self.upcard is not None
        upcard = self.upcard

        self.clear_actions()
        self.clear_hand_buttons()
        self.render()

        while True:
            if self.bid_round == 1 and self.bid_passes >= 4:
                self.bid_round = 2
                self.bid_player = left_of(self.dealer)
                self.bid_passes = 0
                self.phase = "bidding_round_2"

            if self.bid_round == 2 and self.bid_passes >= 4:
                self.status_var.set("Everyone passed. Redealing with the next dealer.")
                self.render()
                self.root.after(900, lambda: self.start_new_hand(advance_dealer=True))
                return

            player = self.bid_player
            is_dealer = player == self.dealer

            if player == 0:
                if self.bid_round == 1:
                    self.prompt_human_order_up()
                else:
                    self.prompt_human_call_trump()
                return

            if self.bid_round == 1:
                wants_order = self.bot_policy.choose_order_up(
                    cast(EuchreGame, self), player, upcard, is_dealer
                )
                if wants_order:
                    self.trump = upcard.suit
                    self.maker = player
                    self.status_var.set(f"{SEAT_NAMES[player]} orders up {self.trump.value}.")
                    self.dealer_pickup()
                    return
                self.status_var.set(f"{SEAT_NAMES[player]} passes.")
                self.bid_passes += 1
                self.bid_player = next_player(self.bid_player)
            else:
                chosen = self.bot_policy.choose_trump(
                    cast(EuchreGame, self), player, upcard.suit, is_dealer
                )
                if chosen is not None:
                    self.trump = chosen
                    self.maker = player
                    self.status_var.set(f"{SEAT_NAMES[player]} calls {self.trump.value}.")
                    self.start_play()
                    return
                self.status_var.set(f"{SEAT_NAMES[player]} passes.")
                self.bid_passes += 1
                self.bid_player = next_player(self.bid_player)

    def prompt_human_order_up(self) -> None:
        assert self.upcard is not None
        upcard = self.upcard

        self.phase = "human_order_up"
        dealer_note = " You are dealer." if self.dealer == 0 else f" Dealer: {SEAT_NAMES[self.dealer]}."
        self.status_var.set(f"Upcard is {upcard}. Order up {upcard.suit.value}?{dealer_note}")

        self.clear_actions()
        tk.Button(self.action_frame, text=f"Order up {upcard.suit.value}", command=self.human_orders_up).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(self.action_frame, text="Pass", command=self.human_passes_bid).pack(side=tk.LEFT, padx=4)
        self.render_hand_buttons(disabled=True)
        self.render()

    def prompt_human_call_trump(self) -> None:
        assert self.upcard is not None
        upcard = self.upcard

        self.phase = "human_call_trump"
        self.status_var.set(f"Choose trump, or pass. You cannot choose {upcard.suit.value}.")

        self.clear_actions()
        for suit in SUITS:
            if suit == upcard.suit:
                continue
            tk.Button(
                self.action_frame,
                text=f"Call {suit.value}",
                command=lambda selected_suit=suit: self.human_calls_trump(selected_suit),
            ).pack(side=tk.LEFT, padx=4)
        tk.Button(self.action_frame, text="Pass", command=self.human_passes_bid).pack(side=tk.LEFT, padx=4)
        self.render_hand_buttons(disabled=True)
        self.render()

    def human_orders_up(self) -> None:
        assert self.upcard is not None
        upcard = self.upcard

        self.trump = upcard.suit
        self.maker = 0
        self.status_var.set(f"You order up {self.trump.value}.")
        self.dealer_pickup()

    def human_calls_trump(self, suit: Suit) -> None:
        self.trump = suit
        self.maker = 0
        self.status_var.set(f"You call {self.trump.value}.")
        self.start_play()

    def human_passes_bid(self) -> None:
        self.status_var.set("You pass.")
        self.bid_passes += 1
        self.bid_player = next_player(self.bid_player)
        self.clear_actions()
        self.clear_hand_buttons()
        self.render()
        self.root.after(350, self.advance_bidding)

    def dealer_pickup(self) -> None:
        assert self.upcard is not None
        assert self.trump is not None
        upcard = self.upcard

        self.hands[self.dealer].append(upcard)
        self.sort_hand(self.hands[self.dealer])

        if self.dealer == 0:
            self.prompt_human_discard()
            return

        discard = self.bot_policy.choose_discard(cast(EuchreGame, self), self.dealer)
        self.hands[self.dealer].remove(discard)
        self.status_var.set(f"{SEAT_NAMES[self.dealer]} picks up and discards.")
        self.start_play()

    def prompt_human_discard(self) -> None:
        self.phase = "human_discard"
        self.clear_actions()
        self.status_var.set("You are dealer. Pick one card to discard.")
        self.render_hand_buttons(mode="discard")
        self.render()

    def human_discards(self, card: Card) -> None:
        self.hands[0].remove(card)
        self.status_var.set(f"You discard {card}.")
        self.start_play()

    # ------------------------------------------------------------------
    # Trick play
    # ------------------------------------------------------------------

    def start_play(self) -> None:
        assert self.trump is not None
        assert self.maker is not None
        maker = self.maker

        self.phase = "play"
        self.clear_actions()
        self.clear_hand_buttons()
        self.trick = []
        self.tricks_by_team = [0, 0]
        self.leader = left_of(self.dealer)
        self.current_player = self.leader
        self.trick_number = 1
        self.led_suit = None

        self.status_var.set(
            f"Trump is {self.trump.value}. Maker: {SEAT_NAMES[maker]} / Team {team_of(maker)}."
        )
        self.render()
        self.root.after(500, self.advance_play)

    def advance_play(self) -> None:
        assert self.trump is not None
        trump = self.trump

        self.clear_actions()
        self.clear_hand_buttons()
        self.render()

        if len(self.trick) == 4:
            self.finish_trick()
            return

        player = self.current_player
        legal = legal_cards(self.hands[player], trump, self.led_suit)

        if player == 0:
            self.prompt_human_play_card(legal)
            return

        card = self.bot_policy.choose_card(cast(EuchreGame, self), player, legal, self.trick)
        self.play_card(player, card)
        self.status_var.set(f"{SEAT_NAMES[player]} plays {card}.")
        self.render()
        self.root.after(500, self.advance_play)

    def prompt_human_play_card(self, legal: list[Card]) -> None:
        self.phase = "human_play"
        if self.trick:
            self.status_var.set("Your turn. Follow suit if possible.")
        else:
            self.status_var.set("Your turn. You are leading this trick.")
        self.render_hand_buttons(mode="play", legal=legal)
        self.render()

    def human_plays_card(self, card: Card) -> None:
        self.play_card(0, card)
        self.status_var.set(f"You play {card}.")
        self.clear_hand_buttons()
        self.render()
        self.root.after(350, self.advance_play)

    def play_card(self, player: int, card: Card) -> None:
        assert self.trump is not None
        trump = self.trump

        if self.led_suit is None:
            self.led_suit = effective_suit(card, trump)
        self.hands[player].remove(card)
        self.trick.append((player, card))
        self.current_player = next_player(player)

    def finish_trick(self) -> None:
        assert self.trump is not None
        trump = self.trump

        winner = trick_winner(self.trick, trump)
        self.tricks_by_team[team_of(winner)] += 1
        self.leader = winner
        self.current_player = winner

        self.phase = "trick_over"
        winner_name = SEAT_NAMES[winner]
        win_verb = "win" if winner == 0 else "wins"
        self.status_var.set(
            f"{SEAT_NAMES[winner]} {win_verb} trick {self.trick_number}. "
            f"Tricks: Team 0 = {self.tricks_by_team[0]}, Team 1 = {self.tricks_by_team[1]}."
        )
        self.clear_actions()

        if self.trick_number >= 5:
            tk.Button(self.action_frame, text="Score Hand", command=self.finish_hand).pack(side=tk.LEFT, padx=4)
        else:
            tk.Button(self.action_frame, text="Next Trick", command=self.start_next_trick).pack(side=tk.LEFT, padx=4)

        self.render()

    def start_next_trick(self) -> None:
        self.trick_number += 1
        self.trick = []
        self.led_suit = None
        self.phase = "play"
        self.clear_actions()
        self.status_var.set(f"Starting trick {self.trick_number}. Leader: {SEAT_NAMES[self.leader]}.")
        self.render()
        self.root.after(350, self.advance_play)

    def finish_hand(self) -> None:
        assert self.maker is not None
        maker = self.maker

        maker_team = team_of(maker)
        maker_tricks = self.tricks_by_team[maker_team]
        points_by_team = [0, 0]

        if maker_tricks >= 3:
            points_by_team[maker_team] = 2 if maker_tricks == 5 else 1
        else:
            points_by_team[1 - maker_team] = 2

        self.scores[0] += points_by_team[0]
        self.scores[1] += points_by_team[1]

        if max(self.scores) >= self.winning_score:
            winner = 0 if self.scores[0] > self.scores[1] else 1
            self.status_var.set(f"Team {winner} wins the game {self.scores[winner]}-{self.scores[1 - winner]}!")
            self.phase = "game_over"
            self.clear_actions()
            tk.Button(self.action_frame, text="New Game", command=self.new_game).pack(side=tk.LEFT, padx=4)
            self.render()
            return

        if points_by_team[maker_team] > 0:
            summary = f"Team {maker_team} made the hand and scores {points_by_team[maker_team]}."
        else:
            euchred_team = 1 - maker_team
            summary = f"Team {maker_team} was euchred. Team {euchred_team} scores 2."

        self.status_var.set(summary)
        self.clear_actions()
        tk.Button(self.action_frame, text="Next Hand", command=lambda: self.start_new_hand(advance_dealer=True)).pack(
            side=tk.LEFT, padx=4
        )
        self.phase = "hand_over"
        self.render()

    def new_game(self) -> None:
        self.scores = [0, 0]
        self.dealer = 0
        self.start_new_hand(advance_dealer=False)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> None:
        self.score_var.set(
            f"Team 0: {self.scores[0]}    Team 1: {self.scores[1]}    Dealer: {SEAT_NAMES[self.dealer]}"
        )
        self.canvas.delete("all")

        # Draw objects first, labels last. This prevents cards from hiding text.
        self.draw_upcard_and_trump()
        self.draw_bot_hands()
        self.draw_trick()
        self.draw_table_labels()

        if self.phase not in {"human_play", "human_discard", "human_order_up", "human_call_trump"}:
            self.render_hand_buttons(disabled=True)

    def draw_table_labels(self) -> None:
        positions = {
            0: (TABLE_WIDTH // 2, TABLE_HEIGHT - 92),
            1: (115, 178),
            2: (TABLE_WIDTH // 2, 26),
            3: (TABLE_WIDTH - 115, 178),
        }

        for player, (x, y) in positions.items():
            dealer_mark = " (D)" if player == self.dealer else ""
            label = f"{SEAT_NAMES[player]}{dealer_mark}\nTeam {team_of(player)}"

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
        # Compact bottom-right panel: below the right bot's hand and beside the user's area.
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

        if self.upcard is not None:
            self.draw_card(self.upcard, card_x, card_y, faded=self.trump is not None)

        trump_text = f"Trump: {self.trump.value}" if self.trump else "Trump: not chosen"
        if self.maker is None:
            maker_text = "Maker: none"
        else:
            maker_text = f"Maker: {SEAT_NAMES[self.maker]}"

        self.canvas.create_text(
            card_x + CARD_WIDTH // 2,
            card_y + CARD_HEIGHT + 30,
            text=f"{trump_text}\n{maker_text}",
            fill="white",
            font=("Helvetica", 10, "bold"),
            justify=tk.CENTER,
        )

    def draw_bot_hands(self) -> None:
        self.draw_card_backs(center_x=TABLE_WIDTH // 2, y=64, count=len(self.hands[2]), horizontal=True)
        self.draw_card_backs(center_x=72, y=230, count=len(self.hands[1]), horizontal=False)
        self.draw_card_backs(center_x=TABLE_WIDTH - 72, y=230, count=len(self.hands[3]), horizontal=False)

    def draw_trick(self) -> None:
        positions = {
            0: (TABLE_WIDTH // 2 - CARD_WIDTH // 2, TABLE_HEIGHT // 2 + 72),
            1: (TABLE_WIDTH // 2 - 165, TABLE_HEIGHT // 2 - CARD_HEIGHT // 2),
            2: (TABLE_WIDTH // 2 - CARD_WIDTH // 2, TABLE_HEIGHT // 2 - 118),
            3: (TABLE_WIDTH // 2 + 100, TABLE_HEIGHT // 2 - CARD_HEIGHT // 2),
        }

        for player, card in self.trick:
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
        self.canvas.create_rectangle(x, y, x + CARD_WIDTH, y + CARD_HEIGHT, fill=fill, outline=outline, width=2)

        color = "red" if card.suit in {Suit.HEARTS, Suit.DIAMONDS} else "black"
        self.canvas.create_text(
            x + CARD_WIDTH // 2,
            y + CARD_HEIGHT // 2,
            text=str(card),
            fill=color,
            font=("Helvetica", 18, "bold"),
        )

        if self.trump is not None:
            if is_right_bower(card, self.trump):
                tag = "R"
            elif is_left_bower(card, self.trump):
                tag = "L"
            else:
                tag = ""
            if tag:
                self.canvas.create_text(
                    x + CARD_WIDTH - 11,
                    y + 11,
                    text=tag,
                    fill="blue",
                    font=("Helvetica", 10, "bold"),
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

    def draw_card_back(self, x: int, y: int) -> None:
        self.canvas.create_rectangle(x, y, x + CARD_WIDTH, y + CARD_HEIGHT, fill="#0b3d91", outline="white", width=2)
        self.canvas.create_rectangle(x + 8, y + 8, x + CARD_WIDTH - 8, y + CARD_HEIGHT - 8, outline="white")
        self.canvas.create_text(
            x + CARD_WIDTH // 2,
            y + CARD_HEIGHT // 2,
            text="★",
            fill="white",
            font=("Helvetica", 22, "bold"),
        )

    def render_hand_buttons(
        self,
        mode: str = "disabled",
        legal: Optional[list[Card]] = None,
        disabled: bool = False,
    ) -> None:
        self.clear_hand_buttons()
        if not self.hands[0]:
            return

        legal_set = set(legal or [])

        for card in self.hands[0]:
            text = str(card)
            if self.trump is not None and is_right_bower(card, self.trump):
                text += "  R"
            elif self.trump is not None and is_left_bower(card, self.trump):
                text += "  L"

            state = tk.NORMAL
            command: Optional[Callable[[], None]] = None
            is_illegal = False

            if disabled or mode == "disabled":
                state = tk.DISABLED
            elif mode == "discard":
                command = lambda selected_card=card: self.human_discards(selected_card)
            elif mode == "play":
                if card in legal_set:
                    command = lambda selected_card=card: self.human_plays_card(selected_card)
                else:
                    state = tk.DISABLED
                    is_illegal = True

            suit_color = "red" if card.suit in {Suit.HEARTS, Suit.DIAMONDS} else "black"
            foreground = "#888888" if is_illegal else suit_color
            background = "#f2f2f2" if is_illegal else "white"

            if command is None:
                button = tk.Button(
                    self.hand_frame,
                    text=text,
                    width=10,
                    height=2,
                    state=state,
                    bg=background,
                    fg=foreground,
                    activebackground="#eeeeee",
                    activeforeground=suit_color,
                    disabledforeground=foreground,
                    relief=tk.RAISED,
                    font=("Helvetica", 13, "bold"),
                )
            else:
                button = tk.Button(
                    self.hand_frame,
                    text=text,
                    width=10,
                    height=2,
                    state=state,
                    command=command,
                    bg=background,
                    fg=foreground,
                    activebackground="#eeeeee",
                    activeforeground=suit_color,
                    disabledforeground=foreground,
                    relief=tk.RAISED,
                    font=("Helvetica", 13, "bold"),
                )

            button.pack(side=tk.LEFT, padx=4)


def main() -> None:
    root = tk.Tk()
    EuchreGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
