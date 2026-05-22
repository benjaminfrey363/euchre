from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional, cast, Union, Protocol

from logic.cards import Card, Suit, make_deck, effective_suit
from logic.rules import legal_cards, next_player, left_of, team_of, trick_winner
from logic.policies import HandResult, Policy, EuchreGame

from collections.abc import Sequence

@dataclass(frozen=True)
class GameResult:
    winner_team: int
    final_score: tuple[int, int]
    hands_played: int


@dataclass(frozen=True)
class SimulationStats:
    games_played: int
    team_wins: tuple[int, int]
    average_score: tuple[float, float]
    average_hands_per_game: float









@dataclass(frozen=True)
class OrderUpAction:
    order_up: bool


@dataclass(frozen=True)
class CallTrumpAction:
    suit: Optional[Suit]


@dataclass(frozen=True)
class DiscardAction:
    card: Card


@dataclass(frozen=True)
class PlayCardAction:
    card: Card


Action = Union[
    OrderUpAction,
    CallTrumpAction,
    DiscardAction,
    PlayCardAction,
]


@dataclass(frozen=True)
class Observation:
    """
    What a player is allowed to know.

    This is not yet encoded numerically for ML. It is a clean symbolic
    representation that we can later convert into tensors/features.
    """

    player: int
    dealer: int
    scores: tuple[int, int]
    hand: tuple[Card, ...]
    upcard: Optional[Card]
    trump: Optional[Suit]
    maker: Optional[int]
    trick: tuple[tuple[int, Card], ...]
    tricks_by_team: tuple[int, int]
    phase: str
    legal_actions: tuple[Action, ...]



class ActionPolicy(Protocol):
    def choose_action(self, observation: Observation) -> Action:
        ...


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: dict[str, object]






class EuchreEnv:
    """
    Headless Euchre simulator for bot-vs-bot games.

    This class deliberately has no input(), no print(), and no GUI code.
    It is the bridge between the current playable game and future ML training.

    Current limitations:
    - Four-player fixed-partner Euchre.
    - No going alone yet.
    - Policies must choose legal actions themselves, but this environment
      enforces legal card play through legal_cards().
    - If everyone passes both bidding rounds, the hand is redealt with the
      next dealer and no points are awarded.

    Note: Policy methods are currently typed against EuchreGame in
    logic.policies. This environment exposes the same fields the policies use,
    so we cast self to EuchreGame when calling policies. Later, we should replace
    that with a smaller GameView Protocol.
    """

    def __init__(
        self,
        policies: Sequence[Policy],
        winning_score: int = 10,
        seed: Optional[int] = None,
    ):
        if len(policies) != 4:
            raise ValueError("EuchreEnv requires exactly four policies.")

        self.policies = list(policies)
        self.winning_score = winning_score
        self.random = random.Random(seed)

        self.dealer = 0
        self.scores = [0, 0]

        self.hands: list[list[Card]] = [[] for _ in range(4)]
        self.kitty: list[Card] = []
        self.upcard: Optional[Card] = None
        self.trump: Optional[Suit] = None
        self.maker: Optional[int] = None

        self.phase = "not_started"
        self.current_player = 0
        self.trick: list[tuple[int, Card]] = []
        self.tricks_by_team = [0, 0]
        self.led_suit: Optional[Suit] = None

    def reset_game(self) -> None:
        self.dealer = 0
        self.scores = [0, 0]
        self.hands = [[] for _ in range(4)]
        self.kitty = []
        self.upcard = None
        self.trump = None
        self.maker = None

        self.phase = "not_started"
        self.current_player = 0
        self.trick = []
        self.tricks_by_team = [0, 0]
        self.led_suit = None

    
    def observation_for_player(self, player: int) -> Observation:
        if player not in {0, 1, 2, 3}:
            raise ValueError(f"Invalid player index: {player}")

        return Observation(
            player=player,
            dealer=self.dealer,
            scores=(self.scores[0], self.scores[1]),
            hand=tuple(self.hands[player]),
            upcard=self.upcard,
            trump=self.trump,
            maker=self.maker,
            trick=tuple(self.trick),
            tricks_by_team=(self.tricks_by_team[0], self.tricks_by_team[1]),
            phase=self.phase,
            legal_actions=tuple(self.legal_actions_for_player(player)),
        )


    def legal_actions_for_player(self, player: int) -> list[Action]:
        """
        Symbolic legal actions for the given player in the current phase.

        This is the method future ML policies will use before choosing an action.
        """

        if self.phase == "bidding_round_1":
            actions: list[Action] = [
                OrderUpAction(order_up=False),
                OrderUpAction(order_up=True),
            ]
            return actions

        if self.phase == "bidding_round_2":
            assert self.upcard is not None
            
            actions: list[Action] = [CallTrumpAction(suit=None)]
            actions.extend(
                CallTrumpAction(suit=suit)
                for suit in Suit
                if suit != self.upcard.suit
            )
            return actions

        if self.phase == "discard":
            actions: list[Action] = [
                DiscardAction(card=card)
                for card in self.hands[player]
            ]
            return actions

        if self.phase == "play_card":
            assert self.trump is not None
            legal = legal_cards(self.hands[player], self.trump, self.led_suit)
            actions: list[Action] = [
                PlayCardAction(card=card)
                for card in legal
            ]
            return actions

        return []

    def deal(self) -> None:
        deck = make_deck()
        self.random.shuffle(deck)

        self.hands = [deck[i * 5 : (i + 1) * 5] for i in range(4)]
        self.upcard = deck[20]
        self.kitty = deck[21:]
        self.trump = None
        self.maker = None

        self.phase = "bidding_round_1"
        self.current_player = left_of(self.dealer)
        self.trick = []
        self.tricks_by_team = [0, 0]
        self.led_suit = None

    def bid_hand(self) -> bool:
        """
        Returns True if trump is chosen, False if all players pass.
        """
        assert self.upcard is not None
        upcard = self.upcard

        # Round 1: order up the upcard suit.
        player = left_of(self.dealer)
        for _ in range(4):
            self.phase = "bidding_round_1"
            self.current_player = player
            is_dealer = player == self.dealer
            wants_order = self.policies[player].choose_order_up(
                cast(EuchreGame, self),
                player,
                upcard,
                is_dealer,
            )
            if wants_order:
                self.trump = upcard.suit
                self.maker = player
                self.dealer_pickup()
                return True
            player = next_player(player)

        # Round 2: choose any suit except the upcard suit.
        player = left_of(self.dealer)
        for _ in range(4):
            self.phase = "bidding_round_2"
            self.current_player = player
            is_dealer = player == self.dealer
            chosen = self.policies[player].choose_trump(
                cast(EuchreGame, self),
                player,
                upcard.suit,
                is_dealer,
            )
            if chosen is not None:
                self.trump = chosen
                self.maker = player
                return True
            player = next_player(player)

        return False

    def dealer_pickup(self) -> None:
        assert self.upcard is not None
        assert self.trump is not None

        self.phase = "discard"
        self.current_player = self.dealer

        self.hands[self.dealer].append(self.upcard)
        discard = self.policies[self.dealer].choose_discard(cast(EuchreGame, self), self.dealer)

        if discard not in self.hands[self.dealer]:
            raise ValueError(f"Dealer policy tried to discard a card not in hand: {discard}")

        self.hands[self.dealer].remove(discard)

    def play_tricks(self) -> list[int]:
        assert self.trump is not None
        trump = self.trump

        self.tricks_by_team = [0, 0]
        leader = left_of(self.dealer)

        for _ in range(5):
            self.trick: list[tuple[int, Card]] = []
            self.led_suit: Optional[Suit] = None
            player = leader

            for _ in range(4):

                self.phase = "play_card"
                self.current_player = player

                legal = legal_cards(self.hands[player], trump, self.led_suit)
                card = self.policies[player].choose_card(
                    cast(EuchreGame, self),
                    player,
                    legal,
                    self.trick,
                )

                if card not in legal:
                    raise ValueError(
                        f"Policy for player {player} tried to play illegal card {card}; "
                        f"legal cards were {legal}."
                    )

                self.hands[player].remove(card)
                self.trick.append((player, card))

                if self.led_suit is None:
                    self.led_suit = effective_suit(card, trump)

                player = next_player(player)

            winner = trick_winner(self.trick, trump)
            self.tricks_by_team[team_of(winner)] += 1
            leader = winner


        return self.tricks_by_team

    def score_hand(self, tricks_by_team: list[int]) -> HandResult:
        assert self.maker is not None
        maker_team = team_of(self.maker)
        maker_tricks = tricks_by_team[maker_team]
        points_by_team = [0, 0]

        if maker_tricks >= 3:
            points_by_team[maker_team] = 2 if maker_tricks == 5 else 1
        else:
            points_by_team[1 - maker_team] = 2

        self.scores[0] += points_by_team[0]
        self.scores[1] += points_by_team[1]

        return HandResult(
            maker_team=maker_team,
            tricks_by_team=tricks_by_team,
            points_by_team=points_by_team,
        )

    def play_hand(self) -> Optional[HandResult]:
        self.deal()

        trump_chosen = self.bid_hand()
        if not trump_chosen:
            self.dealer = next_player(self.dealer)
            return None

        tricks_by_team = self.play_tricks()
        result = self.score_hand(tricks_by_team)
        self.dealer = next_player(self.dealer)
        return result

    def play_game(self) -> GameResult:
        self.reset_game()
        hands_played = 0

        while max(self.scores) < self.winning_score:
            self.play_hand()
            hands_played += 1

        winner_team = 0 if self.scores[0] > self.scores[1] else 1
        return GameResult(
            winner_team=winner_team,
            final_score=(self.scores[0], self.scores[1]),
            hands_played=hands_played,
        )

    def run_many_games(self, n_games: int) -> SimulationStats:
        if n_games <= 0:
            raise ValueError("n_games must be positive.")

        wins = [0, 0]
        total_scores = [0, 0]
        total_hands = 0

        for _ in range(n_games):
            result = self.play_game()
            wins[result.winner_team] += 1
            total_scores[0] += result.final_score[0]
            total_scores[1] += result.final_score[1]
            total_hands += result.hands_played

        return SimulationStats(
            games_played=n_games,
            team_wins=(wins[0], wins[1]),
            average_score=(total_scores[0] / n_games, total_scores[1] / n_games),
            average_hands_per_game=total_hands / n_games,
        )
    
    def apply_order_up_action(self, player: int, action: OrderUpAction) -> bool:
        if self.phase != "bidding_round_1":
            raise ValueError(f"Cannot order up during phase {self.phase}")

        if not action.order_up:
            return False

        assert self.upcard is not None
        self.trump = self.upcard.suit
        self.maker = player

        self.hands[self.dealer].append(self.upcard)
        self.phase = "discard"
        self.current_player = self.dealer

        return True


    def apply_call_trump_action(self, player: int, action: CallTrumpAction) -> bool:
        if self.phase != "bidding_round_2":
            raise ValueError(f"Cannot call trump during phase {self.phase}")

        if action.suit is None:
            return False

        assert self.upcard is not None
        if action.suit == self.upcard.suit:
            raise ValueError("Cannot call the turned-down upcard suit in bidding round 2.")

        self.trump = action.suit
        self.maker = player
        return True


    def apply_discard_action(self, player: int, action: DiscardAction) -> None:
        if self.phase != "discard":
            raise ValueError(f"Cannot discard during phase {self.phase}")

        if action.card not in self.hands[player]:
            raise ValueError(f"Player {player} cannot discard {action.card}; card is not in hand.")

        self.hands[player].remove(action.card)


    def apply_play_card_action(self, player: int, action: PlayCardAction) -> None:
        if self.phase != "play_card":
            raise ValueError(f"Cannot play a card during phase {self.phase}")

        assert self.trump is not None
        legal = legal_cards(self.hands[player], self.trump, self.led_suit)

        if action.card not in legal:
            raise ValueError(
                f"Player {player} cannot play {action.card}; legal cards are {legal}."
            )

        self.hands[player].remove(action.card)
        self.trick.append((player, action.card))

        if self.led_suit is None:
            self.led_suit = effective_suit(action.card, self.trump)

    def choose_action_for_player(
        self,
        player: int,
        policy: ActionPolicy,
    ) -> Action:
        observation = self.observation_for_player(player)
        action = policy.choose_action(observation)

        if action not in observation.legal_actions:
            raise ValueError(
                f"Policy chose illegal action {action}; legal actions were {observation.legal_actions}."
            )

        return action
    
    def reset(self) -> Observation:
        """
        Start a fresh game and return the first player's observation.
        """
        self.reset_game()
        self.deal()

        assert self.current_player in {0, 1, 2, 3}
        return self.observation_for_player(self.current_player)
    

    def advance_to_next_bidder(self) -> None:
        self.current_player = next_player(self.current_player)

        # If we have returned to the player left of dealer after round 1,
        # move to round 2.
        if self.phase == "bidding_round_1" and self.current_player == left_of(self.dealer):
            self.phase = "bidding_round_2"

        # If round 2 also makes a full loop, redeal with next dealer.
        elif self.phase == "bidding_round_2" and self.current_player == left_of(self.dealer):
            self.dealer = next_player(self.dealer)
            self.deal()


    def begin_play(self) -> None:
        if self.trump is None:
            raise ValueError("Cannot begin play before trump is chosen.")

        self.phase = "play_card"
        self.trick = []
        self.tricks_by_team = [0, 0]
        self.led_suit = None
        self.current_player = left_of(self.dealer)


    def finish_trick_if_complete(self) -> Optional[int]:
        """
        Finish the trick if four cards have been played.

        Returns the trick winner if a trick ended, otherwise None.
        """
        if len(self.trick) < 4:
            return None

        assert self.trump is not None

        winner = trick_winner(self.trick, self.trump)
        self.tricks_by_team[team_of(winner)] += 1

        self.trick = []
        self.led_suit = None
        self.current_player = winner

        return winner
    

    def finish_hand_if_complete(self) -> Optional[HandResult]:
        """
        Score the hand after all five tricks are complete.

        Returns a HandResult if the hand ended, otherwise None.
        """
        if sum(self.tricks_by_team) < 5:
            return None

        result = self.score_hand(self.tricks_by_team)

        if max(self.scores) >= self.winning_score:
            self.phase = "game_over"
        else:
            self.dealer = next_player(self.dealer)
            self.deal()

        return result
    
    def step(self, action: Action) -> StepResult:
        """
        Apply one action for the current player.

        Returns:
            StepResult(observation, reward, done, info)

        Reward convention for now:
        - 0.0 during the game
        - +1.0 to players on the winning team at game end
        - -1.0 to players on the losing team at game end

        Since this method returns only the next current player's observation,
        the reward is from that next player's team perspective. Later, for RL,
        we may want per-player or per-team reward records instead.
        """
        player = self.current_player
        legal_actions = self.legal_actions_for_player(player)

        if action not in legal_actions:
            raise ValueError(
                f"Illegal action {action} for player {player}; legal actions are {legal_actions}."
            )

        info: dict[str, object] = {
            "player": player,
            "phase": self.phase,
        }

        if self.phase == "bidding_round_1":
            assert isinstance(action, OrderUpAction)
            trump_chosen = self.apply_order_up_action(player, action)

            if not trump_chosen:
                self.advance_to_next_bidder()

        elif self.phase == "bidding_round_2":
            assert isinstance(action, CallTrumpAction)
            trump_chosen = self.apply_call_trump_action(player, action)

            if trump_chosen:
                self.begin_play()
            else:
                self.advance_to_next_bidder()

        elif self.phase == "discard":
            assert isinstance(action, DiscardAction)
            self.apply_discard_action(player, action)
            self.begin_play()

        elif self.phase == "play_card":
            assert isinstance(action, PlayCardAction)
            self.apply_play_card_action(player, action)

            if len(self.trick) == 4:
                winner = self.finish_trick_if_complete()
                info["trick_winner"] = winner

                hand_result = self.finish_hand_if_complete()
                if hand_result is not None:
                    info["hand_result"] = hand_result
            else:
                self.current_player = next_player(player)

        else:
            raise ValueError(f"Cannot step during phase {self.phase}")

        done = self.phase == "game_over"

        reward = 0.0
        if done:
            winning_team = 0 if self.scores[0] > self.scores[1] else 1
            reward = 1.0 if team_of(self.current_player) == winning_team else -1.0
            info["winning_team"] = winning_team
            info["final_score"] = (self.scores[0], self.scores[1])

        return StepResult(
            observation=self.observation_for_player(self.current_player),
            reward=reward,
            done=done,
            info=info,
        )
    



class RandomActionPolicy:
    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)

    def choose_action(self, observation: Observation) -> Action:
        if not observation.legal_actions:
            raise ValueError("No legal actions available.")

        return self.random.choice(observation.legal_actions)


