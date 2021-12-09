from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    FrozenSet,
    TypeVar,
    Type,
    Dict,
    Generator,
    Callable,
    Optional,
    Tuple,
    Union,
    DefaultDict,
)

import math
from frozendict import frozendict

from trains.game.action import Action
from trains.game.box import DestinationCard, TrainCards, Player, Box, TrainCard, Route
from trains.game.clusters import Clusters
from trains.game.turn import TurnState, GameOverTurn
from trains.mypy_util import add_slots
from trains.util import randomly_sample_distribution


@add_slots
@dataclass(frozen=True)
class ObservedHandState:
    known_destination_cards: FrozenSet[DestinationCard]
    destination_cards_count: int
    known_unselected_destination_cards: FrozenSet[DestinationCard]
    unselected_destination_cards_count: int
    known_train_cards: TrainCards
    train_cards_count: int
    remaining_trains: int

    known_points_so_far: int
    known_complete_destination_cards: FrozenSet[DestinationCard]
    known_incomplete_destination_cards: FrozenSet[DestinationCard]


@add_slots
@dataclass(frozen=True)
class KnownHandState:
    destination_cards: FrozenSet[DestinationCard]
    unselected_destination_cards: FrozenSet[DestinationCard]
    train_cards: TrainCards
    remaining_trains: int

    points_so_far: int
    complete_destination_cards: FrozenSet[DestinationCard]
    incomplete_destination_cards: FrozenSet[DestinationCard]

    @property
    def known_destination_cards(self) -> FrozenSet[DestinationCard]:
        return self.destination_cards

    @property
    def destination_cards_count(self) -> int:
        return len(self.destination_cards)

    @property
    def known_unselected_destination_cards(self) -> FrozenSet[DestinationCard]:
        return self.unselected_destination_cards

    @property
    def unselected_destination_cards_count(self) -> int:
        return len(self.unselected_destination_cards)

    @property
    def known_train_cards(self) -> TrainCards:
        return self.train_cards

    @property
    def train_cards_count(self) -> int:
        return len(self.train_cards)

    @property
    def known_points_so_far(self) -> int:
        return self.points_so_far

    @property
    def known_complete_destination_cards(self) -> FrozenSet[DestinationCard]:
        return self.complete_destination_cards

    @property
    def known_incomplete_destination_cards(self) -> FrozenSet[DestinationCard]:
        return self.incomplete_destination_cards


HandState = Union[ObservedHandState, KnownHandState]


@dataclass(frozen=True)  # type: ignore
class AbstractState(ABC):
    box: Box
    discarded_train_cards: TrainCards
    face_up_train_cards: TrainCards
    train_card_pile_distribution: TrainCards
    destination_card_pile_distribution: FrozenSet[DestinationCard]
    destination_card_pile_size: int
    built_routes: frozendict[Route, Player]
    built_clusters: frozendict[Player, Clusters]
    turn_state: TurnState

    @property
    @abstractmethod
    def player_hands(
        self,
    ) -> Union[Dict[Player, HandState], frozendict[Player, KnownHandState]]:
        pass

    @abstractmethod
    def player_hand(self, player: Player) -> KnownHandState:
        pass

    @classmethod
    @abstractmethod
    def make(cls: Type[State], box: Box, player: Player) -> State:
        pass

    @abstractmethod
    def next_state(self: State, action: Action) -> State:
        pass

    @dataclass(frozen=True)
    class LegalAction:
        """
        Represents a possible legal action to take. The probability is the probability
        of the actor having the right cards for the move.
        """

        action: Action
        probability: float = 1

    @abstractmethod
    def get_legal_actions(self) -> Generator[LegalAction, None, None]:
        pass

    @abstractmethod
    def assumed_hands(
        self: State,
        player: Player,
        route_building_probability_calculator: Callable[
            [Optional[FrozenSet[DestinationCard]]], float
        ] = lambda _: 1,
    ) -> Generator[Tuple[State, float], None, None]:
        """
        Generate a list of all possible states where the given player's hand is fully
        known, together with the probability of the player having the hand.

        Args:
            player: The player whose hand is being assumed
            route_building_probability_calculator: A function that given a set of
                destination cards returns the probability that the player would build
                the routes that they have built given those destination cards. If the
                input is None, the it returns the probability of building the routes
                disregarding destination cards
        """
        pass

    @abstractmethod
    def hand_is_known(self, player: Player) -> bool:
        """
        Determine if the state has full knowledge of the hand of the given player
        """
        pass

    @classmethod
    def _deal_train_cards(
        cls, cards: int, deck: TrainCards
    ) -> Generator[Tuple[TrainCards, float], None, None]:
        # this function was very slow, so I replaced it with a monte-carlo solution
        # the original code is below
        results: DefaultDict[TrainCards, int] = defaultdict(int)
        mc_count = 100
        for _ in range(mc_count):
            result: DefaultDict[TrainCard, int] = defaultdict(int)
            current_deck = dict(deck)
            for _ in range(min(cards, deck.total)):
                drawn_card = next(randomly_sample_distribution(current_deck.items(), 1))
                current_deck[drawn_card] -= 1
                result[drawn_card] += 1
            results[TrainCards(result)] += 1
        for drawn_cards, count in results.items():
            yield drawn_cards, count / mc_count
        return None

        # def _deal_train_cards_helper(
        #     remaining_cards: int,
        #     current_deck: Optional[Cons[Tuple[TrainCard, int]]],
        #     deal_so_far: Optional[Cons[Tuple[TrainCard, int]]] = None,
        # ) -> Generator[TrainCards, None, None]:
        #     if remaining_cards == 0 or current_deck is None:
        #         yield TrainCards(Cons.iterate(deal_so_far))
        #     else:
        #         color, color_cards = current_deck.head
        #         max_count = max(color_cards, remaining_cards)
        #         for count in range(0, max_count + 1):
        #             yield from _deal_train_cards_helper(
        #                 remaining_cards - count,
        #                 current_deck.rest,
        #                 Cons((color, count), deal_so_far),
        #             )
        #
        # for train_cards in _deal_train_cards_helper(cards, Cons.make(deck.items())):
        #     yield train_cards, probability_of_having_cards(train_cards, cards, deck)

    @classmethod
    def _deal_destination_cards(
        cls, cards: int, destination_cards: FrozenSet[DestinationCard]
    ) -> Generator[Tuple[FrozenSet[DestinationCard], float], None, None]:
        prob = 1 / math.comb(len(destination_cards), cards)
        for result_cards in itertools.combinations(destination_cards, cards):
            yield frozenset(result_cards), prob

    def is_game_over(self) -> bool:
        return isinstance(self.turn_state, GameOverTurn)

    def winner(self) -> Player:
        return max(self.player_hands.items(), key=lambda p: p[1].known_points_so_far)[0]


State = TypeVar("State", bound=AbstractState)
