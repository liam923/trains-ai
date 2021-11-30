from __future__ import annotations

import functools
import operator
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Callable, Dict, Union, Iterable, Optional, DefaultDict, Tuple, Type

from trains.game.box import Player
from trains.game.state import State


@dataclass(frozen=True)
class Utility(DefaultDict[Optional[Player], float]):
    """
    A class that holds the utilities for each player
    """

    def __init__(
        self,
        cards: Union[
            Iterable[Tuple[Optional[Player], float]],
            Dict[Optional[Player], float],
            Type[float],
            None,
        ] = None,
    ):
        if isinstance(cards, dict):
            super().__init__(float, cards.items())
        elif cards is None:
            super().__init__(float, {})
        elif isinstance(cards, type):
            super().__init__(cards)
        else:
            super().__init__(float, cards)

    def __add__(self, other: Union[Utility, float]) -> Utility:
        if isinstance(other, int) or isinstance(other, float):
            return Utility({key: val + other for key, val in self.items()})
        else:
            return Utility(
                {
                    key: self.get(key, 0) + other.get(key, 0)
                    for key in self.keys() | other.keys()
                }
            )

    def __radd__(self, other: float) -> Utility:
        return Utility({key: val + other for key, val in self.items()})

    def __mul__(self, other: float) -> Utility:
        return Utility({key: val * other for key, val in self.items()})

    def __rmul__(self, other: float) -> Utility:
        return self.__mul__(other)

    def __truediv__(self, other: float) -> Utility:
        return Utility({key: val / other for key, val in self.items()})

    @classmethod
    def sum(cls, utilities: Iterable[Utility]) -> Utility:
        return functools.reduce(operator.add, utilities, cls())


class UtilityFunction(ABC):
    def __call__(self, state: State) -> Utility:
        return self.calculate_utility(state)

    @abstractmethod
    def calculate_utility(self, state: State) -> Utility:
        pass


class BuildRoutesUf(UtilityFunction):
    """
    A utility function meant for testing that only rewards players for building routes
    """

    def calculate_utility(self, state: State) -> Utility:
        utility = Utility()
        for player in state.box.players:
            route_points = sum(
                state.box.route_point_values[route.length]
                for route, builder in state.built_routes.items()
                if builder == player
            )
            if player == state.player:
                max_train_cards = max(state.hand.train_cards.values(), default=0)
            else:
                max_train_cards = max(
                    state.opponent_hands[player].known_train_cards.values(), default=0
                )
            utility[player] = route_points + max_train_cards * 0.75
        return utility
