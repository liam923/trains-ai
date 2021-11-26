from __future__ import annotations

import functools
import operator
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


UtilityFunction = Callable[[State], Utility]
