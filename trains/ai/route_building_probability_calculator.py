from abc import ABC, abstractmethod
from typing import Optional, FrozenSet

from trains.game.box import DestinationCard
from trains.game.state import State


class RouteBuildingProbabilityCalculator(ABC):
    def __init__(self, state: State):
        self.state = state

    def probability_of_building_routes(
        self, destination_cards: Optional[FrozenSet[DestinationCard]] = None
    ) -> float:
        if destination_cards is None:
            return self._probability_of_building_routes_ignoring_cards()
        else:
            return self._probability_of_building_routes_with_cards(destination_cards)

    @abstractmethod
    def _probability_of_building_routes_with_cards(
        self, destination_cards: FrozenSet[DestinationCard]
    ) -> float:
        pass

    @abstractmethod
    def _probability_of_building_routes_ignoring_cards(self) -> float:
        pass

    def __call__(
        self, destination_cards: Optional[FrozenSet[DestinationCard]] = None
    ) -> float:
        return self.probability_of_building_routes(destination_cards)


class DummyRbpc(RouteBuildingProbabilityCalculator):
    def _probability_of_building_routes_ignoring_cards(self) -> float:
        return 1

    def _probability_of_building_routes_with_cards(
        self, destination_cards: FrozenSet[DestinationCard]
    ) -> float:
        return 1
