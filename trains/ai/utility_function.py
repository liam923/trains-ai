from __future__ import annotations

import functools
import operator
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Union,
    Iterable,
    Optional,
    DefaultDict,
    Tuple,
    Type,
    Generic,
    Mapping,
)

import math

from trains.game.box import Player, Box
from trains.game.state import (
    AbstractState,
    State,
    HandState,
)
from trains.mypy_util import cache
from trains.util import (
    cards_needed_to_build_routes,
    best_routes_between_cities,
    best_routes_between_many_cities,
)


@dataclass(frozen=True)
class Utility(DefaultDict[Optional[Player], float]):
    """
    A class that holds the utilities for each player
    """

    def __init__(
        self,
        values: Union[
            Iterable[Tuple[Optional[Player], float]],
            Mapping[Optional[Player], float],
            Mapping[Player, float],
            Type[float],
            None,
        ] = None,
    ):
        if isinstance(values, Mapping):
            super().__init__(float, values.items())
        elif values is None:
            super().__init__(float, {})
        elif isinstance(values, type):
            super().__init__(values)
        else:
            super().__init__(float, values)

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


class UtilityFunction(Generic[State], ABC):
    def __call__(self, state: State) -> Utility:
        return self.calculate_utility(state)

    @abstractmethod
    def calculate_utility(self, state: State) -> Utility:
        pass


class BuildRoutesUf(UtilityFunction[AbstractState]):
    """
    A utility function meant for testing that only rewards players for building routes.
    """

    def calculate_utility(self, state: AbstractState) -> Utility:
        utility = Utility()
        for player in state.box.players:
            route_points = sum(
                state.box.route_point_values[route.length]
                for route, builder in state.built_routes.items()
                if builder == player
            )
            max_train_cards = max(
                state.player_hands[player].known_train_cards.values(), default=0
            )
            utility[player] = route_points + max_train_cards * 0.75
        return utility


@dataclass
class ExpectedScoreUf(UtilityFunction[AbstractState]):
    """
    Calculates an expect final score for each player.

    This is achieved by calculating the expected number of moves remaining in the game
    and the minimum number of moves needed to complete known destination cards. Based
    on this, the probability of completing all known destinations is computed to get an
    expected value of points from known destination cards. For unknown destination
    cards, an expected value is computed for each based on the number of routes built
    so far, the remaining number of turns, and how many destination cards the player
    has. Additionally, an expected number of points from building routes is calculated.
    These are combined with the points already obtained to computer a total expected
    score, with a discount penalty.

    Args:
        discount: the penalty for points not yet gotten
        distance_normalizer: see _calculate_additional_known_destination_cards_score
    """

    discount: float = 1  # unscientifically chosen
    distance_normalizer: Callable[[float], float] = (
        lambda x: x ** 0.9
    )  # obtained by playing around with desmos
    finish_destinations_probability_estimator: Callable[
        [float, float], float
    ] = lambda remaining_moves, needed_moves: 1 / (
        1
        + math.exp(
            -3.3 * (1 - needed_moves / remaining_moves) - 0.5 * remaining_moves + 2.1
        )
    )  # obtained by playing around with desmos: https://www.desmos.com/calculator/lzes35garz

    def calculate_utility(self, state: AbstractState) -> Utility:
        remaining_moves = self._calculate_remaining_turns(state)
        utilities: Dict[Optional[Player], float] = {
            player: hand.known_points_so_far
            + self.discount
            * self._calculate_additional_score(state, remaining_moves, player, hand)
            for player, hand in state.player_hands.items()
        }
        return Utility(utilities)

    def _calculate_remaining_turns(self, state: AbstractState) -> float:
        return min(
            hand.remaining_trains / self._average_route_length(state.box)
            + (
                hand.remaining_trains
                + self._average_leftover_train_cards(state.box)
                - hand.train_cards_count
            )
            / self._average_cards_per_draw_turn(state.box)
            for player, hand in state.player_hands.items()
        )

    def _calculate_additional_score(
        self,
        state: AbstractState,
        remaining_moves: float,
        player: Player,
        hand: HandState,
    ) -> float:
        return self._calculate_additional_route_building_score(
            state, remaining_moves, player, hand
        ) + self._calculate_additional_destination_cards_score(
            state, remaining_moves, player, hand
        )

    def _calculate_additional_route_building_score(
        self,
        state: AbstractState,
        remaining_moves: float,
        player: Player,
        hand: HandState,
    ) -> float:
        # draw_turns = turns - building_turns
        # building_turns = (cards + draw_turns * avg_cards_per_draw - leftovers) / avg_length
        # => building_turns = (cards + (turns - building_turns) * avg_cards_per_draw - leftovers) / avg_length
        # => building_turns = (cards + turns * avg_cards_per_draw - leftovers) / avg_length - building_turns * avg_cards_per_draw / avg_length
        # => building_turns * (1 + avg_cards_per_draw / avg_length) = (cards + turns * avg_cards_per_draw - leftovers) / avg_length
        # => building_turns = (cards + turns * avg_cards_per_draw - leftovers) / (avg_length * (1 + avg_cards_per_draw / avg_length))
        building_turns = (
            hand.train_cards_count
            + remaining_moves * self._average_cards_per_draw_turn(state.box)
            - self._average_leftover_train_cards(state.box)
        ) / (
            self._average_route_length(state.box)
            * (
                1
                + self._average_cards_per_draw_turn(state.box)
                / self._average_route_length(state.box)
            )
        )
        return self._average_route_value(state.box) * building_turns

    def _calculate_additional_destination_cards_score(
        self,
        state: AbstractState,
        remaining_moves: float,
        player: Player,
        hand: HandState,
    ) -> float:
        unknown_count = hand.destination_cards_count - len(hand.known_destination_cards)

        # Ideally want to find minimum number of trains to complete all destination
        # cards, but it is an NP-Hard problem (travelling salesman reduces to it).
        # Instead, sum the minimum number for each card, and reduce it down if it is
        # a ridiculously high number. (The reason for this is best seen by an example.
        # You may have three very long destination cards to complete, but there is
        # high overlap. The sum is then a ridiculously large number. On the flip side,
        # if there are a few cards that need only a few each, it is less likely for
        # overlap).
        opponent_built_routes = [
            route for route, builder in state.built_routes.items() if builder != player
        ]
        known_cities = (
            (card.cities, card) for card in hand.known_incomplete_destination_cards
        )
        known_best_routes = (
            (
                best_routes_between_cities(
                    s,
                    t,
                    state.built_clusters[player],
                    opponent_built_routes,
                    state.box,
                ),
                card,
            )
            for (s, t), card in known_cities
        )
        known_distance_per_route = (
            min(
                (
                    max(
                        cards_needed_to_build_routes(hand.known_train_cards, route_set)
                        - (hand.train_cards_count - hand.known_train_cards.total),
                        0,
                    )
                    for route_set in route_sets
                ),
                default=state.built_clusters[player].distances[card.cities],
            )
            for route_sets, card in known_best_routes
        )

        known_summed_distances = sum(known_distance_per_route)
        unknown_summed_distances = self._average_route_length(state.box) * unknown_count
        total_distance = self.distance_normalizer(
            known_summed_distances + unknown_summed_distances
        )

        cards_to_finish = max(
            total_distance
            + self._average_leftover_train_cards(state.box)
            - hand.train_cards_count,
            0,
        )
        turns_to_draw = cards_to_finish / self._average_cards_per_draw_turn(state.box)
        turns_to_build = total_distance / self._average_route_length(state.box)
        needed_turns = turns_to_draw + turns_to_build

        probability_to_complete = self.finish_destinations_probability_estimator(
            remaining_moves, needed_turns
        )

        known_values = sum(c.value for c in hand.known_incomplete_destination_cards)
        unknown_values = unknown_count * self._average_route_value(state.box)

        return probability_to_complete * (known_values + unknown_values) - (
            1 - probability_to_complete
        ) * (known_values + unknown_values)

    @staticmethod
    @cache
    def _average_route_length(box: Box) -> float:
        return sum(route.length for route in box.board.routes) / len(box.board.routes)

    @staticmethod
    @cache
    def _average_route_value(box: Box) -> float:
        # weighted by route length
        weighted_values = sum(
            route.length * box.route_point_values[route.length]
            for route in box.board.routes
        )
        total_weight = sum(route.length for route in box.board.routes)
        return weighted_values / total_weight

    @staticmethod
    @cache
    def _average_cards_per_draw_turn(box: Box) -> float:
        return 1.75

    @staticmethod
    @cache
    def _average_leftover_train_cards(box: Box) -> float:
        return len(box.colors) / 3


class ImprovedExpectedScoreUf(ExpectedScoreUf):
    def _calculate_additional_destination_cards_score(
        self,
        state: AbstractState,
        remaining_moves: float,
        player: Player,
        hand: HandState,
    ) -> float:
        unknown_count = hand.destination_cards_count - len(hand.known_destination_cards)

        # Use a heuristic to find the minimum number of trains to complete all
        # destination cards.
        opponent_built_routes = [
            route for route, builder in state.built_routes.items() if builder != player
        ]
        paths = best_routes_between_many_cities(
            (tuple(card.cities) for card in hand.known_incomplete_destination_cards),  # type: ignore
            state.built_clusters[player],
            opponent_built_routes,
            state.box,
        )
        distance_per_path = (
            max(
                cards_needed_to_build_routes(hand.known_train_cards, path)
                - (hand.train_cards_count - hand.known_train_cards.total),
                0,
            )
            for path in paths
        )

        known_summed_distances = min(distance_per_path, default=0)
        unknown_summed_distances = self._average_route_length(state.box) * unknown_count
        total_distance = self.distance_normalizer(
            known_summed_distances + unknown_summed_distances
        )

        cards_to_finish = max(
            total_distance
            + self._average_leftover_train_cards(state.box)
            - hand.train_cards_count,
            0,
        )
        turns_to_draw = cards_to_finish / self._average_cards_per_draw_turn(state.box)
        turns_to_build = total_distance / self._average_route_length(state.box)
        needed_turns = turns_to_draw + turns_to_build

        probability_to_complete = self.finish_destinations_probability_estimator(
            remaining_moves, needed_turns
        )

        known_values = sum(c.value for c in hand.known_incomplete_destination_cards)
        unknown_values = unknown_count * self._average_route_value(state.box)

        return probability_to_complete * (known_values + unknown_values) - (
            1 - probability_to_complete
        ) * (known_values + unknown_values)
