from __future__ import annotations

import heapq
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Tuple,
    TypeVar,
    Generic,
    Optional,
    Iterable,
    Generator,
    DefaultDict,
    Dict,
    FrozenSet,
    Collection,
    List,
    Set,
)

from trains.game.box import TrainCards, Route, TrainCard, City, Box
from trains.game.clusters import Clusters
from trains.mypy_util import cache, add_slots


def sufficient_cards_to_build(
    route: Route, cards: TrainCards, unknown_cards: int = 0
) -> bool:
    if route.color is None:
        max_color_card_set = max(
            (count for color, count in cards.items() if color is not None), default=0
        )
        return cards[None] + max_color_card_set + unknown_cards >= route.length
    else:
        return cards[None] + cards[route.color] + unknown_cards >= route.length


def merge_train_cards(*cards: TrainCards) -> TrainCards:
    result_cards: DefaultDict[TrainCard, int] = defaultdict(int)
    for card_set in cards:
        for color, count in card_set.items():
            result_cards[color] += count
    return TrainCards(result_cards)


def subtract_train_cards(
    original: TrainCards, minus: TrainCards
) -> Tuple[TrainCards, TrainCards]:
    """
    Subtract the second set of train cards from the first set. Return (the resultant
    cards, the leftover cards)
    """
    new_cards: Dict[TrainCard, int] = {}
    extras: Dict[TrainCard, int] = {}
    for color in original.keys() | minus.keys():
        diff = original[color] - minus[color]
        if diff > 0:
            new_cards[color] = diff
        elif diff < 0:
            extras[color] = -diff
    return TrainCards(new_cards), TrainCards(extras)


def probability_of_having_cards(
    cards: TrainCards, hand_size: int, pile_distribution: TrainCards
) -> float:
    """
    Returns the probability that the given cards are located inside a hand of the
    specified size, with the given distribution.
    """

    needed_cards = tuple(cards.values())
    favorables = tuple(pile_distribution[color] for color in cards.keys())
    total_favorables = sum(favorables)
    total_unfavorables = pile_distribution.total - total_favorables

    return _probability_of_having_cards_helper(
        hand_size, needed_cards, favorables, total_favorables, total_unfavorables
    )


@cache
def _probability_of_having_cards_helper(
    remaining: int,
    needed_cards: Tuple[int, ...],
    favorables: Tuple[int, ...],
    total_favorables: int,
    total_unfavorables: int,
) -> float:
    """
    Helper for probability_of_having_cards that actually does the calculation
    """
    if len(needed_cards) == 0:
        return 1
    elif remaining == 0:
        return 0
    else:
        total_cards = total_favorables + total_unfavorables
        prob = 0.0
        for i, needed in enumerate(needed_cards):
            if needed > favorables[i]:
                return 0
            draw_prob = favorables[i] / total_cards

            if needed == 1:
                prob += draw_prob * _probability_of_having_cards_helper(
                    remaining=remaining - 1,
                    needed_cards=needed_cards[:i] + needed_cards[i + 1 :],
                    favorables=favorables[:i] + favorables[i + 1 :],
                    total_favorables=total_favorables - favorables[i],
                    total_unfavorables=total_unfavorables + favorables[i] - 1,
                )
            else:
                prob += draw_prob * _probability_of_having_cards_helper(
                    remaining=remaining - 1,
                    needed_cards=needed_cards[:i]
                    + (needed_cards[i] - 1,)
                    + needed_cards[i + 1 :],
                    favorables=favorables[:i]
                    + (favorables[i] - 1,)
                    + favorables[i + 1 :],
                    total_favorables=total_favorables - 1,
                    total_unfavorables=total_unfavorables,
                )

        if total_unfavorables > 0:
            draw_prob = total_unfavorables / total_cards
            prob += draw_prob * _probability_of_having_cards_helper(
                remaining - 1,
                needed_cards,
                favorables,
                total_favorables,
                total_unfavorables - 1,
            )

        return prob


_T = TypeVar("_T")


@add_slots
@dataclass(frozen=True)
class Cons(Generic[_T]):
    head: _T
    rest: Optional[Cons[_T]]

    @classmethod
    def make(cls, iterable: Iterable[_T]) -> Optional[Cons[_T]]:
        cons = None
        for i in iterable:
            cons = cls(i, cons)
        return cons

    @staticmethod
    def iterate(l: Optional[Cons[_T]]) -> Generator[_T, None, None]:
        while l is not None:
            yield l.head
            l = l.rest


def randomly_sample_distribution(
    iterable: Iterable[Tuple[_T, float]], sample_size: int = 1
) -> Generator[_T, None, None]:
    """
    Randomly sample a number of elements from a distribution
    """
    collected = list(iterable)
    total = sum(prob for _, prob in collected)
    for _ in range(sample_size):
        rand = random.random() * total
        i = 0
        cum = collected[i][1]
        while rand > cum and i < len(collected) - 1:
            i += 1
            cum += collected[i][1]
        yield collected[i][0]


@add_slots
@dataclass(order=True, frozen=True)
class PrioritizedItem(Generic[_T]):
    priority: int
    item: _T = field(compare=False)


def best_routes_between_many_cities(
    city_pairs: Iterable[Tuple[City, City]],
    player_built_routes: Clusters,
    opponent_built_routes: Collection[Route],
    box: Box,
) -> Generator[FrozenSet[Route], None, None]:
    """
    Find good paths that connect all of the city pairs. player_built_routes represents
    the routes already built by the user, and opponent_built_routes represents the
    routes already built by other users.

    The paths are "good" and not optimal because this is an NP-Hard problem and a
    heuristic is used instead.

    Note that the returned paths do not include routes already built.
    """
    # The strategy is to choose the pair of cities that are farthest apart and find
    # the best path between them. Then, recursively find a good path between the
    # remaining pairs of cities, where it is assumed that path has been built.

    candidate_paths: List[Tuple[FrozenSet[Route], Clusters]] = [
        (frozenset(), player_built_routes)
    ]
    remaining_city_pairs: Set[Tuple[City, City]] = set(city_pairs)
    while len(remaining_city_pairs) != 0:
        farthest_pair = max(
            remaining_city_pairs, key=lambda pair: player_built_routes.distance(*pair)
        )
        remaining_city_pairs.remove(farthest_pair)
        city_a, city_b = farthest_pair

        best_distance: Optional[int] = None
        new_candidate_paths: List[Tuple[FrozenSet[Route], Clusters]] = []
        for path_so_far, built_routes in candidate_paths:
            for rest_path in best_routes_between_cities(
                city_a, city_b, built_routes, opponent_built_routes, box
            ):
                distance = sum(route.length for route in rest_path)
                if best_distance is None or distance < best_distance:
                    new_candidate_paths = []
                    best_distance = distance
                if distance <= best_distance:
                    additional_built_routes = built_routes
                    for route in rest_path:
                        additional_built_routes = additional_built_routes.connect(
                            *route.cities
                        )
                    new_candidate_paths.append(
                        (path_so_far | rest_path, additional_built_routes)
                    )
        candidate_paths = new_candidate_paths

    for path, _ in candidate_paths:
        yield path


def best_routes_between_cities(
    from_city: City,
    to_city: City,
    player_built_routes: Clusters,
    opponent_built_routes: Collection[Route],
    box: Box,
) -> Generator[FrozenSet[Route], None, None]:
    """
    Find all optimal paths from one city to another. player_built_routes represents the
    routes already built by the user, and opponent_built_routes represents the routes
    already built by other users.

    Note that the returned optimal paths do not include routes already built.
    """
    # Use a slightly modified A* search to find all the shortest paths between the
    # two cities. Instead of terminating once the goal is reached, it notes the optimal
    # length. It then continues until all priorities in the queue are greater than the
    # optimal length.

    # The heuristic used is simply the distance between cities according to
    # built_clusters, which is admissible because that is the optimal distance if there
    # are no routes built by opponents blocking the path

    blocked_routes = set(opponent_built_routes)
    if len(box.players) < box.double_routes_player_minimum:
        for route in opponent_built_routes:
            blocked_routes.update(box.board.double_routes[route])

    start = player_built_routes.get_cluster_for_city(from_city)
    goal = player_built_routes.get_cluster_for_city(to_city)

    optimal_length: Optional[int] = None
    frontier: List[
        PrioritizedItem[Tuple[FrozenSet[City], Optional[Cons[Route]], int]]
    ] = []
    frontier_set: Set[Tuple[FrozenSet[City], Optional[Cons[Route]], int]] = set()
    queue_item: PrioritizedItem[
        Tuple[FrozenSet[City], Optional[Cons[Route]], int]
    ] = PrioritizedItem(0, (start, None, 0))
    heapq.heappush(frontier, queue_item)
    while len(frontier) > 0 and (
        optimal_length is None or frontier[0].priority <= optimal_length
    ):
        current_cluster, current_path, current_cost = heapq.heappop(frontier).item

        if current_cluster == goal:
            optimal_length = current_cost
            yield frozenset(Cons.iterate(current_path))

        for current_city in current_cluster:
            for successor, route in box.board.routes_from_city(current_city):
                if (
                    successor not in current_cluster
                    and route not in blocked_routes
                    and route not in Cons.iterate(current_path)
                ):
                    successor_cluster = player_built_routes.get_cluster_for_city(
                        successor
                    )
                    cost = player_built_routes.distance(current_city, successor)
                    successor_cost = current_cost + cost
                    successor_path = Cons(route, current_path)
                    successor_priority = successor_cost + player_built_routes.distance(
                        successor, to_city
                    )

                    queue_item = PrioritizedItem(
                        successor_priority,
                        (successor_cluster, successor_path, successor_cost),
                    )
                    heapq.heappush(frontier, queue_item)


def cards_needed_to_build_routes(
    cards_in_hand: TrainCards, routes: Collection[Route]
) -> int:
    cards_remaining = defaultdict(int, cards_in_hand)
    needed_cards = 0

    colored_routes = {route for route in routes if route.color is not None}
    gray_routes = {route for route in routes if route.color is None}

    for route in colored_routes:
        if route.length >= cards_remaining[route.color]:
            needed_cards += route.length - cards_remaining[route.color]
            cards_remaining[route.color] = 0
        else:
            cards_remaining[route.color] -= route.length

    colored_cards_remaining = [
        (-count, color.name, color)
        for color, count in cards_remaining.items()
        if color is not None
    ]
    heapq.heapify(colored_cards_remaining)
    for route in sorted(gray_routes, reverse=True, key=lambda r: r.length):
        if len(colored_cards_remaining) == 0:
            needed_cards += route.length
        else:
            neg_count, _, color = heapq.heappop(colored_cards_remaining)
            count = -neg_count
            if route.length >= count:
                needed_cards += route.length - count
            else:
                heapq.heappush(
                    colored_cards_remaining, ((count - route.length), color.name, color)
                )

    needed_cards = max(needed_cards - cards_in_hand[None], 0)
    return needed_cards
