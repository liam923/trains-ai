from __future__ import annotations

import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import (
    FrozenSet,
    Optional,
    List,
    Tuple,
    Dict,
    Set,
    DefaultDict,
    Iterable,
    Union,
    TypeVar,
    Type,
)

from trains.mypy_util import cache

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class HashableDict(Dict[_KT, _VT]):
    def __hash__(self) -> int:  # type: ignore
        return hash(tuple(self.items()))


@dataclass(frozen=True)
class Color:
    name: str


@dataclass(frozen=True)
class Player:
    name: str


@dataclass(frozen=True)
class City:
    name: str


@dataclass(frozen=True)
class Route:
    cities: FrozenSet[City]
    color: Optional[Color]  # None represents a gray route (any color can be used)
    length: int
    id: uuid.UUID = field(default_factory=lambda: uuid.uuid4())


@dataclass(frozen=True)
class Board:
    cities: FrozenSet[City]
    routes: FrozenSet[Route]
    double_routes: HashableDict[Route, FrozenSet[Route]]

    @classmethod
    def make(cls, routes: List[Tuple[str, str, Optional[str], int]]) -> Board:
        cities = {
            city_name: City(city_name) for route in routes for city_name in route[0:2]
        }
        colors = {color: Color(color) if color else None for _, _, color, _ in routes}
        board_routes = [
            Route(
                cities=frozenset((cities[city1], cities[city2])),
                color=colors[color],
                length=length,
            )
            for city1, city2, color, length in routes
        ]

        routes_by_cities: Dict[FrozenSet[City], Set[Route]] = {
            route.cities: set() for route in board_routes
        }
        for route in board_routes:
            routes_by_cities[route.cities].add(route)
        double_routes = {
            route: frozenset(routes_by_cities[route.cities]) - {route}
            for route in board_routes
        }

        return Board(
            cities=frozenset(cities.values()),
            routes=frozenset(board_routes),
            double_routes=HashableDict(double_routes),
        )

    @property  # type: ignore
    @cache
    def cities_to_routes(self) -> DefaultDict[FrozenSet[City], List[Route]]:
        d = defaultdict(list)
        for route in self.routes:
            d[route.cities].append(route)
        return d


@dataclass(frozen=True)
class DestinationCard:
    cities: FrozenSet[City]
    value: int
    id: uuid.UUID = field(default_factory=lambda: uuid.uuid4())

    @property
    def cities_list(self) -> List[City]:
        return list(self.cities)

    @classmethod
    def make(cls, city1: str, city2: str, value: int) -> DestinationCard:
        return cls(frozenset([City(city1), City(city2)]), value)


TrainCard = Optional[Color]  # None represents a wildcard

_TrainCard = TypeVar("_TrainCard", bound=TrainCard)  # needed because mypy is stupid


class TrainCards(DefaultDict[TrainCard, int]):
    def __init__(
        self,
        cards: Union[
            Iterable[Tuple[_TrainCard, int]], Dict[_TrainCard, int], Type[int], None
        ] = None,
    ):
        if isinstance(cards, dict):
            super().__init__(int, cards.items())
        elif cards is None:
            super().__init__(int, {})
        elif isinstance(cards, type):
            super().__init__(cards)
        else:
            super().__init__(int, cards)

        self._total = sum(self.values())
        self._normalized = defaultdict(
            int, {color: count / self._total for color, count in self.items()}
        )

    @property
    def total(self) -> int:
        return self._total

    @property
    def normalized(self) -> DefaultDict[TrainCard, float]:
        return self._normalized

    def replacing(self, key: TrainCard, value: int) -> TrainCards:
        train_cards = deepcopy(self)
        train_cards[key] = value
        return train_cards

    def incrementing(self, key: TrainCard, value: int) -> TrainCards:
        return self.replacing(key, self[key] + value)

    def __hash__(self) -> int:  # type: ignore
        return hash(tuple(self.items()))


class MutableTrainCards(TrainCards):
    @property
    def total(self) -> int:
        return sum(self.values())

    @property
    def normalized(self) -> DefaultDict[TrainCard, float]:
        return defaultdict(
            int, {color: count / self._total for color, count in self.items()}
        )


@dataclass(frozen=True)
class Box:
    """
    A Box represents the information about the setup of the game.

    Args:
        board: The game board
        players: The players playing the game, in the order of their turns
        destination_cards: The deck of destination cards in the box
        train_cards: The deck of train cards in the box
        starting_train_count: The number of trains each player starts with
        starting_destination_cards_range: If the tuple is (i, j), players get dealt j
            cards at the start of the game and must keep at least i of them
        dealt_destination_cards_range: If the tuple is (i, j), players get dealt j
            destination cards when they choose to draw them and must keep at least i
            of them
        starting_train_cards_count: The number of train cards players get dealt
        longest_path_bonus: The size of the bonus for getting the longest path
        starting_score: The number of points players begin with
        double_routes_player_minimum: The minimum number of players for double routes
            to be allowed
        trains_to_end: The threshold of remaining trains to trigger the end of the game
        wildcards_to_clear: The number of wildcards needed to clear the face up cards.
        face_up_train_cards: The number of train cards that should be face up
        colors: The colors existent in the game
        route_point_values: A map of route lengths to points given for building a route
            of the given length
    """

    board: Board
    players: Tuple[Player, ...]
    destination_cards: FrozenSet[DestinationCard]
    train_cards: TrainCards
    starting_train_count: int
    starting_destination_cards_range: Tuple[int, int]
    dealt_destination_cards_range: Tuple[int, int]
    starting_train_cards_count: int
    longest_path_bonus: int
    starting_score: int
    double_routes_player_minimum: int
    trains_to_end: int
    wildcards_to_clear: int
    face_up_train_cards: int
    route_point_values: HashableDict[int, int]

    @property  # type: ignore
    @cache
    def next_player_map(self) -> Dict[Player, Player]:
        return dict(zip(self.players, self.players[1:] + self.players[:1]))

    @property  # type: ignore
    @cache
    def colors(self) -> FrozenSet[Color]:
        return frozenset(self.train_cards.keys()) - {None}  # type: ignore

    @classmethod
    def standard(cls, players: List[Player]) -> Box:
        return Box(
            board=Board.make(
                [
                    ("Vancouver", "Calgary", None, 3),
                    ("Vancouver", "Seattle", None, 1),
                    ("Vancouver", "Seattle", None, 1),
                    ("Calgary", "Seattle", None, 4),
                    ("Seattle", "Helena", "yellow", 6),
                    ("Calgary", "Helena", None, 4),
                    ("Seattle", "Portland", None, 1),
                    ("Seattle", "Portland", None, 1),
                    ("Portland", "San-Francisco", "green", 5),
                    ("Portland", "San-Francisco", "pink", 5),
                    ("Portland", "Salt-Lake-City", "blue", 6),
                    ("San-Francisco", "Salt-Lake-City", "orange", 5),
                    ("San-Francisco", "Salt-Lake-City", "white", 5),
                    ("Helena", "Salt-Lake-City", "pink", 3),
                    ("San-Francisco", "Los-Angeles", "yellow", 3),
                    ("San-Francisco", "Los-Angeles", "pink", 3),
                    ("Los-Angeles", "Las-Vegas", None, 2),
                    ("Salt-Lake-City", "Las-Vegas", "orange", 3),
                    ("Calgary", "Winnipeg", "white", 6),
                    ("Helena", "Winnipeg", "blue", 4),
                    ("Los-Angeles", "Phoenix", None, 3),
                    ("Los-Angeles", "El-Paso", "black", 6),
                    ("Phoenix", "El-Paso", None, 3),
                    ("Salt-Lake-City", "Denver", "red", 3),
                    ("Salt-Lake-City", "Denver", "yellow", 3),
                    ("Phoenix", "Denver", "white", 5),
                    ("Helena", "Denver", "green", 4),
                    ("Phoenix", "Santa-Fe", None, 3),
                    ("El-Paso", "Santa-Fe", None, 2),
                    ("Denver", "Santa-Fe", None, 2),
                    ("El-Paso", "Houston", "green", 6),
                    ("El-Paso", "Dallas", "red", 4),
                    ("El-Paso", "Oklahoma-City", "yellow", 5),
                    ("Santa-Fe", "Oklahoma-City", "blue", 3),
                    ("Denver", "Oklahoma-City", "red", 4),
                    ("Denver", "Kansas-City", "black", 4),
                    ("Denver", "Kansas-City", "orange", 4),
                    ("Denver", "Omaha", "pink", 4),
                    ("Helena", "Omaha", "red", 5),
                    ("Helena", "Duluth", "orange", 5),
                    ("Winnipeg", "Duluth", "black", 4),
                    ("Winnipeg", "Sault-St.-Marie", None, 6),
                    ("Duluth", "Sault-St.Marie", None, 3),
                    ("Duluth", "Omaha", None, 2),
                    ("Duluth", "Omaha", None, 2),
                    ("Kansas-City", "Omaha", None, 1),
                    ("Kansas-City", "Omaha", None, 1),
                    ("Kansas-City", "Oklahoma-City", None, 2),
                    ("Kansas-City", "Oklahoma-City", None, 2),
                    ("Dallas", "Oklahoma-City", None, 2),
                    ("Dallas", "Oklahoma-City", None, 2),
                    ("Dallas", "Houston", None, 2),
                    ("Dallas", "Houston", None, 2),
                    ("Oklahoma-City", "Little-Rock", None, 2),
                    ("Dallas", "Little-Rock", None, 2),
                    ("Houston", "New-Orleans", None, 2),
                    ("Little-Rock", "New-Orleans", "green", 3),
                    ("Kansas-City", "Saint-Louis", "blue", 2),
                    ("Kansas-City", "Saint-Louis", "pink", 2),
                    ("Little-Rock", "Saint-Louis", None, 2),
                    ("Omaha", "Chicago", "blue", 4),
                    ("Duluth", "Chicago", "red", 3),
                    ("Duluth", "Toronto", "pink", 6),
                    ("Sault-St.-Marie", "Toronto", None, 2),
                    ("Sault-St.-Marie", "Montreal", "black", 5),
                    ("Toronto", "Montreal", None, 3),
                    ("New-Orleans", "Miami", "red", 6),
                    ("New-Orleans", "Atlanta", "yellow", 4),
                    ("New-Orleans", "Atlanta", "orange", 4),
                    ("Little-Rock", "Nashville", "white", 3),
                    ("Saint-Louis", "Nashville", None, 2),
                    ("Nashville", "Atlanta", None, 1),
                    ("Atlanta", "Miami", "blue", 5),
                    ("Charleston", "Miami", "pink", 4),
                    ("Atlanta", "Charleston", None, 2),
                    ("Atlanta", "Raleigh", None, 2),
                    ("Atlanta", "Raleigh", None, 2),
                    ("Charleston", "Raleigh", None, 2),
                    ("Nashville", "Raleigh", "black", 3),
                    ("Chicago", "Saint-Louis", "green", 2),
                    ("Chicago", "Saint-Louis", "white", 2),
                    ("Chicago", "Pittsburgh", "orange", 3),
                    ("Chicago", "Pittsburgh", "black", 3),
                    ("Saint-Louis", "Pittsburgh", "green", 5),
                    ("Nashville", "Pittsburgh", "yellow", 4),
                    ("Chicago", "Toronto", "white", 4),
                    ("Raleigh", "Pittsburgh", None, 2),
                    ("Raleigh", "Washington", None, 2),
                    ("Raleigh", "Washington", None, 2),
                    ("Pittsburgh", "Washington", None, 2),
                    ("Toronto", "Pittsburgh", None, 2),
                    ("Pittsburgh", "New-York", "white", 2),
                    ("Pittsburgh", "New-York", "green", 2),
                    ("Washington", "New-York", "orange", 2),
                    ("Washington", "New-York", "black", 2),
                    ("Boston", "New-York", "yellow", 2),
                    ("Boston", "New-York", "red", 2),
                    ("New-York", "Montreal", "blue", 3),
                    ("Montreal", "Boston", None, 2),
                    ("Montreal", "Boston", None, 2),
                ]
            ),
            players=tuple(players),
            destination_cards=frozenset(
                [
                    DestinationCard.make("Helena", "Los-Angeles", 8),
                    DestinationCard.make("Portland", "Nashville", 17),
                    DestinationCard.make("Portland", "Phoenix", 11),
                    DestinationCard.make("Montreal", "Atlanta", 9),
                    DestinationCard.make("Montreal", "New-Orleans", 13),
                    DestinationCard.make("Winnipeg", "Little-Rock", 11),
                    DestinationCard.make("Sault-St.-Marie", "Oklahoma-City", 9),
                    DestinationCard.make("Boston", "Miami", 12),
                    DestinationCard.make("San-Francisco", "Atlanta", 17),
                    DestinationCard.make("Toronto", "Miami", 10),
                    DestinationCard.make("Winnipeg", "Houston", 12),
                    DestinationCard.make("Chicago", "New-Orleans", 7),
                    DestinationCard.make("Los-Angeles", "Miami", 20),
                    DestinationCard.make("Sault-St.-Marie", "Nashville", 8),
                    DestinationCard.make("New-York", "Atlanta", 6),
                    DestinationCard.make("Duluth", "Houston", 8),
                    DestinationCard.make("Calgary", "Salt-Lake-City", 7),
                    DestinationCard.make("Denver", "El-Paso", 4),
                    DestinationCard.make("Duluth", "El-Paso", 10),
                    DestinationCard.make("Los-Angeles", "New-York", 21),
                    DestinationCard.make("Calgary", "Phoenix", 13),
                    DestinationCard.make("Chicago", "Santa-Fe", 9),
                    DestinationCard.make("Denver", "Pittsburgh", 11),
                    DestinationCard.make("Dallas", "New-York", 11),
                    DestinationCard.make("Vancouver", "Santa-Fe", 13),
                    DestinationCard.make("Los-Angeles", "Chicago", 16),
                    DestinationCard.make("Kansas-City", "Houston", 5),
                    DestinationCard.make("Seattle", "Los-Angeles", 9),
                    DestinationCard.make("Vancouver", "Montreal", 20),
                    DestinationCard.make("Seattle", "New-York", 22),
                ]
            ),
            train_cards=TrainCards(
                {
                    Color("pink"): 12,
                    Color("white"): 12,
                    Color("blue"): 12,
                    Color("yellow"): 12,
                    Color("orange"): 12,
                    Color("black"): 12,
                    Color("red"): 12,
                    Color("green"): 12,
                    None: 14,
                }
            ),
            starting_train_count=45,
            starting_destination_cards_range=(2, 3),
            dealt_destination_cards_range=(1, 3),
            starting_train_cards_count=4,
            longest_path_bonus=10,
            starting_score=1,
            double_routes_player_minimum=4,
            trains_to_end=2,
            wildcards_to_clear=3,
            face_up_train_cards=5,
            route_point_values=HashableDict(
                {
                    1: 1,
                    2: 2,
                    3: 4,
                    4: 7,
                    5: 10,
                    6: 15,
                }
            ),
        )

    @classmethod
    def small(cls, players: List[Player]) -> Box:
        return Box(
            board=Board.make(
                [
                    ("A", "C", "red", 1),
                    ("A", "E", None, 2),
                    ("B", "C", "blue", 1),
                    ("C", "D", "blue", 1),
                    ("C", "D", None, 1),
                    ("D", "E", None, 1),
                    ("D", "F", "red", 1),
                ]
            ),
            players=tuple(players),
            destination_cards=frozenset(
                [
                    DestinationCard.make("A", "F", 6),
                    DestinationCard.make("A", "E", 3),
                    DestinationCard.make("B", "D", 4),
                    DestinationCard.make("C", "D", 1),
                    DestinationCard.make("C", "F", 2),
                    DestinationCard.make("E", "F", 2),
                ]
            ),
            train_cards=TrainCards({Color("red"): 10, Color("blue"): 10, None: 8}),
            starting_train_count=5,
            starting_destination_cards_range=(1, 2),
            dealt_destination_cards_range=(1, 2),
            starting_train_cards_count=1,
            longest_path_bonus=5,
            starting_score=1,
            double_routes_player_minimum=3,
            trains_to_end=1,
            wildcards_to_clear=2,
            face_up_train_cards=2,
            route_point_values=HashableDict({1: 1, 2: 3}),
        )
