from typing import Optional, FrozenSet, Set, Tuple, Collection

import pytest

from trains.game.box import TrainCards, Color, City, Route, Box, Board
from trains.game.clusters import Clusters
from trains.util import (
    probability_of_having_cards,
    subtract_train_cards,
    best_routes,
    cards_needed_to_build_routes,
)

blue = Color("blue")
red = Color("red")
wild = None


@pytest.mark.parametrize(
    "cards, hand_size, pile_distribution, expected",
    [
        (TrainCards(), 0, TrainCards(), 1),
        (TrainCards({blue: 1}), 1, TrainCards({blue: 1}), 1),
        (TrainCards({blue: 1}), 1, TrainCards({blue: 1, red: 1}), 0.5),
        (TrainCards({blue: 1}), 2, TrainCards({blue: 1, red: 1}), 1),
        (
            TrainCards({blue: 1}),
            3,
            TrainCards({blue: 1, red: 9}),
            1 - (9 / 10) * (8 / 9) * (7 / 8),
        ),
        (
            TrainCards({blue: 1}),
            2,
            TrainCards({blue: 10, red: 10}),
            1 - ((10 * 9) / (20 * 19)),
        ),
        (
            TrainCards({blue: 1, red: 1}),
            2,
            TrainCards({blue: 10, red: 10}),
            1 - 2 * ((10 * 9) / (20 * 19)),
        ),
        (
            TrainCards({blue: 1, red: 1}),
            3,
            TrainCards({blue: 10, red: 10}),
            1 - 2 * ((10 * 9 * 8) / (20 * 19 * 18)),
        ),
        (
            TrainCards({blue: 3}),
            3,
            TrainCards({blue: 10, red: 10}),
            (10 * 9 * 8) / (20 * 19 * 18),
        ),
        (TrainCards({blue: 2}), 2, TrainCards({blue: 1, red: 1}), 0),
        (TrainCards({None: 1}), 2, TrainCards({blue: 1, red: 1}), 0),
        (
            TrainCards(
                {blue: 1, red: 3, None: 1, Color("black"): 2, Color("green"): 1}
            ),
            10,
            TrainCards(
                {
                    color: 10
                    for color in [
                        blue,
                        red,
                        None,
                        Color("black"),
                        Color("green"),
                        Color("pink"),
                    ]
                }
            ),
            None,
        ),
    ],
)
def test_probability_of_having_cards(
    cards: TrainCards,
    hand_size: int,
    pile_distribution: TrainCards,
    expected: Optional[float],
):
    actual = probability_of_having_cards(cards, hand_size, pile_distribution)
    if expected is not None:
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "original, minus, expected_result, expected_leftovers",
    [
        (TrainCards(), TrainCards(), TrainCards(), TrainCards()),
        (
            TrainCards({blue: 2, red: 1}),
            TrainCards({blue: 1}),
            TrainCards({blue: 1, red: 1}),
            TrainCards(),
        ),
        (
            TrainCards({blue: 2, red: 1}),
            TrainCards({blue: 2}),
            TrainCards({red: 1}),
            TrainCards(),
        ),
        (
            TrainCards({blue: 2, red: 1}),
            TrainCards({blue: 3}),
            TrainCards({red: 1}),
            TrainCards({blue: 1}),
        ),
        (
            TrainCards({blue: 2, red: 1}),
            TrainCards({blue: 1, red: 1}),
            TrainCards({blue: 1}),
            TrainCards(),
        ),
        (
            TrainCards({blue: 2, red: 1}),
            TrainCards({blue: 2, red: 2}),
            TrainCards(),
            TrainCards({red: 1}),
        ),
        (
            TrainCards({blue: 2, red: 1, wild: 4}),
            TrainCards({blue: 3, red: 2}),
            TrainCards({wild: 4}),
            TrainCards({red: 1, blue: 1}),
        ),
    ],
)
def test_subtract_train_cards(
    original: TrainCards,
    minus: TrainCards,
    expected_result: TrainCards,
    expected_leftovers: TrainCards,
):
    actual_result, actual_leftovers = subtract_train_cards(original, minus)
    assert actual_result == expected_result
    assert actual_leftovers == expected_leftovers


small_box = Box.small([])


def _small_board_route(
    from_city: City, to_city: City, double_color: Optional[Color] = None
) -> Route:
    routes = small_box.board.cities_to_routes[frozenset([from_city, to_city])]
    if len(routes) == 1:
        return routes[0]
    else:
        return [route for route in routes if route.color == double_color][0]


A = City("A")
B = City("B")
C = City("C")
D = City("D")
E = City("E")
F = City("F")
A_E = _small_board_route(A, E)
A_C = _small_board_route(A, C)
B_C = _small_board_route(B, C)
C_D_blue = _small_board_route(C, D, Color("blue"))
C_D_grey = _small_board_route(C, D, None)
D_E = _small_board_route(D, E)
D_F = _small_board_route(D, F)


@pytest.mark.parametrize(
    "from_city, to_city, player_built_routes, opponent_built_routes, box, expected",
    [
        (
            City("A"),
            City("E"),
            frozenset(),
            frozenset(),
            small_box,
            frozenset({frozenset({A_E})}),
        ),
        (
            City("A"),
            City("F"),
            frozenset(),
            frozenset(),
            small_box,
            ((A_C, C_D_blue, D_F), (A_C, C_D_grey, D_F)),
        ),
        (
            City("A"),
            City("A"),
            frozenset(),
            frozenset(),
            small_box,
            frozenset({frozenset()}),
        ),
        (
            City("A"),
            City("D"),
            frozenset({D_E}),
            frozenset(),
            small_box,
            frozenset(
                {
                    frozenset({A_E}),
                    frozenset({A_C, C_D_blue}),
                    frozenset({A_C, C_D_grey}),
                }
            ),
        ),
        (City("A"), City("D"), (D_E,), (C_D_grey,), small_box, ((A_E,),)),
        (City("A"), City("D"), (D_E,), (C_D_grey, C_D_blue), small_box, ((A_E,),)),
        (City("A"), City("D"), tuple(), (C_D_grey,), small_box, ((A_E, D_E),)),
        (
            City("D"),
            City("E"),
            tuple(),
            (D_E,),
            small_box,
            ((C_D_grey, A_C, A_E), (C_D_blue, A_C, A_E)),
        ),
        (
            City("D"),
            City("E"),
            frozenset({A_E, A_C}),
            frozenset({D_E}),
            small_box,
            frozenset({frozenset({C_D_blue}), frozenset({C_D_grey})}),
        ),
    ],
)
def test_best_routes(
    from_city: City,
    to_city: City,
    player_built_routes: Collection[Route],
    opponent_built_routes: Collection[Route],
    box: Box,
    expected: Collection[Collection[Route]],
):
    clusters = Clusters(frozenset(), box.board.shortest_paths)
    for route in player_built_routes:
        clusters = clusters.connect(*route.cities)

    actual = frozenset(
        best_routes(from_city, to_city, clusters, frozenset(opponent_built_routes), box)
    )
    expected_as_sets = frozenset(frozenset(ex) for ex in expected)
    assert actual == expected_as_sets


@pytest.mark.parametrize(
    "train_cards, routes, expected",
    [
        (TrainCards({red: 0}), (A_E, A_C), 3),
        (TrainCards({red: 1}), (A_E, A_C), 2),
        (TrainCards({red: 2}), (A_E, A_C), 1),
        (TrainCards({red: 3}), (A_E, A_C), 0),
        (TrainCards({red: 0, blue: 2}), (A_E, A_C), 1),
        (TrainCards({red: 1, blue: 1}), (A_E, A_C), 1),
        (TrainCards({red: 2, blue: 1}), (A_E, A_C), 1),
        (TrainCards({red: 3, blue: 1}), (A_E, A_C), 0),
        (TrainCards({red: 4, blue: 2}), (A_E, C_D_grey, B_C), 0),
        (TrainCards({red: 0, blue: 0}), (A_E, C_D_grey, B_C), 4),
        (TrainCards({red: 0, blue: 2}), (A_E, C_D_grey, B_C), 2),
        (TrainCards({red: 2, blue: 2}), (A_E, C_D_grey, B_C), 0),
    ],
)
def test_cards_needed_to_build_routes(
    train_cards: TrainCards, routes: Tuple[Route, ...], expected: int
):
    actual = cards_needed_to_build_routes(train_cards, routes)
    assert actual == expected
