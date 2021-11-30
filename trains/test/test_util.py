from typing import Optional

import pytest

from trains.game.box import TrainCards, Color
from trains.util import probability_of_having_cards, subtract_train_cards

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
