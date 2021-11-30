import pytest

from trains.game.box import Board, Box, City


@pytest.mark.parametrize(
    "city_a, city_b, board, expected",
    [
        ("A", "B", Box.small([]).board, 2),
        ("A", "E", Box.small([]).board, 2),
        ("A", "C", Box.small([]).board, 1),
        ("A", "F", Box.small([]).board, 3),
        ("Las-Vegas", "El-Paso", Box.standard([]).board, 8),
    ],
)
def test_shortest_path(city_a: str, city_b: str, board: Board, expected: int):
    actual = board.shortest_path(City(city_a), City(city_b))
    actual_reversed = board.shortest_path(City(city_b), City(city_a))
    assert actual == expected
    assert actual_reversed == expected
