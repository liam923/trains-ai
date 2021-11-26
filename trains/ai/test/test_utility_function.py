from trains.ai.utility_function import Utility

utility_a = Utility({"a": 1, "b": 2})
utility_b = Utility({"a": 3, "c": -2})


def test_utility_add():
    result = utility_a + utility_b
    assert result["a"] == 4
    assert result["b"] == 2
    assert result["c"] == -2


def test_utility_mul():
    result = utility_a * 10
    assert result["a"] == 10
    assert result["b"] == 20


def test_utility_rmul():
    result = -2 * utility_b
    assert result["a"] == -6
    assert result["c"] == 4


def test_utility_div():
    result = utility_a / 2
    assert result["a"] == 0.5
    assert result["b"] == 1
