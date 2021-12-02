from trains.game.box import Box, City
from trains.game.clusters import Clusters


def test_connect():
    board = Box.small([]).board
    clusters = Clusters(frozenset(), board.shortest_paths)
    A = City("A")
    B = City("B")
    C = City("C")
    D = City("D")
    E = City("E")
    F = City("F")

    assert clusters.distance(B, E) == 3
    assert clusters.distance(A, E) == 2
    assert clusters.distance(C, F) == 2

    clusters = clusters.connect(C, D)
    assert clusters.distance(B, E) == 2
    assert clusters.distance(A, E) == 2
    assert clusters.distance(C, F) == 1

    clusters = clusters.connect(A, E)
    assert clusters.distance(B, E) == 2
    assert clusters.distance(A, E) == 0
    assert clusters.distance(C, F) == 1

    clusters = clusters.connect(F, D)
    assert clusters.distance(B, E) == 2
    assert clusters.distance(A, E) == 0
    assert clusters.distance(C, F) == 0

    clusters = clusters.connect(E, D)
    assert clusters.distance(B, E) == 1
    assert clusters.distance(A, E) == 0
    assert clusters.distance(C, F) == 0

    clusters = clusters.connect(A, C)
    assert clusters.distance(B, E) == 1
    assert clusters.distance(A, E) == 0
    assert clusters.distance(C, F) == 0

    clusters = clusters.connect(B, C)
    assert clusters.distance(B, E) == 0
    assert clusters.distance(A, E) == 0
    assert clusters.distance(C, F) == 0
