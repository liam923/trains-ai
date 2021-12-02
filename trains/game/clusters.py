from __future__ import annotations

from typing import FrozenSet, Generic, TypeVar, Hashable, Any

from frozendict import frozendict

from trains.game.box import City, Board


class Clusters:
    def __init__(
        self,
        clusters: FrozenSet[FrozenSet[City]],
        distances: frozendict[FrozenSet[City], int],
    ):
        self.clusters = clusters
        self.distances = distances

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Clusters) and self.clusters == other.clusters

    def __hash__(self) -> int:
        return hash(self.clusters)

    def connect(self, a: City, b: City) -> Clusters:
        cities = frozenset([a, b])
        clusters = []
        clusters_to_merge = []
        for cluster in self.clusters:
            if not cities.isdisjoint(cluster):
                clusters_to_merge.append(cluster)
            else:
                clusters.append(cluster)

        new_cluster = cities.union(*clusters_to_merge)
        clusters.append(new_cluster)

        # see https://cs.stackexchange.com/a/76850 for reasoning behind how new
        # distances are calculated
        new_distances = frozendict(
            (
                frozenset([s, t]),
                min(
                    old_distance,
                    self.distance(s, a) + self.distance(b, t),
                    self.distance(s, b) + self.distance(a, t),
                ),
            )
            for edge, (s, t), old_distance in (
                (edge, edge, old_distance)
                for edge, old_distance in self.distances.items()
            )
        )

        return Clusters(frozenset(clusters), frozendict(new_distances))

    def is_connected(self, cities: FrozenSet[City]) -> bool:
        return any(cities.issubset(cluster) for cluster in self.clusters)

    def distance(self, from_city: City, to_city: City) -> int:
        if from_city == to_city:
            return 0
        else:
            return self.distances[frozenset([from_city, to_city])]

    def get_cluster_for_city(self, city: City) -> FrozenSet[City]:
        for cluster in self.clusters:
            if city in cluster:
                return cluster
        return frozenset([city])
