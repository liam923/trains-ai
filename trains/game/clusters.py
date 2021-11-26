from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, FrozenSet, Generic, TypeVar, Hashable

_Hashable = TypeVar("_Hashable", bound=Hashable)


@dataclass(frozen=True)
class Clusters(Generic[_Hashable]):
    clusters: Tuple[FrozenSet[_Hashable], ...] = field(default_factory=tuple)

    def connect(self, elements: FrozenSet[_Hashable]) -> Clusters:
        clusters = []
        clusters_to_merge = set()
        for i, cluster in enumerate(self.clusters):
            if not elements.isdisjoint(cluster):
                clusters_to_merge.add(i)
            else:
                clusters.append(cluster)

        new_cluster = elements.union(*[self.clusters[i] for i in clusters_to_merge])
        clusters.append(new_cluster)
        return Clusters(tuple(clusters))

    def is_connected(self, elements: FrozenSet[_Hashable]) -> bool:
        return any(elements.issubset(cluster) for cluster in self.clusters)
