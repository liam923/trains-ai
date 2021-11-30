from __future__ import annotations

import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    Tuple,
    TypeVar,
    Generic,
    Optional,
    Iterable,
    Generator,
    DefaultDict,
    Dict,
)

from trains.game.box import TrainCards, Route, TrainCard
from trains.mypy_util import cache


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

    needed_colors = tuple(cards.keys())
    needed_cards = tuple(cards[color] for color in needed_colors)
    favorables = tuple(pile_distribution[color] for color in needed_colors)
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


@dataclass
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
        if l is not None:
            yield l.head
            yield from Cons.iterate(l.rest)


def randomly_sample_distribution(
    iterable: Iterable[Tuple[_T, float]], sample_size: int
) -> Generator[_T, None, None]:
    """
    Randomly sample a number of elements from a distribution
    """
    collected = list(iterable)
    for _ in range(sample_size):
        rand = random.random()
        i = 0
        cum = collected[i][1]
        while rand < cum and i < len(collected) - 1:
            i += 1
            cum += collected[i][1]
        yield collected[i][0]
