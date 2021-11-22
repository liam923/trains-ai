import string
from copy import deepcopy
from copy import deepcopy
from random import Random
from typing import Tuple

from trains.game.box import TrainCards, Route


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
    result_cards = TrainCards()
    for card_set in cards:
        for color, count in card_set.items():
            result_cards[color] += count
    return result_cards


def subtract_train_cards(
    original: TrainCards, minus: TrainCards
) -> Tuple[TrainCards, TrainCards]:
    """
    Subtract the second set of train cards from the first set. Return (the resultant
    cards, the leftover cards)
    """
    new_cards = deepcopy(original)
    extras = TrainCards()
    for color, count in minus.items():
        diff = original[color] - minus[color]
        if diff >= 0:
            new_cards[color] -= diff
        else:
            extras[color] -= diff
    return new_cards, extras


def probability_of_having_cards(
    cards: TrainCards, pile_size: int, pile_distribution: TrainCards
) -> float:
    """
    Returns the probability that the given cards are located inside a pile of the
    specified size, with the given distribution.
    """
    return 1  # TODO
