from dataclasses import dataclass
from typing import FrozenSet, Dict, Union

from trains.game.box import DestinationCard, Route, TrainCard, TrainCards, Player


@dataclass(frozen=True)
class TrainCardDealAction:
    """
    Represents the deck randomly producing train cards
    """

    cards: TrainCards


@dataclass(frozen=True)
class DestinationCardDealAction:
    """
    Represents the deck randomly producing destination cards
    """

    cards: FrozenSet[DestinationCard]


@dataclass(frozen=True)
class TrainCardPickAction:
    """
    Represents a player choosing to draw a train card

    Args:
        draw_known: True if the player is selecting a face-up train card
        selected_card_if_known: The draw_known is True, the color of the card to draw
    """

    draw_known: bool
    selected_card_if_known: TrainCard = None


@dataclass(frozen=True)
class DestinationCardPickAction:
    """
    Represents a player choosing to draw destination cards
    """


@dataclass(frozen=True)
class DestinationCardSelectionAction:
    """
    Represents a player selecting some subset of drawn destination cards
    """

    selected_cards: FrozenSet[DestinationCard]


@dataclass(frozen=True)
class BuildAction:
    """
    Represents a player building a route

    Args:
        route: The route being build
        train_card_distribution: The train cards used to build the route
    """

    route: Route
    train_cards: TrainCards


@dataclass(frozen=True)
class RevealDestinationCardSelectionsAction:
    """
    Represents the game revealing to players what how many destination cards each player
    selected at the start of the game
    """

    kept_destination_cards: Dict[Player, int]


@dataclass(frozen=True)
class InitialDealAction:
    """
    Represents the game dealing hands to all players at the start of the game
    """

    train_cards: Dict[Player, TrainCards]
    destination_cards: Dict[Player, FrozenSet[DestinationCard]]
    face_up_train_cards: TrainCards


@dataclass(frozen=True)
class PassAction:
    """
    Represents a player passing their turn
    """


Action = Union[
    TrainCardDealAction,
    DestinationCardDealAction,
    TrainCardPickAction,
    DestinationCardPickAction,
    DestinationCardSelectionAction,
    BuildAction,
    RevealDestinationCardSelectionsAction,
    InitialDealAction,
    PassAction,
]
