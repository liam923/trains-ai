from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Union, Optional

from trains.game.box import Player


@dataclass(frozen=True)
class PlayerTurn(ABC):
    """
    Represents a turn for a player to make

    Args:
        player: The player whose turn it is.
    """

    player: Player


@dataclass(frozen=True)
class GameTurn(ABC):
    """
    Represents a turn for the game to make
    """

    @property
    def player(self) -> Optional[None]:
        return None


@dataclass(frozen=True)
class GameOverTurn:
    """
    Represents the game being over
    """

    @property
    def player(self) -> Optional[None]:
        return None


@dataclass(frozen=True)
class RotatingTurn:
    """
    An abstract turn that represents a turn during the main circling of turns during
    the game.

    Args:
        last_turn_forced: If the last turn has been forced by a player, the player who
            forced the last turn. Otherwise, None.
    """

    last_turn_forced: Optional[Player]


@dataclass(frozen=True)
class PlayerInitialDestinationCardChoiceTurn(PlayerTurn):
    """
    Represents the player choosing destination cards at the start of the game
    """


@dataclass(frozen=True)
class PlayerStartTurn(PlayerTurn, RotatingTurn):
    """
    Represents the start of a player's turn
    """


@dataclass(frozen=True)
class PlayerTrainCardDrawMidTurn(PlayerTurn, RotatingTurn):
    """
    Represents the player drawing a second train card
    """


@dataclass(frozen=True)
class PlayerDestinationCardDrawMidTurn(PlayerTurn, RotatingTurn):
    """
    Represents the player choosing from three destination cards
    """


@dataclass(frozen=True)
class InitialTurn(GameTurn):
    """
    Represents the very start of the game, waiting for the game to deal initial hands to
    players
    """


@dataclass(frozen=True)
class DestinationCardDealTurn(GameTurn, RotatingTurn):
    """
    Represents the game randomly dealing destination cards.
    """

    to_player: Player


@dataclass(frozen=True)
class TrainCardDealTurn(GameTurn):
    """
    Represents the game randomly dealing train cards.

    Args:
        count: The number of cards to deal
        to_player: Who to deal the cards to. If None, the face up cards
        next_turn_state: The state the game should go into after the dealing is done
    """

    count: int
    to_player: Optional[Player]
    next_turn_state: TurnState


@dataclass(frozen=True)
class RevealInitialDestinationCardChoicesTurn(GameTurn):
    """
    Represents the game revealing to players how many destination cards they each chose
    to keep.
    """


TurnState = Union[
    GameOverTurn,
    PlayerInitialDestinationCardChoiceTurn,
    PlayerStartTurn,
    PlayerTrainCardDrawMidTurn,
    PlayerDestinationCardDrawMidTurn,
    InitialTurn,
    DestinationCardDealTurn,
    TrainCardDealTurn,
    RevealInitialDestinationCardChoicesTurn,
]
