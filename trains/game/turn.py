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
        last_turn_started: True if the last turn of the game has been started or
            completed.
    """

    last_turn_started: bool


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

    @staticmethod
    def make_or_end(
        last_turn_started: bool, player: Player
    ) -> Union[PlayerStartTurn, RevealFinalDestinationCardsTurn]:
        if last_turn_started:
            return RevealFinalDestinationCardsTurn()
        else:
            return PlayerStartTurn(last_turn_started=last_turn_started, player=player)


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
class RevealFinalDestinationCardsTurn(GameTurn):
    """
    Represents the game revealing to players how which destination cards they each
    have at the end of the game.
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
    RevealFinalDestinationCardsTurn,
]
