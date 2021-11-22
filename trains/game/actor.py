from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from trains.game.action import Action
from trains.game.box import Box
from trains.game.turn import TurnState


@dataclass  # type: ignore
class Actor(ABC):
    box: Box
    turn_state: TurnState

    @abstractmethod
    def validate_action(self, action: Action) -> Optional[str]:
        """
        Validate that the given action is ok with this Actor. It the Actor believes it
        is illegal, an error message is returned. If it is ok, None.
        """

    @abstractmethod
    def observe_action(self, action: Action) -> None:
        """
        Observe the given action being performed in the game.
        """

    @abstractmethod
    def get_action(self) -> Action:
        """
        Get the action to be performed by the player.
        """
