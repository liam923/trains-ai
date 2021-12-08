from typing import Generic

from trains.game.player_actor import PlayerActor
from trains.game.state import State


class AiActor(Generic[State], PlayerActor[State]):
    pass
