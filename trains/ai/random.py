import random
from dataclasses import dataclass

from trains.ai.actor import AiActor
from trains.game.action import Action


@dataclass
class RandomActor(AiActor):
    def get_action(self) -> Action:
        all_actions = list(self.state.get_legal_actions())
        return random.choice(all_actions).action
