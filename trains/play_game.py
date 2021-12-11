"""
A script to play against UF-MCTS on a small board.
"""

from trains.ai.mcts import UfMctsActor
from trains.ai.utility_function import (
    ImprovedExpectedScoreUf,
    RelativeUf,
)
from trains.game.box import Box, Player
from trains.game.game_actor import play_game, SimulatedGameActor
from trains.game.known_state import KnownState
from trains.game.player_actor import UserActor

if __name__ == "__main__":
    players = [Player("User"), Player("AI")]
    box = Box.small(players)
    play_game(
        {
            Player("User"): UserActor.make(box, Player("User"), state_type=KnownState),
            Player("AI"): UfMctsActor.make(
                box,
                Player("AI"),
                iterations=10000,
                utility_function=RelativeUf(ImprovedExpectedScoreUf()),
            ),
        },
        SimulatedGameActor.make(box),
    )
