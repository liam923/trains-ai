from trains.ai.mcts_expectiminimax import MctsExpectiminimaxActor
from trains.ai.mcts import BasicMctsActor, UfMctsActor
from trains.ai.random import RandomActor
from trains.ai.route_building_probability_calculator import DummyRbpc
from trains.ai.utility_function import ExpectedScoreUf
from trains.game.box import Box, Player
from trains.game.game_actor import play_game, SimulatedGameActor
from trains.game.player_actor import UserActor

if __name__ == "__main__":
    players = [Player("AI"), Player("User")]
    box = Box.small(players)
    play_game(
        # {player: UserActor.make(box, player) for player in players},
        {
            # Player("User"): UserActor.make(box, Player("User")),
            # Player("AI"): MctsExpectiminimaxActor.make(
            #     box,
            #     Player("AI"),
            #     utility_function=ExpectedScoreUf(discount=1),
            #     route_building_probability_calculator=DummyRbpc,
            #     depth=2,
            #     breadth=lambda depth: None if depth >= 1 else 10,
            # ),
            Player("AI"): BasicMctsActor.make(box, Player("AI"), iterations=100),
            Player("User"): RandomActor.make(box, Player("User")),
        },
        SimulatedGameActor.make(box),
    )
