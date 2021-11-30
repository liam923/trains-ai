from trains.ai.mcts_expectiminimax import MctsExpectiminimaxActor
from trains.ai.route_building_probability_calculator import DummyRbpc
from trains.ai.utility_function import BuildRoutesUf
from trains.game.box import Box, Player
from trains.game.game_actor import play_game, SimulatedGameActor
from trains.game.player_actor import UserActor

if __name__ == "__main__":
    players = [Player("one"), Player("two")]
    box = Box.small(players)
    play_game(
        # {player: UserActor.make(box, player) for player in players},
        {
            Player("one"): UserActor.make(box, Player("one")),
            Player("two"): MctsExpectiminimaxActor.make(
                box,
                Player("two"),
                utility_function=BuildRoutesUf(),
                route_building_probability_calculator=DummyRbpc,
                depth=3,
                breadth=lambda _: 10
            ),
        },
        SimulatedGameActor.make(box),
    )
