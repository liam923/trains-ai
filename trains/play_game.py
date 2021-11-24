from trains.game.box import Box, Player
from trains.game.game_actor import play_game, SimulatedGameActor
from trains.game.player_actor import UserActor

if __name__ == "__main__":
    players = [Player("one"), Player("two")]
    box = Box.small(players)
    play_game(
        {player: UserActor.make(box, player) for player in players},
        SimulatedGameActor.make(box),
    )
