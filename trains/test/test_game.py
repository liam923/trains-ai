from random import Random
from typing import Callable, List, Tuple

import pytest

from trains.game.action import Action, InitialDealAction
from trains.game.box import Box, Player
from trains.game.game_actor import SimulatedGameActor
from trains.game.player_actor import RandomActor
from trains.game.turn import TurnState


@pytest.mark.parametrize(
    "box_maker, player_count",
    [
        (Box.small, 2),
        # (Box.standard, 2),
        # (Box.standard, 4),
    ],
)
@pytest.mark.parametrize("seed", [1])
def test_gameplay(
    box_maker: Callable[[List[Player], Random], Box], player_count: int, seed: int
):
    """
    Test playing a multitude of games with random moves and check that no obvious
    errors occur.
    """
    seeder = Random(seed)
    players = [Player(str(i + 1)) for i in range(player_count)]
    box = box_maker(players, Random(seeder.random()))
    player_actors = {
        player: RandomActor.make(box, player, randomizer=Random(seeder.random()))
        for player in players
    }
    game = SimulatedGameActor.make(box, seed=seeder.random())

    actors: List[Actor] = list(player_actors.values()) + [game]  # type: ignore
    history: List[Tuple[TurnState, Action]] = []
    while game.winner is None:
        for p, player in player_actors.items():
            assert player.turn_state == game.turn_state
            assert player.state.face_up_train_cards == game.face_up_train_cards
            assert player.state.discarded_train_cards == game.discarded_train_cards
            assert player.state.built_routes == game.built_routes
            assert (
                player.state.hand.destination_cards
                == game.player_hands[p].destination_cards
            )
            assert (
                player.state.hand.unselected_destination_cards
                == game.player_hands[p].unselected_destination_cards
            )
            assert player.state.hand.train_cards == game.player_hands[p].train_cards
            assert (
                player.state.hand.remaining_trains
                == game.player_hands[p].remaining_trains
            )

        if game.turn_state.player is None:
            action = game.get_action()
        else:
            action = player_actors[game.turn_state.player].get_action()

        for actor in actors:
            error = actor.validate_action(action)
            assert error is None, error

        history.append((game.turn_state, action))
        for actor in actors:
            actor.observe_action(action)

    assert game.winner is not None
