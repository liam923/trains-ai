from typing import Callable, List, Tuple, Dict

import pytest

from trains.ai.random import RandomActor
from trains.game.action import Action
from trains.game.box import Box, Player
from trains.game.game_actor import SimulatedGameActor
from trains.game.player_actor import PlayerActor
from trains.game.turn import TurnState, RotatingTurn


@pytest.mark.parametrize(
    "box_maker, player_count",
    [(Box.small, 2), (Box.standard, 2), (Box.standard, 4)],
)
@pytest.mark.parametrize("game_index", list(range(20)))
def test_gameplay(
    box_maker: Callable[[List[Player]], Box], player_count: int, game_index: int
):
    """
    Test that when playing a multitude of games with random moves, no data gets out of
    sync between State and GameActor instances, and that no errors are raised
    """
    players = [Player(str(i + 1)) for i in range(player_count)]
    box = box_maker(players)
    player_actors = {player: RandomActor.make(box, player) for player in players}
    game = SimulatedGameActor.make(box)

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

            for op, op_hand in player.state.opponent_hands.items():
                assert (
                    op_hand.remaining_trains == game.player_hands[op].remaining_trains
                )
                assert (
                    op_hand.train_cards_count == game.player_hands[op].train_cards.total
                )
                assert op_hand.destination_cards_count == len(
                    game.player_hands[op].destination_cards
                ) or not isinstance(game.turn_state, RotatingTurn)
                assert op_hand.unselected_destination_cards_count == len(
                    game.player_hands[op].unselected_destination_cards
                ) or not isinstance(game.turn_state, RotatingTurn)
                assert op_hand.train_cards_count >= len(op_hand.known_train_cards)
                assert all(
                    count <= game.player_hands[op].train_cards[color]
                    for color, count in op_hand.known_train_cards.items()
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
