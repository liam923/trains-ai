from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Tuple, Optional

import trains.game.turn as gturn
from trains.ai.actor import AiActor
from trains.ai.route_building_probability_calculator import (
    RouteBuildingProbabilityCalculator,
)
from trains.ai.utility_function import UtilityFunction, Utility
from trains.game.action import Action
from trains.game.state import State
from trains.mypy_util import assert_never


@dataclass
class ExpectimaxActor(AiActor):
    depth: int
    utility_function: UtilityFunction
    route_building_probability_calculator: RouteBuildingProbabilityCalculator

    def get_action(self) -> Action:
        action = self._score_state(
            self.state,
            self.depth,
            self.utility_function,
            self.route_building_probability_calculator,
        )[1]
        assert (
            action is not None
        ), "_score_state returned None; check that depth > 0 and it is the player's turn"
        return action

    @classmethod
    def _score_state(
        cls,
        state: State,
        depth: int,
        utility_function: UtilityFunction,
        rbpc: RouteBuildingProbabilityCalculator,
    ) -> Tuple[Utility, Optional[Action], Optional[State]]:
        """
        Calculate a utility for the given state. If the state's turn state is a
        player state and depth > 0, also return the optimal action and resulting state
        from the action
        """

        def _score_state(next_state: State) -> Utility:
            """
            A helper to recursively call _score_state without worrying about params
            besides state and only returning the utility
            """
            return cls._score_state(next_state, depth - 1, utility_function, rbpc)[0]

        # note: action_state is lazy, so this does not cost computation time if not
        # used
        successors = (
            (action, state.next_state(action.action))
            for action in state.get_legal_actions()
        )

        if depth <= 0 or isinstance(state.turn_state, gturn.GameOverTurn):
            return utility_function(state), None, None
        elif isinstance(state.turn_state, gturn.PlayerTurn):
            if state.hand_is_known(state.turn_state.player):
                # if we know the player's hand, then we can do normal expectiminimax

                # perform the action with the highest utility if the current player is
                # the player, otherwise perform the one with the lowest action

                # (for now, we assume that opponents are trying to minimize the player's
                # utility, although this doesn't make too much sense for games with more
                # than two players)

                min_or_max = max if state.turn_state.player == state.player else min
                successors_with_utility = (
                    (_score_state(next_state), action.action, next_state)
                    for action, next_state in successors
                )
                return min_or_max(successors_with_utility, key=operator.itemgetter(0))
            else:
                # If we don't know the player's hand, we consider all possible hands
                # that the player could have. For each hand, we compute its utility and
                # the probability of a player having the hand. Using this, we compute
                # the expected utility across all hands

                possible_hands = state.assumed_hands(state.turn_state.player, rbpc)
                expected_utility = Utility.sum(
                    prob * _score_state(known_state)
                    for known_state, prob in possible_hands
                )
                return expected_utility, None, None
        elif isinstance(state.turn_state, gturn.GameTurn):
            # the utility of the state when it is the game's turn is the sum over all
            # actions of the probability of the game doing the action times the utility
            # of the resulting state

            return (
                Utility.sum(
                    action.probability * _score_state(next_state)
                    for action, next_state in successors
                ),
                None,
                None,
            )
        else:
            assert_never(state.turn_state)
