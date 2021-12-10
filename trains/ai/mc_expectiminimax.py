from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable

import trains.game.turn as gturn
from trains.ai.actor import AiActor
from trains.ai.route_building_probability_calculator import (
    RouteBuildingProbabilityCalculator,
)
from trains.ai.utility_function import UtilityFunction, Utility
from trains.game.action import Action
from trains.game.observed_state import ObservedState
from trains.mypy_util import assert_never, cache
from trains.util import randomly_sample_distribution


@dataclass
class McExpectiminimaxActor(AiActor):
    """
    An AI actor that can function as both a simple expectimax actor or as a monte-carlo
    expectimax actor. At each step, the breadth function is called to determine
    what the branching factor should be for that step. If it is None, it allows
    unlimited branching, which is expectiminimax. If it is an integer, it performs that
    number of monte carlo samples.
    """

    utility_function: UtilityFunction
    route_building_probability_calculator: Callable[
        [ObservedState], RouteBuildingProbabilityCalculator
    ]
    depth: int
    breadth: Callable[[int], Optional[int]] = field(default=lambda x: None)
    assume_states: bool = True

    def _get_action(self) -> Action:
        action = self._score_state(self.state, self.depth - 1)[1]
        assert (
            action is not None
        ), "_score_state returned None; check that depth > 0 and it is the player's turn"
        return action

    def _score_state(
        self,
        state: ObservedState,
        current_depth: int,
    ) -> Tuple[Utility, Optional[Action], Optional[ObservedState]]:
        """
        Calculate a utility for the given state. If the state's turn state is a
        player state and depth > 0, also return the optimal action and resulting state
        from the action
        """

        @cache
        def _score_state(next_state: ObservedState) -> Utility:
            """
            A helper to recursively call _score_state without worrying about params
            besides state and only returning the utility
            """
            return self._score_state(next_state, current_depth - 1)[0]

        breadth = self.breadth(current_depth)

        if current_depth <= 0 or isinstance(state.turn_state, gturn.GameOverTurn):
            return self.utility_function(state), None, None
        elif isinstance(state.turn_state, gturn.PlayerTurn):
            if state.hand_is_known(state.turn_state.player) or not self.assume_states:
                # if we know the player's hand, then we can perform the action with the
                # highest utility for the current player
                successors = (
                    (action, state.next_state(action.action))
                    for action in state.get_legal_actions()
                )
                successors_with_utility = (
                    (_score_state(next_state), action.action, next_state)
                    for action, next_state in successors
                )
                return max(
                    successors_with_utility, key=lambda r: r[0][state.turn_state.player]
                )
            else:
                # If we don't know the player's hand, we consider all possible hands
                # that the player could have. For each hand, we compute its utility and
                # the probability of a player having the hand. Using this, we compute
                # the expected utility across all hands
                possible_hands = state.assumed_hands(
                    state.turn_state.player,
                    self.route_building_probability_calculator(state),
                )
                if breadth is None:
                    expected_utility = Utility.sum(
                        prob * _score_state(known_state)
                        for known_state, prob in possible_hands
                    )
                else:
                    # breadth is limited, so we use monte carlo sampling instead
                    mc_hands = randomly_sample_distribution(possible_hands, breadth)
                    expected_utility = (
                        Utility.sum(
                            _score_state(known_state) for known_state in mc_hands
                        )
                        / breadth
                    )
                return expected_utility, None, None
        elif isinstance(state.turn_state, gturn.GameTurn):
            # the utility of the state when it is the game's turn is the sum over all
            # actions of the probability of the game doing the action times the utility
            # of the resulting state

            if breadth is None:
                return (
                    Utility.sum(
                        action.probability
                        * _score_state(state.next_state(action.action))
                        for action in state.get_legal_actions()
                    ),
                    None,
                    None,
                )
            else:
                mc_actions = randomly_sample_distribution(
                    (
                        (action.action, action.probability)
                        for action in state.get_legal_actions()
                    ),
                    breadth,
                )
                return (
                    Utility.sum(
                        _score_state(state.next_state(action)) for action in mc_actions
                    )
                    / breadth,
                    None,
                    None,
                )
        else:
            assert_never(state.turn_state)
