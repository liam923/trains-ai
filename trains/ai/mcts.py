from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Tuple, Generic

import numpy as np
from tqdm import tqdm

from trains.ai.actor import AiActor
from trains.ai.utility_function import Utility, UtilityFunction, ExpectedScoreUf
from trains.game.action import Action
from trains.game.state import State
from trains.game.turn import GameTurn, GameOverTurn, PlayerTurn
from trains.mypy_util import assert_never
from trains.util import randomly_sample_distribution


@dataclass
class _Node(Generic[State]):
    def __init__(
        self, state: State, parent: Optional[Tuple[_Node, int]] = None, c: float = 2
    ):
        self.state = state
        self.parent = parent

        self.children: Dict[Action, _Node] = {}
        self.children_utilities = np.array([])
        self.children_visits = np.array([])
        self.visits = 0

        self.c = c

        self.untried_actions = list(state.get_legal_actions())
        self.action_probabilities = {
            a.action: a.probability for a in self.untried_actions
        }

    @property
    def children_qs(self) -> np.ndarray:
        return self.children_utilities / self.children_visits

    @property
    def children_us(self) -> np.ndarray:
        return np.sqrt(self.c * np.log(self.visits) / self.children_visits)

    @property
    def children_ucts(self) -> np.ndarray:
        # https: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation
        return self.children_qs + self.children_us

    def best_child(self) -> _Node[State]:
        if isinstance(self.state.turn_state, GameTurn):
            action = list(
                randomly_sample_distribution(self.action_probabilities.items(), 1)
            )[0]
            return self.children[action]
        elif isinstance(self.state.turn_state, PlayerTurn):
            best_child_index: int = np.argmax(self.children_ucts)  # type: ignore
            return next(
                itertools.islice(self.children.values(), best_child_index, None)
            )
        else:
            assert not isinstance(self.state.turn_state, GameOverTurn)
            assert_never(self.state.turn_state)

    def select_leaf(self) -> _Node[State]:
        current_node = self
        while current_node.expanded and not current_node.terminal:
            current_node = current_node.best_child()
        return current_node

    @property
    def expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def terminal(self) -> bool:
        return self.state.is_game_over()

    def expand(self) -> _Node[State]:
        if self.expanded:
            return self

        if isinstance(self.state.turn_state, GameTurn):
            action = next(
                randomly_sample_distribution(
                    (a, a.probability) for a in self.untried_actions
                )
            )
        else:
            action = random.choice(self.untried_actions)

        child = _Node(
            self.state.next_state(action.action), (self, len(self.children)), c=self.c
        )

        self.untried_actions.remove(action)
        self.children[action.action] = child
        self.action_probabilities[action.action] = action.probability
        self.children_utilities = np.append(self.children_utilities, 0)
        self.children_visits = np.append(self.children_visits, 0)

        return child

    def backup(self, utility: Utility) -> None:
        current_node = self
        while current_node.parent is not None:
            current_node.parent[0].children_visits[current_node.parent[1]] += 1
            current_node.visits += 1
            if isinstance(current_node.parent[0].state.turn_state, PlayerTurn):
                current_node.parent[0].children_utilities[
                    current_node.parent[1]
                ] += utility[current_node.parent[0].state.turn_state.player]
            elif isinstance(
                current_node.parent[0].state.turn_state, GameOverTurn
            ) or isinstance(current_node.parent[0].state.turn_state, GameTurn):
                pass
            else:
                assert_never(current_node.parent[0].state.turn_state)
            current_node = current_node.parent[0]
        current_node.visits += 1


@dataclass  # type: ignore
class MctsActor(AiActor[State], ABC):
    """
    Based on:
    https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Principle_of_operation
    https://web.archive.org/web/20160308053456/http://mcts.ai/code/python.html
    https://github.com/brilee/python_uct/blob/master/numpy_impl.py
    """

    iterations: int
    tree: _Node = None  # type: ignore
    show_progress: bool = True

    def __post_init__(self) -> None:
        if self.tree is None:
            self.tree = _Node(self.state)  # type: ignore

    def observe_action(self, action: Action) -> None:
        super().observe_action(action)
        if action is self.tree.children:
            self.tree = self.tree.children[action]
            self.tree.parent = None
        else:
            self.tree = _Node(self.tree.state.next_state(action))

    def _get_action(self) -> Action:
        iterations = range(self.iterations)
        if self.show_progress:
            iterations = tqdm(iterations, desc="Thinking")
        for _ in iterations:
            leaf = self.tree.select_leaf()  # selection
            leaf = leaf.expand()  # expansion
            winner = self._get_state_utilities(leaf.state)  # simulation
            leaf.backup(winner)  # backup
        best_child_index: int = np.argmax(self.tree.children_qs)  # type: ignore
        return next(itertools.islice(self.tree.children.keys(), best_child_index, None))

    @abstractmethod
    def _get_state_utilities(self, state: State) -> Utility:
        pass


@dataclass
class BasicMctsActor(MctsActor):
    def _get_state_utilities(self, state: State) -> Utility:
        current_state = state
        while not current_state.is_game_over():
            legal_actions = current_state.get_legal_actions()
            if isinstance(current_state.turn_state, GameTurn):
                action = next(
                    randomly_sample_distribution(
                        ((a.action, a.probability) for a in legal_actions), 1
                    )
                )
            else:
                action = random.choice([a.action for a in legal_actions])
            current_state = current_state.next_state(action)
        return Utility({current_state.winner(): 1})


@dataclass
class UfMctsActor(MctsActor):
    utility_function: UtilityFunction = ExpectedScoreUf()

    def _get_state_utilities(self, state: State) -> Utility:
        return self.utility_function(state)
