from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

from trains.ai.actor import AiActor
from trains.ai.utility_function import Utility, UtilityFunction, ExpectedScoreUf
from trains.game.action import Action
from trains.game.known_state import KnownState
from trains.game.turn import GameTurn, GameOverTurn, PlayerTurn
from trains.mypy_util import assert_never
from trains.util import randomly_sample_distribution


@dataclass
class _Node:
    def __init__(self, state: KnownState, parent: Optional[Tuple[_Node, int]] = None):
        self.state = state
        self.parent = parent
        self.children: Dict[Action, _Node] = {}
        self.children_utilities = np.array([])
        self.children_visits = np.array([])
        self.visits = 0

        self.untried_actions = list(state.get_legal_actions())
        self.action_probabilities = {
            a.action: a.probability for a in self.untried_actions
        }

    state: KnownState
    parent: Optional[Tuple[_Node, int]] = None
    children: Dict[Action, _Node] = field(default_factory=dict)
    children_utilities: np.ndarray = field(default_factory=partial(np.ndarray, []))
    children_visits: np.ndarray = field(default_factory=partial(np.ndarray, []))
    visits: int = 0
    action_probabilities: Dict[Action, float] = field(default_factory=dict)

    @property
    def children_qs(self) -> np.ndarray:
        return self.children_utilities / self.children_visits

    @property
    def children_us(self) -> np.ndarray:
        return np.sqrt(2 * np.log(self.visits) / self.children_visits)

    @property
    def children_ucts(self) -> np.ndarray:
        return self.children_qs + self.children_us

    def best_child(self) -> _Node:
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

    def select_leaf(self) -> _Node:
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

    def expand(self) -> _Node:
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

        child = _Node(self.state.next_state(action.action), (self, len(self.children)))

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
                assert_never(current_node.state.turn_state)
            current_node = current_node.parent[0]
        current_node.visits += 1


@dataclass  # type: ignore
class MctsActor(AiActor[KnownState], ABC):
    """
    Based on:
    https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Principle_of_operation
    https://web.archive.org/web/20160308053456/http://mcts.ai/code/python.html
    https://github.com/brilee/python_uct/blob/master/numpy_impl.py
    """

    iterations: int
    tree: _Node = None  # type: ignore

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

    def get_action(self) -> Action:
        for _ in tqdm(range(self.iterations)):
            leaf = self.tree.select_leaf()  # selection
            leaf = leaf.expand()  # expansion
            winner = self._get_state_winner(leaf.state)  # simulation
            leaf.backup(winner)  # backup
        best_child_index: int = np.argmax(self.tree.children_qs)  # type: ignore
        action = next(
            itertools.islice(self.tree.children.keys(), best_child_index, None)
        )
        print(action)
        return action

    @abstractmethod
    def _get_state_winner(self, state: KnownState) -> Utility:
        pass


@dataclass
class BasicMctsActor(MctsActor):
    def _get_state_winner(self, state: KnownState) -> Utility:
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

    def get_action(self) -> Action:
        action = super().get_action()

        print(f"Current: {self.utility_function(self.state)[self.player]}")
        for pos_action in self.state.get_legal_actions():
            print(
                f"{pos_action.action}: {self.utility_function(self.state.next_state(pos_action.action))[self.player]}"
            )

        return action

    def _get_state_winner(self, state: KnownState) -> Utility:
        return self.utility_function(state)
