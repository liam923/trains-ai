from __future__ import annotations

import itertools
import random
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Dict,
    List,
    Set,
    Optional,
    Union,
    FrozenSet,
    Iterable,
    Tuple,
    DefaultDict,
)

import trains.game.action as gaction
import trains.game.turn as gturn
from trains.error import TrainsException
from trains.game.action import Action
from trains.game.actor import Actor
from trains.game.box import (
    DestinationCard,
    TrainCard,
    Box,
    Player,
    Route,
    TrainCards,
    frozendict,
)
from trains.game.clusters import Clusters
from trains.game.player_actor import PlayerActor
from trains.game.turn import TurnState
from trains.mypy_util import assert_never
from trains.util import (
    sufficient_cards_to_build,
    merge_train_cards,
    subtract_train_cards,
)


@dataclass  # type: ignore
class GameActor(Actor, ABC):
    """
    An Actor that is responsible for verifying that moves are legal and executing
    "game" moves (card draws, e.g.)
    """

    @dataclass
    class PlayerHand:
        destination_cards: Set[DestinationCard]
        train_cards: TrainCards
        remaining_trains: int
        unselected_destination_cards: FrozenSet[DestinationCard]

    box: Box
    player_hands: Dict[Player, PlayerHand]
    face_up_train_cards: TrainCards
    discarded_train_cards: TrainCards
    built_routes: Dict[Route, Player]
    turn_state: TurnState

    def validate_action(self, action: Action) -> Optional[str]:
        unexpected_action_message = f"unexpected action type {type(action)}"

        def verify_train_draw(
            action: gaction.TrainCardPickAction, second: bool = False
        ) -> Optional[str]:
            if action.draw_known:
                if self.face_up_train_cards[action.selected_card_if_known] <= 0:
                    return "There are no face up cards of the given color"
            else:
                if self._train_card_deck_empty:
                    return "There are no train cards left to draw"
            if second and action.draw_known and action.selected_card_if_known is None:
                return "Cannot select wildcard on second draw"
            return None

        def verify_destination_pick(
            turn_state: gturn.PlayerTurn,
            min_cards: int,
        ) -> Optional[str]:
            if isinstance(action, gaction.DestinationCardSelectionAction):
                if len(action.selected_cards) < min_cards:
                    return f"Number of selected destination cards must be at least {min_cards}"
                if not action.selected_cards.issubset(
                    self.player_hands[turn_state.player].unselected_destination_cards
                ):
                    return f"Selected destination cards are not valid"
                else:
                    return None
            else:
                return unexpected_action_message

        if isinstance(self.turn_state, gturn.PlayerInitialDestinationCardChoiceTurn):
            return verify_destination_pick(
                self.turn_state, self.box.starting_destination_cards_range[0]
            )
        elif isinstance(self.turn_state, gturn.PlayerStartTurn):
            if isinstance(action, gaction.PassAction):
                return None
            elif isinstance(action, gaction.BuildAction):
                if action.route in self.built_routes:
                    return "Route is already built"
                if len(
                    self.box.players
                ) < self.box.double_routes_player_minimum and any(
                    double_route in self.built_routes
                    for double_route in self.box.board.double_routes[action.route]
                ):
                    return f"Cannot build double routes in games with less than {self.box.double_routes_player_minimum} players"
                if any(
                    self.built_routes.get(double_route, None) == self.turn_state.player
                    for double_route in self.box.board.double_routes[action.route]
                ):
                    return "Cannot build route. You cannot build two routes next to each other"
                if not sufficient_cards_to_build(
                    action.route, action.train_cards
                ) or not all(
                    count
                    <= self.player_hands[self.turn_state.player].train_cards[color]
                    for color, count in action.train_cards.items()
                ):
                    return "Not enough train cards to build route"
                if (
                    self.player_hands[self.turn_state.player].remaining_trains
                    < action.route.length
                ):
                    return "Not enough trains to build route"
                return None
            elif isinstance(action, gaction.TrainCardPickAction):
                return verify_train_draw(action)
            elif isinstance(action, gaction.DestinationCardPickAction):
                if self._destination_card_deck_empty:
                    return "There are no destination cards left to draw"
                else:
                    return None
            else:
                return unexpected_action_message
        elif isinstance(self.turn_state, gturn.PlayerTrainCardDrawMidTurn):
            if isinstance(action, gaction.TrainCardPickAction):
                return verify_train_draw(action, second=True)
            else:
                return unexpected_action_message
        elif isinstance(self.turn_state, gturn.PlayerDestinationCardDrawMidTurn):
            return verify_destination_pick(
                self.turn_state, self.box.dealt_destination_cards_range[0]
            )
        elif isinstance(self.turn_state, gturn.DestinationCardDealTurn):
            if isinstance(action, gaction.DestinationCardDealAction):
                return None
            else:
                return unexpected_action_message
        elif isinstance(self.turn_state, gturn.TrainCardDealTurn):
            if isinstance(action, gaction.TrainCardDealAction):
                return None
            else:
                return unexpected_action_message
        elif isinstance(self.turn_state, gturn.RevealInitialDestinationCardChoicesTurn):
            if isinstance(action, gaction.RevealDestinationCardSelectionsAction):
                return None
            else:
                return unexpected_action_message
        elif isinstance(self.turn_state, gturn.GameOverTurn):
            if isinstance(action, gaction.PassAction):
                return None
            else:
                return unexpected_action_message
        elif isinstance(self.turn_state, gturn.InitialTurn):
            if isinstance(action, gaction.InitialDealAction):
                return None
            else:
                return unexpected_action_message
        elif isinstance(self.turn_state, gturn.RevealFinalDestinationCardsTurn):
            if isinstance(action, gaction.RevealFinalDestinationCardsAction):
                return None
            else:
                return unexpected_action_message
        else:
            assert_never(self.turn_state)

    def observe_action(self, action: Action) -> None:
        self.turn_state = self._record_action(action)

    def _record_action(self, action: Action) -> TurnState:
        unexpected_action_error = TrainsException(
            f"unexpected action type {type(action)}"
        )

        def perform_train_draw(
            turn_state: Union[gturn.PlayerStartTurn, gturn.PlayerTrainCardDrawMidTurn],
            action: gaction.TrainCardPickAction,
        ) -> None:
            if action.draw_known:
                self.player_hands[turn_state.player].train_cards = self.player_hands[
                    turn_state.player
                ].train_cards.incrementing(action.selected_card_if_known, 1)
                self.face_up_train_cards = self.face_up_train_cards.incrementing(
                    action.selected_card_if_known, -1
                )

        if isinstance(self.turn_state, gturn.PlayerInitialDestinationCardChoiceTurn):
            if isinstance(action, gaction.DestinationCardSelectionAction):
                self._recycle_destination_cards(
                    self.player_hands[
                        self.turn_state.player
                    ].unselected_destination_cards
                    - action.selected_cards
                )
                self.player_hands[self.turn_state.player].destination_cards = set(
                    action.selected_cards
                )
                self.player_hands[
                    self.turn_state.player
                ].unselected_destination_cards = frozenset()
                next_player = self.box.next_player_map[self.turn_state.player]
                if next_player == self.box.players[0]:
                    return gturn.RevealInitialDestinationCardChoicesTurn()
                else:
                    return gturn.PlayerInitialDestinationCardChoiceTurn(next_player)
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.PlayerStartTurn):
            last_turn_started = (
                self.turn_state.last_turn_started
                or self.player_hands[self.turn_state.player].remaining_trains
                <= self.box.trains_to_end
            )
            if isinstance(action, gaction.PassAction):
                return gturn.PlayerStartTurn.make_or_end(
                    last_turn_started,
                    self.box.next_player_map[self.turn_state.player],
                )
            elif isinstance(action, gaction.BuildAction):
                self.built_routes[action.route] = self.turn_state.player
                self.player_hands[
                    self.turn_state.player
                ].remaining_trains -= action.route.length
                self.player_hands[
                    self.turn_state.player
                ].train_cards = subtract_train_cards(
                    self.player_hands[self.turn_state.player].train_cards,
                    action.train_cards,
                )[
                    0
                ]
                self.discarded_train_cards = merge_train_cards(
                    self.discarded_train_cards, action.train_cards
                )

                return gturn.PlayerStartTurn.make_or_end(
                    last_turn_started,
                    self.box.next_player_map[self.turn_state.player],
                )
            elif isinstance(action, gaction.TrainCardPickAction):
                perform_train_draw(self.turn_state, action)
                if action.draw_known and action.selected_card_if_known is None:
                    next_turn_state: TurnState = gturn.PlayerStartTurn.make_or_end(
                        last_turn_started,
                        self.box.next_player_map[self.turn_state.player],
                    )
                else:
                    next_turn_state = gturn.PlayerTrainCardDrawMidTurn(
                        last_turn_started, self.turn_state.player
                    )
                return gturn.TrainCardDealTurn(
                    count=1,
                    to_player=None if action.draw_known else self.turn_state.player,
                    next_turn_state=next_turn_state,
                )
            elif isinstance(action, gaction.DestinationCardPickAction):
                return gturn.DestinationCardDealTurn(
                    last_turn_started, self.turn_state.player
                )
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.PlayerTrainCardDrawMidTurn):
            if isinstance(action, gaction.TrainCardPickAction):
                perform_train_draw(self.turn_state, action)
                return gturn.TrainCardDealTurn(
                    count=1,
                    to_player=None if action.draw_known else self.turn_state.player,
                    next_turn_state=gturn.PlayerStartTurn.make_or_end(
                        self.turn_state.last_turn_started,
                        self.box.next_player_map[self.turn_state.player],
                    ),
                )
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.PlayerDestinationCardDrawMidTurn):
            if isinstance(action, gaction.DestinationCardSelectionAction):
                self._recycle_destination_cards(
                    self.player_hands[
                        self.turn_state.player
                    ].unselected_destination_cards
                    - action.selected_cards
                )
                self.player_hands[self.turn_state.player].destination_cards.update(
                    action.selected_cards
                )
                self.player_hands[
                    self.turn_state.player
                ].unselected_destination_cards = frozenset()
                return gturn.PlayerStartTurn.make_or_end(
                    self.turn_state.last_turn_started,
                    self.box.next_player_map[self.turn_state.player],
                )
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.DestinationCardDealTurn):
            if isinstance(action, gaction.DestinationCardDealAction):
                self.player_hands[
                    self.turn_state.to_player
                ].unselected_destination_cards = action.cards
                return gturn.PlayerDestinationCardDrawMidTurn(
                    self.turn_state.last_turn_started, self.turn_state.to_player
                )
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.TrainCardDealTurn):
            if isinstance(action, gaction.TrainCardDealAction):
                if self.turn_state.to_player is None:
                    for card, count in action.cards.items():
                        self.face_up_train_cards = (
                            self.face_up_train_cards.incrementing(card, count)
                        )
                    if self.face_up_train_cards[None] >= self.box.wildcards_to_clear:
                        self.face_up_train_cards = TrainCards()
                        return gturn.TrainCardDealTurn(
                            count=self.box.face_up_train_cards,
                            to_player=None,
                            next_turn_state=self.turn_state.next_turn_state,
                        )
                else:
                    self.player_hands[
                        self.turn_state.to_player
                    ].train_cards = merge_train_cards(
                        self.player_hands[self.turn_state.to_player].train_cards,
                        action.cards,
                    )
                return self.turn_state.next_turn_state
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.RevealInitialDestinationCardChoicesTurn):
            if isinstance(action, gaction.RevealDestinationCardSelectionsAction):
                return gturn.PlayerStartTurn.make_or_end(False, self.box.players[0])
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.GameOverTurn):
            if isinstance(action, gaction.PassAction):
                return gturn.GameOverTurn()
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.InitialTurn):
            if isinstance(action, gaction.InitialDealAction):
                for player, train_cards in action.train_cards.items():
                    self.player_hands[player].train_cards = train_cards
                for player, destination_cards in action.destination_cards.items():
                    self.player_hands[
                        player
                    ].unselected_destination_cards = destination_cards
                self.face_up_train_cards = merge_train_cards(
                    self.face_up_train_cards, action.face_up_train_cards
                )
                return gturn.PlayerInitialDestinationCardChoiceTurn(self.box.players[0])
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.RevealFinalDestinationCardsTurn):
            if isinstance(action, gaction.RevealFinalDestinationCardsAction):
                return gturn.GameOverTurn()
            else:
                raise unexpected_action_error
        else:
            assert_never(self.turn_state)

    @property
    def _destination_card_deck_empty(self) -> bool:
        cards_not_in_pile = sum(
            len(hand.destination_cards) + len(hand.unselected_destination_cards)
            for hand in self.player_hands.values()
        )
        deck_size = len(self.box.destination_cards)
        return deck_size - cards_not_in_pile == 0

    @property
    def _train_card_deck_empty(self) -> bool:
        cards_not_in_pile = (
            sum(hand.train_cards.total for hand in self.player_hands.values())
            + len(self.face_up_train_cards)
            + len(self.discarded_train_cards)
        )
        deck_size = self.box.train_cards.total
        return deck_size - cards_not_in_pile == 0

    def _recycle_destination_cards(self, cards: Iterable[DestinationCard]) -> None:
        """
        Recycle the given destination cards back into the deck. This function is meant
        to be overridden by subclasses where this is necessary.
        """

    @property
    def is_over(self) -> bool:
        return isinstance(self.turn_state, gturn.GameOverTurn)

    @property
    def scores(self) -> Dict[Player, int]:
        player_score_infos = {
            player: self._get_player_score_info(player) for player in self.box.players
        }

        # add in longest path bonus
        longest_path = max(
            path_length for _, path_length in player_score_infos.values()
        )
        for player, (raw_score, path_length) in player_score_infos.items():
            if path_length == longest_path:
                player_score_infos[player] = (
                    raw_score + self.box.longest_path_bonus,
                    path_length,
                )

        return {player: score for player, (score, _) in player_score_infos.items()}

    def _get_player_score_info(self, player: Player) -> Tuple[int, int]:
        """
        Calculate the player's score from routes and destination cards, along with the
        length of their longest route.
        """
        built_routes = {
            route for route, builder in self.built_routes.items() if builder == player
        }
        clusters = Clusters(frozenset(), self.box.board.shortest_paths)
        for route in built_routes:
            clusters.connect(*route.cities)

        cards_points = sum(
            card.value * (1 if clusters.is_connected(card.cities) else -1)
            for card in self.player_hands[player].destination_cards
        )
        routes_points = sum(
            self.box.route_point_values[route.length] for route in built_routes
        )
        longest = max(
            (
                self.box.board.shortest_path(*cities)
                for cluster in clusters.clusters
                for cities in itertools.combinations(cluster, 2)
            ),
            default=0,
        )

        return cards_points + routes_points, longest


@dataclass  # type: ignore
class SimulatedGameActor(GameActor):
    """
    An implementation of GameActor that simulates the game.
    """

    destination_card_pile: List[DestinationCard]  # last is top of pile
    train_card_pile: List[TrainCard]  # last is top of pile

    @staticmethod
    def make(box: Box) -> SimulatedGameActor:
        train_card_pile = [
            color for color, count in box.train_cards.items() for _ in range(count)
        ]
        destination_card_pile = list(box.destination_cards)
        random.shuffle(train_card_pile)
        random.shuffle(destination_card_pile)
        return SimulatedGameActor(
            box=box,
            turn_state=gturn.InitialTurn(),
            player_hands={
                player: GameActor.PlayerHand(
                    destination_cards=set(),
                    train_cards=TrainCards(),
                    remaining_trains=box.starting_train_count,
                    unselected_destination_cards=frozenset(),
                )
                for player in box.players
            },
            face_up_train_cards=TrainCards(),
            discarded_train_cards=TrainCards(),
            built_routes={},
            destination_card_pile=destination_card_pile,
            train_card_pile=train_card_pile,
        )

    def get_action(self) -> Action:
        if isinstance(self.turn_state, gturn.GameTurn):
            if isinstance(self.turn_state, gturn.TrainCardDealTurn):
                return gaction.TrainCardDealAction(
                    self._deal_train_cards(self.turn_state.count)
                )
            elif isinstance(self.turn_state, gturn.DestinationCardDealTurn):
                return gaction.DestinationCardDealAction(
                    self._deal_destination_cards(
                        self.box.dealt_destination_cards_range[1]
                    )
                )
            elif isinstance(
                self.turn_state, gturn.RevealInitialDestinationCardChoicesTurn
            ):
                return gaction.RevealDestinationCardSelectionsAction(
                    frozendict(
                        {
                            player: len(hand.destination_cards)
                            for player, hand in self.player_hands.items()
                        }
                    )
                )
            elif isinstance(self.turn_state, gturn.InitialTurn):
                return gaction.InitialDealAction(
                    train_cards=frozendict(
                        {
                            player: self._deal_train_cards(
                                self.box.starting_train_cards_count
                            )
                            for player in self.box.players
                        }
                    ),
                    destination_cards=frozendict(
                        {
                            player: self._deal_destination_cards(
                                self.box.starting_destination_cards_range[1]
                            )
                            for player in self.box.players
                        }
                    ),
                    face_up_train_cards=self._deal_train_cards(
                        self.box.face_up_train_cards
                    ),
                )
            elif isinstance(self.turn_state, gturn.RevealFinalDestinationCardsTurn):
                return gaction.RevealFinalDestinationCardsAction(
                    frozendict(
                        {
                            player: frozenset(hand.destination_cards)
                            for player, hand in self.player_hands.items()
                        }
                    )
                )
            else:
                assert_never(self.turn_state)
        else:
            raise TrainsException(f"Unexpected turn state {type(self.turn_state)}")

    def _deal_train_cards(self, count: int) -> TrainCards:
        cards: DefaultDict[TrainCard, int] = defaultdict(int)
        for _ in range(count):
            if len(self.train_card_pile) == 0:
                # need to reshuffle deck
                if len(self.discarded_train_cards) == 0:
                    # there are no train cards left to be drawn
                    break
                self.train_card_pile = [
                    card
                    for card, count in self.discarded_train_cards.items()
                    for _ in range(count)
                ]
                random.shuffle(self.train_card_pile)
            cards[self.train_card_pile.pop()] += 1
        return TrainCards(cards)

    def _deal_destination_cards(self, count: int) -> FrozenSet[DestinationCard]:
        destination_cards: List[DestinationCard] = []
        for _ in range(
            min(
                count,
                len(self.destination_card_pile),
            )
        ):
            destination_cards.append(self.destination_card_pile.pop())
        return frozenset(destination_cards)

    def _recycle_destination_cards(self, cards: Iterable[DestinationCard]) -> None:
        for card in cards:
            self.destination_card_pile.insert(0, card)


def play_game(players: Dict[Player, PlayerActor], game: GameActor) -> None:
    actors: List[Actor] = list(players.values()) + [game]  # type: ignore

    history = []
    while not game.is_over:
        if game.turn_state.player is None:
            action = game.get_action()
        else:
            action = players[game.turn_state.player].get_action()

        error = next(
            (
                error
                for error in (actor.validate_action(action) for actor in actors)
                if error is not None
            ),
            None,
        )
        if error is not None:
            print(f"Error: {error}")
        else:
            history.append((game.turn_state, action))
            for actor in actors:
                actor.observe_action(action)

    for player, score in game.scores.items():
        print(f"{player} scored {score}")
