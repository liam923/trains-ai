from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import (
    Callable,
    Optional,
    FrozenSet,
    Generator,
    Tuple,
    Dict,
    Union,
    Iterable,
)

from frozendict import frozendict

import trains.game.action as gaction
import trains.game.turn as gturn
from trains.error import TrainsException
from trains.game.action import Action
from trains.game.box import Player, DestinationCard, Box, TrainCards, Route, Color
from trains.game.clusters import Clusters
from trains.game.state import AbstractState, State, ObservedHandState, KnownHandState
from trains.game.turn import TurnState
from trains.mypy_util import assert_never
from trains.util import (
    subtract_train_cards,
    merge_train_cards,
    Cons,
    probability_of_having_cards,
)


@dataclass(frozen=True)
class KnownState(AbstractState):
    hands: frozendict[Player, KnownHandState]

    @property
    def player_hands(self) -> frozendict[Player, KnownHandState]:
        return self.hands

    def player_hand(self, player: Player) -> KnownHandState:
        return self.hands[player]

    @classmethod
    def make(cls, box: Box, player: Player) -> KnownState:
        return KnownState(
            box=box,
            hands=frozendict(
                {
                    p: KnownHandState(
                        destination_cards=frozenset(),
                        unselected_destination_cards=frozenset(),
                        train_cards=TrainCards(),
                        remaining_trains=box.starting_train_count,
                        points_so_far=box.starting_score,
                        complete_destination_cards=frozenset(),
                        incomplete_destination_cards=frozenset(),
                    )
                    for p in box.players
                }
            ),
            discarded_train_cards=TrainCards(),
            face_up_train_cards=TrainCards(),
            train_card_pile_distribution=box.train_cards,
            destination_card_pile_distribution=box.destination_cards,
            destination_card_pile_size=len(box.destination_cards),
            built_routes=frozendict({}),
            built_clusters=frozendict(
                {
                    player: Clusters(frozenset(), box.board.shortest_paths)
                    for player in box.players
                }
            ),
            turn_state=gturn.InitialTurn(),
        )

    def next_state(self, action: Action) -> KnownState:
        unexpected_action_error = TrainsException(
            f"unexpected action type {type(action)}"
        )

        def perform_train_draw(
            turn_state: Union[gturn.PlayerStartTurn, gturn.PlayerTrainCardDrawMidTurn],
            action: gaction.TrainCardPickAction,
            last_turn_started: bool,
            second_draw: bool = False,
        ) -> KnownState:
            if action.draw_known:
                next_turn_state = gturn.TrainCardDealTurn(
                    1,
                    None,
                    gturn.PlayerStartTurn.make_or_end(
                        last_turn_started,
                        self.box.next_player_map[turn_state.player],
                    )
                    if second_draw or action.selected_card_if_known is None
                    else gturn.PlayerTrainCardDrawMidTurn(
                        last_turn_started,
                        turn_state.player,
                    ),
                )
                old_hand = self.hands[turn_state.player]
                return replace(
                    self,
                    turn_state=next_turn_state,
                    hands=frozendict(
                        {
                            **self.hands,
                            turn_state.player: replace(
                                old_hand,
                                train_cards=old_hand.train_cards.incrementing(
                                    action.selected_card_if_known, 1
                                ),
                            ),
                        }
                    ),
                    face_up_train_cards=self.face_up_train_cards.incrementing(
                        action.selected_card_if_known, -1
                    ),
                )
            else:
                return replace(
                    self,
                    turn_state=gturn.TrainCardDealTurn(
                        1,
                        self.turn_state.player,
                        gturn.PlayerStartTurn.make_or_end(
                            last_turn_started,
                            self.box.next_player_map[turn_state.player],
                        )
                        if second_draw
                        else gturn.PlayerTrainCardDrawMidTurn(
                            last_turn_started, turn_state.player
                        ),
                    ),
                )

        if isinstance(self.turn_state, gturn.PlayerInitialDestinationCardChoiceTurn):
            if isinstance(action, gaction.DestinationCardSelectionAction):
                next_player = self.box.next_player_map[self.turn_state.player]
                if next_player == self.box.players[0]:
                    next_turn: TurnState = gturn.PlayerStartTurn(
                        last_turn_started=False, player=next_player
                    )
                else:
                    next_turn = gturn.PlayerInitialDestinationCardChoiceTurn(
                        next_player
                    )
                old_hand = self.hands[self.turn_state.player]
                return replace(
                    self,
                    turn_state=next_turn,
                    hands=frozendict(
                        {
                            **self.hands,
                            self.turn_state.player: replace(
                                old_hand,
                                unselected_destination_cards=frozenset(),
                                destination_cards=action.selected_cards,
                                incomplete_destination_cards=action.selected_cards,
                            ),
                        }
                    ),
                    destination_card_pile_distribution=self.destination_card_pile_distribution
                    | (old_hand.unselected_destination_cards - action.selected_cards),
                    destination_card_pile_size=self.destination_card_pile_size
                    + (
                        len(old_hand.unselected_destination_cards)
                        - len(action.selected_cards)
                    ),
                )
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.PlayerStartTurn):
            old_hand = self.hands[self.turn_state.player]
            last_turn_started = (
                self.turn_state.last_turn_started
                or old_hand.remaining_trains <= self.box.trains_to_end
            )
            if isinstance(action, gaction.PassAction):
                return replace(
                    self,
                    turn_state=gturn.PlayerStartTurn.make_or_end(
                        last_turn_started,
                        self.box.next_player_map[self.turn_state.player],
                    ),
                )
            elif isinstance(action, gaction.BuildAction):
                new_cluster = self.built_clusters[self.turn_state.player].connect(
                    *action.route.cities
                )
                completed_destination_cards = {
                    card
                    for card in old_hand.incomplete_destination_cards
                    if new_cluster.is_connected(card.cities)
                }
                return replace(
                    self,
                    turn_state=gturn.PlayerStartTurn.make_or_end(
                        last_turn_started,
                        self.box.next_player_map[self.turn_state.player],
                    ),
                    built_routes=frozendict(
                        {
                            **self.built_routes,
                            action.route: self.turn_state.player,
                        }
                    ),
                    built_clusters=frozendict(
                        {
                            **self.built_clusters,
                            self.turn_state.player: new_cluster,
                        }
                    ),
                    hands=frozendict(
                        {
                            **self.hands,
                            self.turn_state.player: replace(
                                old_hand,
                                train_cards=subtract_train_cards(
                                    old_hand.train_cards, action.train_cards
                                )[0],
                                remaining_trains=old_hand.remaining_trains
                                - action.route.length,
                                points_so_far=old_hand.points_so_far
                                + self.box.route_point_values[action.route.length]
                                + sum(
                                    card.value for card in completed_destination_cards
                                ),
                                complete_destination_cards=old_hand.complete_destination_cards
                                | completed_destination_cards,
                                incomplete_destination_cards=old_hand.incomplete_destination_cards
                                - completed_destination_cards,
                            ),
                        }
                    ),
                    discarded_train_cards=merge_train_cards(
                        self.discarded_train_cards, action.train_cards
                    ),
                )
            elif isinstance(action, gaction.TrainCardPickAction):
                return perform_train_draw(self.turn_state, action, last_turn_started)
            elif isinstance(action, gaction.DestinationCardPickAction):
                return replace(
                    self,
                    turn_state=gturn.DestinationCardDealTurn(
                        last_turn_started, self.turn_state.player
                    ),
                )
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.PlayerTrainCardDrawMidTurn):
            if isinstance(action, gaction.TrainCardPickAction):
                return perform_train_draw(
                    self.turn_state,
                    action,
                    self.turn_state.last_turn_started,
                    second_draw=True,
                )
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.PlayerDestinationCardDrawMidTurn):
            if isinstance(action, gaction.DestinationCardSelectionAction):
                next_turn_state: TurnState = gturn.PlayerStartTurn.make_or_end(
                    self.turn_state.last_turn_started,
                    self.box.next_player_map[self.turn_state.player],
                )
                old_hand = self.hands[self.turn_state.player]
                return replace(
                    self,
                    hands=frozendict(
                        {
                            **self.hands,
                            self.turn_state.player: replace(
                                old_hand,
                                unselected_destination_cards=frozenset(),
                                destination_cards=old_hand.destination_cards
                                | action.selected_cards,
                                incomplete_destination_cards=old_hand.incomplete_destination_cards
                                | action.selected_cards,
                            ),
                        }
                    ),
                    turn_state=next_turn_state,
                    destination_card_pile_distribution=self.destination_card_pile_distribution
                    | (old_hand.unselected_destination_cards - action.selected_cards),
                    destination_card_pile_size=self.destination_card_pile_size
                    + len(old_hand.unselected_destination_cards)
                    - len(action.selected_cards),
                )
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.DestinationCardDealTurn):
            if isinstance(action, gaction.DestinationCardDealAction):
                next_turn_state = gturn.PlayerDestinationCardDrawMidTurn(
                    self.turn_state.last_turn_started, self.turn_state.to_player
                )
                return replace(
                    self,
                    turn_state=next_turn_state,
                    hands=frozendict(
                        {
                            **self.hands,
                            self.turn_state.to_player: replace(
                                self.hands[self.turn_state.to_player],
                                unselected_destination_cards=action.cards,
                            ),
                        }
                    ),
                    destination_card_pile_distribution=self.destination_card_pile_distribution
                    - action.cards,
                    destination_card_pile_size=self.destination_card_pile_size
                    - len(action.cards),
                )
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.TrainCardDealTurn):
            if isinstance(action, gaction.TrainCardDealAction):
                if self.turn_state.to_player is None:
                    new_face_up_cards = merge_train_cards(
                        self.face_up_train_cards, action.cards
                    )
                    if new_face_up_cards[None] >= self.box.wildcards_to_clear:
                        return replace(
                            self,
                            face_up_train_cards=TrainCards(),
                            turn_state=gturn.TrainCardDealTurn(
                                count=self.box.face_up_train_cards,
                                to_player=None,
                                next_turn_state=self.turn_state.next_turn_state,
                            ),
                        )
                    else:
                        return replace(
                            self,
                            turn_state=self.turn_state.next_turn_state,
                            face_up_train_cards=new_face_up_cards,
                        )
                else:
                    old_hand = self.hands[self.turn_state.to_player]
                    return replace(
                        self,
                        turn_state=self.turn_state.next_turn_state,
                        train_card_pile_distribution=subtract_train_cards(
                            self.train_card_pile_distribution, action.cards
                        )[0],
                        hands=frozendict(
                            {
                                **self.hands,
                                self.turn_state.to_player: replace(
                                    old_hand,
                                    train_cards=merge_train_cards(
                                        old_hand.train_cards, action.cards
                                    ),
                                ),
                            }
                        ),
                    )
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.GameOverTurn):
            if isinstance(action, gaction.PassAction):
                return self
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.InitialTurn):
            if isinstance(action, gaction.InitialDealAction):
                return replace(
                    self,
                    hands=frozendict(
                        (
                            player,
                            replace(
                                old_hand,
                                unselected_destination_cards=action.destination_cards[
                                    player
                                ],
                                train_cards=action.train_cards[player],
                            ),
                        )
                        for player, old_hand in self.hands.items()
                    ),
                    destination_card_pile_distribution=self.destination_card_pile_distribution
                    - set().union(*action.destination_cards.values()),  # type: ignore
                    destination_card_pile_size=self.destination_card_pile_size
                    - sum(map(len, action.destination_cards.values())),
                    train_card_pile_distribution=subtract_train_cards(
                        self.train_card_pile_distribution,
                        merge_train_cards(*action.train_cards.values()),
                    )[0],
                    turn_state=gturn.PlayerInitialDestinationCardChoiceTurn(
                        self.box.players[0]
                    ),
                    face_up_train_cards=merge_train_cards(
                        self.face_up_train_cards, action.face_up_train_cards
                    ),
                )
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.RevealFinalDestinationCardsTurn):
            if isinstance(action, gaction.RevealFinalDestinationCardsAction):
                return replace(self, turn_state=gturn.GameOverTurn())
            else:
                raise unexpected_action_error
        else:
            assert_never(self.turn_state)

    def get_legal_actions(self) -> Generator[AbstractState.LegalAction, None, None]:
        def get_train_card_draw_actions(
            second: bool = False,
        ) -> Generator[KnownState.LegalAction, None, None]:
            for color, count in self.face_up_train_cards.items():
                if count > 0 and not (second and color is None):
                    yield KnownState.LegalAction(
                        gaction.TrainCardPickAction(True, color)
                    )
            yield KnownState.LegalAction(gaction.TrainCardPickAction(False, None))

        def get_build_actions(
            player: Player,
        ) -> Generator[KnownState.LegalAction, None, None]:
            hand = self.hands[player]
            for route in self.box.board.routes:
                if (
                    route not in self.built_routes
                    and route.length <= hand.remaining_trains
                    and (
                        (
                            len(self.box.players)
                            >= self.box.double_routes_player_minimum
                            and not any(
                                self.built_routes.get(double, None) == player
                                for double in self.box.board.double_routes[route]
                            )
                        )
                        or not any(
                            double in self.built_routes
                            for double in self.box.board.double_routes[route]
                        )
                    )
                ):
                    if route.color is None:
                        colors: Iterable[Color] = self.box.colors
                    else:
                        colors = [route.color]

                    for color in colors:
                        max_color_cards = min(hand.train_cards[color], route.length)
                        max_wildcards = min(hand.train_cards[None], route.length - 1)
                        for color_cards in range(
                            route.length - max_wildcards, max_color_cards + 1
                        ):
                            cards_to_build = TrainCards(
                                {color: color_cards, None: route.length - color_cards}
                            )
                            yield KnownState.LegalAction(
                                gaction.BuildAction(route, cards_to_build),
                                probability=probability_of_having_cards(
                                    cards_to_build,
                                    0,
                                    self.train_card_pile_distribution,
                                ),
                            )
                    if hand.train_cards[None] >= route.length:
                        cards_to_build = TrainCards({None: route.length})
                        yield KnownState.LegalAction(
                            gaction.BuildAction(route, cards_to_build),
                            probability=probability_of_having_cards(
                                cards_to_build,
                                0,
                                self.train_card_pile_distribution,
                            ),
                        )

        def get_train_card_deal_actions(
            count: int, distribution: TrainCards
        ) -> Generator[KnownState.LegalAction, None, None]:
            for train_cards, prob in self._deal_train_cards(count, distribution):
                yield KnownState.LegalAction(
                    gaction.TrainCardDealAction(train_cards),
                    probability=prob,
                )

        def get_final_destination_cards(
            revealed_so_far: Optional[Cons[Tuple[Player, FrozenSet[DestinationCard]]]],
            remaining_counts: Optional[Cons[Tuple[Player, int]]],
            pile: FrozenSet[DestinationCard],
        ) -> Generator[Dict[Player, FrozenSet[DestinationCard]], None, None]:
            if remaining_counts is None:
                yield dict(Cons.iterate(revealed_so_far))
            else:
                player, count = remaining_counts.head
                for cards in itertools.combinations(pile, count):
                    cards_set = frozenset(cards)
                    yield from get_final_destination_cards(
                        Cons((player, cards_set), revealed_so_far),
                        remaining_counts.rest,
                        pile=cards_set,
                    )

        if isinstance(self.turn_state, gturn.InitialTurn):
            return  # TODO: probably unnecessary to implement
        elif isinstance(
            self.turn_state, gturn.PlayerInitialDestinationCardChoiceTurn
        ) or isinstance(self.turn_state, gturn.PlayerDestinationCardDrawMidTurn):
            legal_cards_range = (
                self.box.starting_destination_cards_range
                if isinstance(
                    self.turn_state, gturn.PlayerInitialDestinationCardChoiceTurn
                )
                else self.box.dealt_destination_cards_range
            )
            hand = self.hands[self.turn_state.player]
            allowed_card_numbers = list(
                range(
                    legal_cards_range[0],
                    len(hand.unselected_destination_cards) + 1,
                )
            )
            for kept_cards in allowed_card_numbers:
                for card_selections in itertools.combinations(
                    hand.unselected_destination_cards, kept_cards
                ):
                    yield KnownState.LegalAction(
                        gaction.DestinationCardSelectionAction(
                            frozenset(card_selections)
                        )
                    )
        elif isinstance(self.turn_state, gturn.PlayerStartTurn):
            if self.destination_card_pile_size > 0:
                yield KnownState.LegalAction(gaction.DestinationCardPickAction())
            yield from get_train_card_draw_actions()
            yield KnownState.LegalAction(gaction.PassAction())
            yield from get_build_actions(self.turn_state.player)
        elif isinstance(self.turn_state, gturn.PlayerTrainCardDrawMidTurn):
            yield from get_train_card_draw_actions(second=True)
        elif isinstance(self.turn_state, gturn.DestinationCardDealTurn):
            cards = min(
                self.box.dealt_destination_cards_range[1],
                self.destination_card_pile_size,
            )
            prob = 1 / math.comb(len(self.destination_card_pile_distribution), cards)

            for dealt_cards in itertools.combinations(
                self.destination_card_pile_distribution, cards
            ):
                yield KnownState.LegalAction(
                    gaction.DestinationCardDealAction(frozenset(dealt_cards)),
                    probability=prob,
                )
        elif isinstance(self.turn_state, gturn.TrainCardDealTurn):
            yield from get_train_card_deal_actions(
                self.turn_state.count, self.train_card_pile_distribution
            )
        elif isinstance(self.turn_state, gturn.GameOverTurn):
            yield KnownState.LegalAction(gaction.PassAction())
        elif isinstance(self.turn_state, gturn.RevealFinalDestinationCardsTurn):
            yield KnownState.LegalAction(
                gaction.RevealFinalDestinationCardsAction(
                    frozendict(
                        {
                            player: hand.destination_cards
                            for player, hand in self.hands.items()
                        }
                    )
                )
            )
        else:
            assert_never(self.turn_state)

    def assumed_hands(
        self: State,
        player: Player,
        route_building_probability_calculator: Callable[
            [Optional[FrozenSet[DestinationCard]]], float
        ] = lambda _: 1,
    ) -> Generator[Tuple[State, float], None, None]:
        yield self, 1

    def hand_is_known(self, player: Player) -> bool:
        return True
