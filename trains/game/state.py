from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import (
    FrozenSet,
    Dict,
    Union,
    Generator,
    Iterable,
    Tuple,
    Optional,
    Callable,
)

import trains.game.action as gaction
import trains.game.turn as gturn
from trains.error import TrainsException
from trains.game.action import Action
from trains.game.box import (
    DestinationCard,
    Player,
    Route,
    TrainCards,
    Box,
    Color,
    TrainCard,
    City,
)
from trains.game.clusters import Clusters
from trains.game.turn import TurnState
from trains.mypy_util import assert_never
from trains.util import (
    subtract_train_cards,
    merge_train_cards,
    probability_of_having_cards,
    Cons,
)


@dataclass(frozen=True)
class ObservedHandState:
    known_destination_cards: FrozenSet[DestinationCard]
    destination_cards_count: int
    known_unselected_destination_cards: FrozenSet[DestinationCard]
    unselected_destination_cards_count: int
    known_train_cards: TrainCards
    train_cards_count: int
    remaining_trains: int

    known_points_so_far: int
    known_complete_destination_cards: FrozenSet[DestinationCard]
    known_incomplete_destination_cards: FrozenSet[DestinationCard]


@dataclass(frozen=True)
class KnownHandState:
    destination_cards: FrozenSet[DestinationCard]
    unselected_destination_cards: FrozenSet[DestinationCard]
    train_cards: TrainCards
    remaining_trains: int

    points_so_far: int
    complete_destination_cards: FrozenSet[DestinationCard]
    incomplete_destination_cards: FrozenSet[DestinationCard]


@dataclass(frozen=True)
class State:
    """
    A game state, from the perspective of a certain player. Due to this, there is not
    full knowledge of opponent's hands.

    Args:
        box: the box for the game
        player: the state is in the perspective of this player
        hand: the hand of the player
        opponent_hands: the hands of the player's opponents
        discarded_train_cards: the train card discard pile
        face_up_train_cards: the face up train cards
        train_card_pile_distribution: train cards from the deck that have not been seen
        destination_card_pile_size: the number of remaining undrawn destination cards
        built_routes: routes that have been built and the player they were built by
        turn_state: the current turn state of the game
        revealed_destination_cards: if at the end of the game, the destination cards
            that each player has
    """

    box: Box
    player: Player  # the player for this state's perspective
    hand: KnownHandState
    opponent_hands: Dict[Player, ObservedHandState]
    discarded_train_cards: TrainCards
    face_up_train_cards: TrainCards
    train_card_pile_distribution: TrainCards
    destination_card_pile_distribution: FrozenSet[DestinationCard]
    destination_card_pile_size: int
    built_routes: Dict[Route, Player]
    built_clusters: Dict[Player, Clusters[City]]
    turn_state: TurnState
    revealed_destination_cards: Optional[Dict[Player, FrozenSet[DestinationCard]]]

    @classmethod
    def make(cls, box: Box, player: Player) -> State:
        return State(
            box=box,
            player=player,
            hand=KnownHandState(
                destination_cards=frozenset(),
                unselected_destination_cards=frozenset(),
                train_cards=TrainCards(),
                remaining_trains=box.starting_train_count,
                points_so_far=box.starting_score,
                complete_destination_cards=frozenset(),
                incomplete_destination_cards=frozenset(),
            ),
            opponent_hands={
                p: ObservedHandState(
                    known_destination_cards=frozenset(),
                    destination_cards_count=0,
                    known_unselected_destination_cards=frozenset(),
                    unselected_destination_cards_count=0,
                    known_train_cards=TrainCards(),
                    train_cards_count=0,
                    remaining_trains=box.starting_train_count,
                    known_points_so_far=box.starting_score,
                    known_complete_destination_cards=frozenset(),
                    known_incomplete_destination_cards=frozenset(),
                )
                for p in box.players
                if p != player
            },
            discarded_train_cards=TrainCards(),
            face_up_train_cards=TrainCards(),
            train_card_pile_distribution=box.train_cards,
            destination_card_pile_distribution=box.destination_cards,
            destination_card_pile_size=len(box.destination_cards),
            built_routes={},
            built_clusters={player: Clusters() for player in box.players},
            turn_state=gturn.InitialTurn(),
            revealed_destination_cards=None,
        )

    def next_state(self, action: Action) -> State:
        unexpected_action_error = TrainsException(
            f"unexpected action type {type(action)}"
        )

        def perform_train_draw(
            turn_state: Union[gturn.PlayerStartTurn, gturn.PlayerTrainCardDrawMidTurn],
            action: gaction.TrainCardPickAction,
            last_turn_started: bool,
            second_draw: bool = False,
        ) -> State:
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
                if turn_state.player == self.player:
                    return replace(
                        self,
                        turn_state=next_turn_state,
                        hand=replace(
                            self.hand,
                            train_cards=self.hand.train_cards.incrementing(
                                action.selected_card_if_known, 1
                            ),
                        ),
                        face_up_train_cards=self.face_up_train_cards.incrementing(
                            action.selected_card_if_known, -1
                        ),
                    )
                else:
                    return replace(
                        self,
                        turn_state=next_turn_state,
                        opponent_hands={
                            **self.opponent_hands,
                            turn_state.player: replace(
                                self.opponent_hands[turn_state.player],
                                known_train_cards=self.opponent_hands[
                                    turn_state.player
                                ].known_train_cards.incrementing(
                                    action.selected_card_if_known, 1
                                ),
                            ),
                        },
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
                    next_turn: TurnState = (
                        gturn.RevealInitialDestinationCardChoicesTurn()
                    )
                else:
                    next_turn = gturn.PlayerInitialDestinationCardChoiceTurn(
                        next_player
                    )
                if self.turn_state.player == self.player:
                    return replace(
                        self,
                        turn_state=next_turn,
                        hand=replace(
                            self.hand,
                            unselected_destination_cards=frozenset(),
                            destination_cards=action.selected_cards,
                            incomplete_destination_cards=action.selected_cards,
                        ),
                        destination_card_pile_distribution=self.destination_card_pile_distribution
                        | (
                            self.hand.unselected_destination_cards
                            - action.selected_cards
                        ),
                        destination_card_pile_size=self.destination_card_pile_size
                        + len(self.hand.unselected_destination_cards)
                        - len(action.selected_cards),
                    )
                else:
                    return replace(self, turn_state=next_turn)
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.PlayerStartTurn):
            remaining_trains = (
                self.hand.remaining_trains
                if self.turn_state.player == self.player
                else self.opponent_hands[self.turn_state.player].remaining_trains
            )
            last_turn_started = (
                self.turn_state.last_turn_started
                or remaining_trains <= self.box.trains_to_end
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
                    action.route.cities
                )
                if self.turn_state.player == self.player:
                    completed_destination_cards = {
                        card
                        for card in self.hand.incomplete_destination_cards
                        if new_cluster.is_connected(card.cities)
                    }
                    return replace(
                        self,
                        turn_state=gturn.PlayerStartTurn.make_or_end(
                            last_turn_started,
                            self.box.next_player_map[self.turn_state.player],
                        ),
                        built_routes={
                            **self.built_routes,
                            action.route: self.turn_state.player,
                        },
                        built_clusters={
                            **self.built_clusters,
                            self.turn_state.player: new_cluster,
                        },
                        hand=replace(
                            self.hand,
                            train_cards=subtract_train_cards(
                                self.hand.train_cards, action.train_cards
                            )[0],
                            remaining_trains=self.hand.remaining_trains
                            - action.route.length,
                            points_so_far=self.hand.points_so_far
                            + self.box.route_point_values[action.route.length]
                            + sum(card.value for card in completed_destination_cards),
                            complete_destination_cards=self.hand.complete_destination_cards
                            | completed_destination_cards,
                            incomplete_destination_cards=self.hand.incomplete_destination_cards
                            - completed_destination_cards,
                        ),
                        discarded_train_cards=merge_train_cards(
                            self.discarded_train_cards, action.train_cards
                        ),
                    )
                else:
                    old_hand = self.opponent_hands[self.turn_state.player]
                    new_known_train_cards, leftovers = subtract_train_cards(
                        old_hand.known_train_cards, action.train_cards
                    )
                    completed_destination_cards = {
                        card
                        for card in old_hand.known_incomplete_destination_cards
                        if new_cluster.is_connected(card.cities)
                    }
                    return replace(
                        self,
                        turn_state=gturn.PlayerStartTurn.make_or_end(
                            last_turn_started,
                            self.box.next_player_map[self.turn_state.player],
                        ),
                        built_routes={
                            **self.built_routes,
                            action.route: self.turn_state.player,
                        },
                        built_clusters={
                            **self.built_clusters,
                            self.turn_state.player: new_cluster,
                        },
                        opponent_hands={
                            **self.opponent_hands,
                            self.turn_state.player: replace(
                                old_hand,
                                known_train_cards=new_known_train_cards,
                                train_cards_count=old_hand.train_cards_count
                                - action.route.length,
                                remaining_trains=old_hand.remaining_trains
                                - action.route.length,
                                known_points_so_far=old_hand.known_points_so_far
                                + self.box.route_point_values[action.route.length]
                                + sum(
                                    card.value for card in completed_destination_cards
                                ),
                                known_complete_destination_cards=old_hand.known_complete_destination_cards
                                | completed_destination_cards,
                                known_incomplete_destination_cards=old_hand.known_incomplete_destination_cards
                                - completed_destination_cards,
                            ),
                        },
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
                if self.turn_state.player == self.player:
                    return replace(
                        self,
                        hand=replace(
                            self.hand,
                            unselected_destination_cards=frozenset(),
                            destination_cards=self.hand.destination_cards
                            | action.selected_cards,
                            incomplete_destination_cards=self.hand.incomplete_destination_cards
                            | action.selected_cards,
                        ),
                        turn_state=next_turn_state,
                        destination_card_pile_distribution=self.destination_card_pile_distribution
                        | (
                            self.hand.unselected_destination_cards
                            - action.selected_cards
                        ),
                        destination_card_pile_size=self.destination_card_pile_size
                        + len(self.hand.unselected_destination_cards)
                        - len(action.selected_cards),
                    )
                else:
                    old_hand = self.opponent_hands[self.turn_state.player]
                    return replace(
                        self,
                        opponent_hands={
                            **self.opponent_hands,
                            self.turn_state.player: replace(
                                old_hand,
                                unselected_destination_cards_count=0,
                                destination_cards_count=old_hand.destination_cards_count
                                + len(action.selected_cards),
                            ),
                        },
                        turn_state=next_turn_state,
                        destination_card_pile_size=self.destination_card_pile_size
                        + old_hand.unselected_destination_cards_count
                        - len(action.selected_cards),
                    )
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.DestinationCardDealTurn):
            if isinstance(action, gaction.DestinationCardDealAction):
                next_turn_state = gturn.PlayerDestinationCardDrawMidTurn(
                    self.turn_state.last_turn_started, self.turn_state.to_player
                )
                if self.turn_state.to_player == self.player:
                    return replace(
                        self,
                        turn_state=next_turn_state,
                        hand=replace(
                            self.hand, unselected_destination_cards=action.cards
                        ),
                        destination_card_pile_distribution=self.destination_card_pile_distribution
                        - action.cards,
                        destination_card_pile_size=self.destination_card_pile_size
                        - len(action.cards),
                    )
                else:
                    return replace(
                        self,
                        turn_state=next_turn_state,
                        opponent_hands={
                            **self.opponent_hands,
                            self.turn_state.to_player: replace(
                                self.opponent_hands[self.turn_state.to_player],
                                unselected_destination_cards_count=len(action.cards),
                            ),
                        },
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
                    if self.turn_state.to_player == self.player:
                        return replace(
                            self,
                            turn_state=self.turn_state.next_turn_state,
                            hand=replace(
                                self.hand,
                                train_cards=merge_train_cards(
                                    self.hand.train_cards, action.cards
                                ),
                            ),
                        )
                    else:
                        old_hand = self.opponent_hands[self.turn_state.to_player]
                        return replace(
                            self,
                            turn_state=self.turn_state.next_turn_state,
                            opponent_hands={
                                **self.opponent_hands,
                                self.turn_state.to_player: replace(
                                    old_hand,
                                    train_cards_count=old_hand.train_cards_count
                                    + action.cards.total,
                                ),
                            },
                        )
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.RevealInitialDestinationCardChoicesTurn):
            if isinstance(action, gaction.RevealDestinationCardSelectionsAction):
                return replace(
                    self,
                    turn_state=gturn.PlayerStartTurn.make_or_end(
                        False, self.box.players[0]
                    ),
                    opponent_hands={
                        player: replace(
                            hand,
                            unselected_destination_cards_count=0,
                            destination_cards_count=action.kept_destination_cards[
                                player
                            ],
                        )
                        for player, hand in self.opponent_hands.items()
                    },
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
                    hand=replace(
                        self.hand,
                        unselected_destination_cards=action.destination_cards[
                            self.player
                        ],
                        train_cards=action.train_cards[self.player],
                    ),
                    opponent_hands={
                        player: replace(
                            hand,
                            unselected_destination_cards_count=len(
                                action.destination_cards[player]
                            ),
                            train_cards_count=len(action.train_cards[player]),
                        )
                        for player, hand in self.opponent_hands.items()
                    },
                    destination_card_pile_distribution=self.destination_card_pile_distribution
                    - action.destination_cards[self.player],
                    destination_card_pile_size=self.destination_card_pile_size
                    - sum(map(len, action.destination_cards.values())),
                    train_card_pile_distribution=subtract_train_cards(
                        self.train_card_pile_distribution,
                        action.train_cards[self.player],
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
                return replace(
                    self, revealed_destination_cards=action.destination_cards
                )
            else:
                raise unexpected_action_error
        else:
            assert_never(self.turn_state)

    @dataclass(frozen=True)
    class LegalAction:
        """
        Represents a possible legal action to take. The probability is the probability
        of the actor having the right cards for the move.
        """

        action: Action
        probability: float = 1

    def get_legal_actions(self) -> Generator[LegalAction, None, None]:
        def get_train_card_draw_actions(
            second: bool = False,
        ) -> Generator[State.LegalAction, None, None]:
            for color, count in self.face_up_train_cards.items():
                if count > 0 and not (second and color is None):
                    yield State.LegalAction(gaction.TrainCardPickAction(True, color))
            yield State.LegalAction(gaction.TrainCardPickAction(False, None))

        def get_build_actions(
            known_cards: TrainCards, unknown_cards: int
        ) -> Generator[State.LegalAction, None, None]:
            for route in self.box.board.routes:
                if (
                    route not in self.built_routes
                    and route.length <= self.hand.remaining_trains
                    and (
                        (
                            len(self.box.players)
                            >= self.box.double_routes_player_minimum
                            and not any(
                                self.built_routes.get(double, None) == self.player
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
                        max_color_cards = min(
                            known_cards[color] + unknown_cards, route.length
                        )
                        max_wildcards = min(
                            known_cards[None] + unknown_cards, route.length - 1
                        )
                        for color_cards in range(
                            route.length - max_wildcards, max_color_cards + 1
                        ):
                            cards_to_build = TrainCards(
                                {color: color_cards, None: route.length - color_cards}
                            )
                            yield State.LegalAction(
                                gaction.BuildAction(route, cards_to_build),
                                probability=probability_of_having_cards(
                                    cards_to_build,
                                    unknown_cards,
                                    self.train_card_pile_distribution,
                                ),
                            )
                    if known_cards[None] + unknown_cards >= route.length:
                        cards_to_build = TrainCards({None: route.length})
                        yield State.LegalAction(
                            gaction.BuildAction(route, cards_to_build),
                            probability=probability_of_having_cards(
                                cards_to_build,
                                unknown_cards,
                                self.train_card_pile_distribution,
                            ),
                        )

        def get_train_card_deal_actions(
            count: int, distribution: TrainCards
        ) -> Generator[State.LegalAction, None, None]:
            for train_cards, prob in self._deal_train_cards(count, distribution):
                yield State.LegalAction(
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
            if self.player == self.turn_state.player:
                allowed_card_numbers = list(
                    range(
                        legal_cards_range[0],
                        len(self.hand.unselected_destination_cards) + 1,
                    )
                )
                for kept_cards in allowed_card_numbers:
                    for card_selections in itertools.combinations(
                        self.hand.unselected_destination_cards, kept_cards
                    ):
                        yield State.LegalAction(
                            gaction.DestinationCardSelectionAction(
                                frozenset(card_selections)
                            )
                        )
            else:
                # although the action can have many selections of destination cards,
                # the number of selected ones is the only thing that matters, so
                # we can select arbitrary destination cards
                allowed_card_numbers = list(
                    range(
                        legal_cards_range[0],
                        self.opponent_hands[
                            self.turn_state.player
                        ].unselected_destination_cards_count
                        + 1,
                    )
                )
                for kept_cards in allowed_card_numbers:
                    yield State.LegalAction(
                        gaction.DestinationCardSelectionAction(
                            frozenset(
                                itertools.islice(self.box.destination_cards, kept_cards)
                            )
                        )
                    )
        elif isinstance(self.turn_state, gturn.PlayerStartTurn):
            if self.destination_card_pile_size > 0:
                yield State.LegalAction(gaction.DestinationCardPickAction())
            yield from get_train_card_draw_actions()
            yield State.LegalAction(gaction.PassAction())
            if self.player == self.turn_state.player:
                yield from get_build_actions(self.hand.train_cards, 0)
            else:
                yield from get_build_actions(
                    self.opponent_hands[self.turn_state.player].known_train_cards,
                    self.opponent_hands[self.turn_state.player].train_cards_count
                    - self.opponent_hands[
                        self.turn_state.player
                    ].known_train_cards.total,
                )
        elif isinstance(self.turn_state, gturn.PlayerTrainCardDrawMidTurn):
            yield from get_train_card_draw_actions(second=True)
        elif isinstance(self.turn_state, gturn.DestinationCardDealTurn):
            cards = min(
                self.box.dealt_destination_cards_range[1],
                self.destination_card_pile_size,
            )
            deck = self.box.destination_cards - self.hand.destination_cards
            prob = 1 / math.comb(len(deck), cards)

            for dealt_cards in itertools.combinations(deck, cards):
                yield State.LegalAction(
                    gaction.DestinationCardDealAction(frozenset(dealt_cards)),
                    probability=prob,
                )
        elif isinstance(self.turn_state, gturn.TrainCardDealTurn):
            yield from get_train_card_deal_actions(
                self.turn_state.count, self.train_card_pile_distribution
            )
        elif isinstance(self.turn_state, gturn.RevealInitialDestinationCardChoicesTurn):
            opponents = [player for player in self.box.players if player != self.player]
            selection_options = [
                list(
                    range(
                        self.box.starting_destination_cards_range[0],
                        self.opponent_hands[opponent].unselected_destination_cards_count
                        + 1,
                    )
                )
                for opponent in opponents
            ]
            for selections in itertools.product(*selection_options):
                selections_map = dict(zip(opponents, selections))
                yield State.LegalAction(
                    gaction.RevealDestinationCardSelectionsAction(
                        selections_map | {self.player: len(self.hand.destination_cards)}
                    ),
                )
        elif isinstance(self.turn_state, gturn.GameOverTurn):
            yield State.LegalAction(gaction.PassAction())
        elif isinstance(self.turn_state, gturn.RevealFinalDestinationCardsTurn):
            prob = 1
            pile_size = len(self.box.destination_cards) - len(
                self.hand.destination_cards
            )
            for player, hand in self.opponent_hands.items():
                prob /= math.comb(
                    pile_size,
                    hand.destination_cards_count - len(hand.known_destination_cards),
                )
                pile_size -= hand.destination_cards_count

            for destination_card_set in get_final_destination_cards(
                Cons((self.player, self.hand.destination_cards), None),
                Cons.make(
                    (
                        player,
                        hand.destination_cards_count
                        - len(hand.known_destination_cards),
                    )
                    for player, hand in self.opponent_hands.items()
                ),
                self.destination_card_pile_distribution,
            ):
                for opponent, hand in self.opponent_hands.items():
                    destination_card_set[opponent] |= hand.known_destination_cards
                yield State.LegalAction(
                    gaction.RevealFinalDestinationCardsAction(destination_card_set),
                    probability=prob,
                )
        else:
            assert_never(self.turn_state)

    def assumed_hands(
        self,
        player: Player,
        route_building_probability_calculator: Callable[
            [Optional[FrozenSet[DestinationCard]]], float
        ] = lambda _: 1,
    ) -> Generator[Tuple[State, float], None, None]:
        """
        Generate a list of all possible states where the given player's hand is fully
        known, together with the probability of the player having the hand.

        Args:
            player: The player whose hand is being assumed
            route_building_probability_calculator: A function that given a set of
                destination cards returns the probability that the player would build
                the routes that they have built given those destination cards. If the
                input is None, the it returns the probability of building the routes
                disregarding destination cards
        """
        if player in self.opponent_hands:
            hand = self.opponent_hands[player]
            for train_cards, train_cards_prob in self._deal_train_cards(
                hand.train_cards_count - len(hand.known_train_cards),
                self.train_card_pile_distribution,
            ):
                for (
                    destination_cards,
                    destination_cards_prob,
                ) in self._deal_destination_cards(
                    hand.destination_cards_count - len(hand.known_destination_cards),
                    self.destination_card_pile_distribution,
                ):
                    destination_card_pile_after_destination_card_deal = (
                        self.destination_card_pile_distribution - destination_cards
                    )
                    # we can re-calculate destination_cards_prob to consider the routes
                    # that the player has built so far to make it more accurate

                    # Let D = drawing a certain set of destination cards
                    # and R = building a certain set of routes
                    # By bayes:
                    # P(D|R) = P(R|D) * P(D) / P(R)
                    # P(R|D) is route_building_probability_calculator(destination_cards)
                    # P(R) is route_building_probability_calculator(None)
                    # P(D) is destination_cards_prob

                    weighted_destination_cards_prob = (
                        route_building_probability_calculator(destination_cards)
                        * destination_cards_prob
                        / route_building_probability_calculator(None)
                    )
                    for (
                        unselected_destination_cards,
                        unselected_destination_cards_prob,
                    ) in self._deal_destination_cards(
                        hand.unselected_destination_cards_count
                        - len(hand.known_unselected_destination_cards),
                        destination_card_pile_after_destination_card_deal,
                    ):
                        yield replace(
                            self,
                            train_card_pile_distribution=subtract_train_cards(
                                self.train_card_pile_distribution, train_cards
                            )[0],
                            destination_card_pile_distribution=destination_card_pile_after_destination_card_deal
                            - unselected_destination_cards,
                            opponent_hands={
                                **self.opponent_hands,
                                player: replace(
                                    hand,
                                    known_train_cards=merge_train_cards(
                                        hand.known_train_cards, train_cards
                                    ),
                                    known_destination_cards=hand.known_destination_cards
                                    | destination_cards,
                                    known_unselected_destination_cards=hand.known_unselected_destination_cards
                                    | unselected_destination_cards,
                                    known_complete_destination_cards=hand.known_complete_destination_cards
                                    | {
                                        card
                                        for card in destination_cards
                                        if self.built_clusters[player].is_connected(
                                            card.cities
                                        )
                                    },
                                    known_incomplete_destination_cards=hand.known_incomplete_destination_cards
                                    | {
                                        card
                                        for card in destination_cards
                                        if not self.built_clusters[player].is_connected(
                                            card.cities
                                        )
                                    },
                                ),
                            },
                        ), train_cards_prob * weighted_destination_cards_prob * unselected_destination_cards_prob
        else:
            yield self, 1

    def hand_is_known(self, player: Player) -> bool:
        """
        Determine if the state has full knowledge of the hand of the given player
        """
        if player in self.opponent_hands:
            hand = self.opponent_hands[player]
            return (
                len(hand.known_train_cards) == hand.train_cards_count
                and len(hand.known_destination_cards) == hand.destination_cards_count
                and len(hand.known_unselected_destination_cards)
                == hand.unselected_destination_cards_count
            )
        else:
            return True

    @classmethod
    def _deal_train_cards(
        cls, cards: int, deck: TrainCards
    ) -> Generator[Tuple[TrainCards, float], None, None]:
        def _deal_train_cards(
            remaining_cards: int,
            current_deck: Optional[Cons[Tuple[TrainCard, int]]],
            deal_so_far: Optional[Cons[Tuple[TrainCard, int]]] = None,
        ) -> Generator[TrainCards, None, None]:
            if remaining_cards == 0 or current_deck is None:
                yield TrainCards(Cons.iterate(deal_so_far))
            else:
                color, color_cards = current_deck.head
                max_count = max(color_cards, remaining_cards)
                for count in range(0, max_count + 1):
                    yield from _deal_train_cards(
                        remaining_cards - count,
                        current_deck.rest,
                        Cons((color, count), deal_so_far),
                    )

        for train_cards in _deal_train_cards(cards, Cons.make(deck.items())):
            yield train_cards, probability_of_having_cards(train_cards, cards, deck)

    @classmethod
    def _deal_destination_cards(
        cls, cards: int, destination_cards: FrozenSet[DestinationCard]
    ) -> Generator[Tuple[FrozenSet[DestinationCard], float], None, None]:
        prob = 1 / math.comb(len(destination_cards), cards)
        for result_cards in itertools.combinations(destination_cards, cards):
            yield frozenset(result_cards), prob
