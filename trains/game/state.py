from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import List, FrozenSet, Dict, Union, Generator, Iterable, Tuple

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
)
from trains.game.turn import TurnState
from trains.mypy_util import assert_never
from trains.util import (
    subtract_train_cards,
    merge_train_cards,
    probability_of_having_cards,
)


@dataclass(frozen=True)
class ObservedHandState:
    destination_cards_count: int
    unselected_destination_cards: int
    known_train_cards: TrainCards
    unknown_train_cards: int
    remaining_trains: int


@dataclass(frozen=True)
class KnownHandState:
    destination_cards: FrozenSet[DestinationCard]
    unselected_destination_cards: FrozenSet[DestinationCard]
    train_cards: TrainCards
    remaining_trains: int


@dataclass(frozen=True)
class State:
    box: Box
    player: Player  # the player for this state's perspective
    hand: KnownHandState
    opponent_hands: Dict[Player, ObservedHandState]
    discarded_train_cards: TrainCards
    face_up_train_cards: TrainCards
    train_card_pile_distribution: TrainCards
    destination_card_pile_size: int
    built_routes: Dict[Route, Player]
    turn_state: TurnState

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
            ),
            opponent_hands={
                p: ObservedHandState(
                    destination_cards_count=0,
                    unselected_destination_cards=0,
                    known_train_cards=TrainCards(),
                    unknown_train_cards=0,
                    remaining_trains=box.starting_train_count,
                )
                for p in box.players
                if p != player
            },
            discarded_train_cards=TrainCards(),
            face_up_train_cards=TrainCards(),
            train_card_pile_distribution=box.train_cards,
            destination_card_pile_size=len(box.destination_cards),
            built_routes={},
            turn_state=gturn.InitialTurn(),
        )

    def next_state(self, action: Action) -> State:
        unexpected_action_error = TrainsException(
            f"unexpected action type {type(action)}"
        )

        def perform_train_draw(
            turn_state: Union[gturn.PlayerStartTurn, gturn.PlayerTrainCardDrawMidTurn],
            action: gaction.TrainCardPickAction,
            second_draw: bool = False,
        ) -> State:
            if action.draw_known:
                next_turn_state = gturn.TrainCardDealTurn(
                    1,
                    None,
                    gturn.PlayerStartTurn(
                        turn_state.last_turn_forced,
                        self.box.next_player_map[turn_state.player],
                    )
                    if second_draw or action.selected_card_if_known is None
                    else gturn.PlayerTrainCardDrawMidTurn(
                        turn_state.last_turn_forced,
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
                        gturn.PlayerStartTurn(
                            turn_state.last_turn_forced,
                            self.box.next_player_map[turn_state.player],
                        )
                        if second_draw
                        else gturn.PlayerTrainCardDrawMidTurn(
                            turn_state.last_turn_forced, turn_state.player
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
            if isinstance(action, gaction.PassAction):
                return replace(
                    self,
                    turn_state=gturn.PlayerStartTurn(
                        self.turn_state.last_turn_forced,
                        self.box.next_player_map[self.turn_state.player],
                    ),
                )
            elif isinstance(action, gaction.BuildAction):
                if self.turn_state.player == self.player:
                    last_turn_forced = self.turn_state.last_turn_forced
                    if (
                        last_turn_forced is None
                        and self.hand.remaining_trains - action.route.length
                        <= self.box.trains_to_end
                    ):
                        last_turn_forced = self.turn_state.player
                    return replace(
                        self,
                        turn_state=gturn.PlayerStartTurn(
                            last_turn_forced,
                            self.box.next_player_map[self.turn_state.player],
                        ),
                        built_routes={
                            **self.built_routes,
                            action.route: self.turn_state.player,
                        },
                        hand=replace(
                            self.hand,
                            train_cards=subtract_train_cards(
                                self.hand.train_cards, action.train_cards
                            )[0],
                            remaining_trains=self.hand.remaining_trains
                            - action.route.length,
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
                    last_turn_forced = self.turn_state.last_turn_forced
                    if (
                        last_turn_forced is None
                        and old_hand.remaining_trains - action.route.length
                        <= self.box.trains_to_end
                    ):
                        last_turn_forced = self.turn_state.player
                    return replace(
                        self,
                        turn_state=gturn.PlayerStartTurn(
                            last_turn_forced,
                            self.box.next_player_map[self.turn_state.player],
                        ),
                        built_routes={
                            **self.built_routes,
                            action.route: self.turn_state.player,
                        },
                        opponent_hands={
                            **self.opponent_hands,
                            self.turn_state.player: replace(
                                old_hand,
                                known_train_cards=new_known_train_cards,
                                unknown_train_cards=old_hand.unknown_train_cards
                                - leftovers.total,
                                remaining_trains=old_hand.remaining_trains
                                - action.route.length,
                            ),
                        },
                        discarded_train_cards=merge_train_cards(
                            self.discarded_train_cards, action.train_cards
                        ),
                    )
            elif isinstance(action, gaction.TrainCardPickAction):
                return perform_train_draw(self.turn_state, action)
            elif isinstance(action, gaction.DestinationCardPickAction):
                return replace(
                    self,
                    turn_state=gturn.DestinationCardDealTurn(
                        self.turn_state.last_turn_forced, self.turn_state.player
                    ),
                )
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.PlayerTrainCardDrawMidTurn):
            if isinstance(action, gaction.TrainCardPickAction):
                return perform_train_draw(self.turn_state, action, True)
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.PlayerDestinationCardDrawMidTurn):
            if isinstance(action, gaction.DestinationCardSelectionAction):
                next_turn_state: TurnState = gturn.PlayerStartTurn(
                    self.turn_state.last_turn_forced,
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
                        ),
                        turn_state=next_turn_state,
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
                                unselected_destination_cards=0,
                                destination_cards_count=old_hand.destination_cards_count
                                + len(action.selected_cards),
                            ),
                        },
                        turn_state=next_turn_state,
                        destination_card_pile_size=self.destination_card_pile_size
                        + old_hand.unselected_destination_cards
                        - len(action.selected_cards),
                    )
            else:
                raise unexpected_action_error
        elif isinstance(self.turn_state, gturn.DestinationCardDealTurn):
            if isinstance(action, gaction.DestinationCardDealAction):
                next_turn_state = gturn.PlayerDestinationCardDrawMidTurn(
                    self.turn_state.last_turn_forced, self.turn_state.to_player
                )
                if self.turn_state.to_player == self.player:
                    return replace(
                        self,
                        turn_state=next_turn_state,
                        hand=replace(
                            self.hand, unselected_destination_cards=action.cards
                        ),
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
                                unselected_destination_cards=len(action.cards),
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
                                    unknown_train_cards=old_hand.unknown_train_cards
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
                    turn_state=gturn.PlayerStartTurn(None, self.box.players[0]),
                    opponent_hands={
                        player: replace(
                            hand,
                            unselected_destination_cards=0,
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
                            unselected_destination_cards=len(
                                action.destination_cards[player]
                            ),
                            unknown_train_cards=len(action.train_cards[player]),
                        )
                        for player, hand in self.opponent_hands.items()
                    },
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
        def get_train_card_draw_actions() -> Generator[State.LegalAction, None, None]:
            for color in self.face_up_train_cards:
                yield State.LegalAction(gaction.TrainCardPickAction(True, color))
            yield State.LegalAction(gaction.TrainCardPickAction(False, None))

        def get_build_actions(
            known_cards: TrainCards, unknown_cards: int
        ) -> Generator[State.LegalAction, None, None]:
            for route in self.box.board.routes:
                if route not in self.built_routes:
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
            def _inner(
                count: int,
                distribution: List[Tuple[TrainCard, int]],
                distribution_count: int,
            ) -> Generator[Dict[TrainCard, int], None, None]:
                color, available = distribution[-1]
                if len(distribution) == 1:
                    yield {color: count}
                else:
                    remaining_distribution_count = distribution_count - available
                    for drawn in range(
                        max(0, count - remaining_distribution_count), available + 1
                    ):
                        for sub_draw in _inner(
                            count - drawn,
                            distribution[:-1],
                            remaining_distribution_count,
                        ):
                            draw = sub_draw
                            draw[color] = drawn
                            yield draw

            for cards in _inner(count, list(distribution.items()), distribution.total):
                train_cards = TrainCards(cards)
                yield State.LegalAction(
                    gaction.TrainCardDealAction(train_cards),
                    probability=probability_of_having_cards(
                        train_cards,
                        train_cards.total,
                        self.train_card_pile_distribution,
                    ),
                )

        if isinstance(self.turn_state, gturn.InitialTurn):
            return  # TODO: probably unnecessary to implement
        elif isinstance(
            self.turn_state, gturn.PlayerInitialDestinationCardChoiceTurn
        ) or isinstance(self.turn_state, gturn.PlayerDestinationCardDrawMidTurn):
            if self.player == self.turn_state.player:
                allowed_card_numbers = list(
                    range(
                        self.box.starting_destination_cards_range[0],
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
                        self.box.starting_destination_cards_range[0],
                        self.opponent_hands[
                            self.turn_state.player
                        ].unselected_destination_cards
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
                    self.opponent_hands[self.turn_state.player].unknown_train_cards,
                )
        elif isinstance(self.turn_state, gturn.PlayerTrainCardDrawMidTurn):
            yield from get_train_card_draw_actions()
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
                        self.opponent_hands[opponent].unselected_destination_cards + 1,
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
        else:
            assert_never(self.turn_state)
