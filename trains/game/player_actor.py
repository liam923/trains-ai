from __future__ import annotations

import argparse
import random
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Callable, FrozenSet, TypeVar, Type

import trains.game.action as gaction
import trains.game.turn as gturn
from trains.error import TrainsException, ParserException
from trains.game.action import Action
from trains.game.actor import Actor
from trains.game.box import Color, City, Box, Player, TrainCards
from trains.game.state import State


_PlayerActor = TypeVar("_PlayerActor", bound="PlayerActor")


@dataclass  # type: ignore
class PlayerActor(Actor, ABC):
    """
    An abstract actor that represents an actor for a player
    """

    state: State

    @classmethod
    def make(cls: Type[_PlayerActor], box: Box, player: Player) -> _PlayerActor:
        state = State.make(box, player)
        return cls(box=box, state=state, turn_state=state.turn_state)

    def validate_action(self, action: Action) -> Optional[str]:
        return None

    def observe_action(self, action: Action) -> None:
        self.state = self.state.next_state(action)
        self.turn_state = self.state.turn_state


class UserActor(PlayerActor):
    def get_action(self) -> Action:
        if isinstance(self.turn_state, gturn.PlayerTurn):
            print()
            self.print_state()
            print()
            while True:
                try:
                    return self.parse_action(input("Enter action: "))
                except ParserException as error:
                    print(f"Error parsing action: {error}")
        else:
            raise TrainsException(f"Unexpected turn state {type(self.turn_state)}")

    def print_state(self) -> None:
        colors = ", ".join(
            f"{count} {'wildcard' if color is None else color.name}"
            for color, count in self.state.hand.train_cards.items()
            if count > 0
        )
        print(f"Train cards in hand: {colors}")

        print(
            f"Destination cards: {', '.join(f'{card.cities_list[0].name}->{card.cities_list[1].name} for {card.value}' for card in self.state.hand.destination_cards)}"
        )

        if len(self.state.hand.unselected_destination_cards) > 0:
            print(
                f"Destination cards to select from: {', '.join(f'{card.cities_list[0].name}->{card.cities_list[1].name} for {card.value}' for card in self.state.hand.unselected_destination_cards)}"
            )

        colors = ", ".join(
            f"{count} {'wildcard' if color is None else color.name}"
            for color, count in self.state.face_up_train_cards.items()
            if count > 0
        )
        print(f"Face up train cards: {colors}")

    def observe_action(self, action: Action) -> None:
        if isinstance(self.turn_state, gturn.PlayerTurn):
            if isinstance(action, gaction.BuildAction):
                cities = list(action.route.cities)
                colors = " and ".join(
                    f"{count} {'wildcard' if color is None else color.name}"
                    for color, count in action.train_cards.items()
                    if count > 0
                )
                print(
                    f"{self.turn_state.player.name} built {'grey' if action.route.color is None else action.route.color.name} route from {cities[0].name} to {cities[0].name} built using {colors}"
                )
            elif isinstance(action, gaction.DestinationCardSelectionAction):
                print(
                    f"{self.turn_state.player.name} drew {len(action.selected_cards)} destination cards"
                )
            elif isinstance(action, gaction.TrainCardPickAction):
                if action.draw_known:
                    if action.selected_card_if_known is None:
                        print(f"{self.turn_state.player.name} drew a wildcard")
                    else:
                        print(
                            f"{self.turn_state.player.name} drew a {action.selected_card_if_known.name} card"
                        )
                else:
                    print(
                        f"{self.turn_state.player.name} drew a train card from the deck"
                    )
            elif isinstance(action, gaction.RevealDestinationCardSelectionsAction):
                for player, count in action.kept_destination_cards.items():
                    print(
                        f"{self.turn_state.player.name} kept {count} destination cards"
                    )
            elif isinstance(action, gaction.PassAction):
                print(f"{self.turn_state.player.name} chose to pass")

        super().observe_action(action)

    @property  # type: ignore
    def _parser(self) -> Callable[[str], Action]:
        def parse_color(color_str: str) -> Optional[Color]:
            if color_str == "wildcard":
                return None

            color = Color(color_str)
            if color not in self.box.colors:
                raise ParserException(f"Invalid color '{color_str}'")
            return color

        def parse_city(city_str: str) -> City:
            city = City(city_str)
            if city not in self.box.board.cities:
                raise ParserException(f"Invalid city '{city_str}'")
            return city

        def parse_destination_card(route_str: str) -> FrozenSet[City]:
            city_strs = route_str.split(",")
            if len(city_strs) != 2:
                raise ParserException(
                    "Expected two comma-separated cities, like 'New-York,San-Francisco', for each destination card"
                )
            return frozenset(map(parse_city, city_strs))

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="action_name")

        # DestinationCardSelectionAction,

        build_parser = subparsers.add_parser("build")
        build_parser.add_argument("city1", type=parse_city)
        build_parser.add_argument("city2", type=parse_city)
        build_parser.add_argument("--wildcards", type=int, default=0)
        build_parser.add_argument("--color", type=parse_color, default=None)
        build_parser.add_argument("--double-color", type=parse_color, default=None)

        draw_parser = subparsers.add_parser("draw")
        draw_parser.add_argument("color", type=parse_color, nargs="?", default=False)

        _ = subparsers.add_parser("draw-destinations")

        pick_parser = subparsers.add_parser("pick")
        pick_parser.add_argument(
            "destination_cards", nargs="+", type=parse_destination_card
        )

        _ = subparsers.add_parser("pass")

        def parse(action_str: str) -> Action:
            try:
                cmd = parser.parse_args(action_str.split())
            except (Exception, SystemExit) as error:
                raise ParserException(error)

            if cmd.action_name == "build":
                cities = frozenset([cmd.city1, cmd.city2])
                cities_list = list(cities)
                routes = self.box.board.cities_to_routes[cities]
                color: Optional[Color] = cmd.color
                if len(routes) == 0:
                    raise ParserException(
                        f"No route from '{cities_list[0]}' to '{cities_list[1]}' exists"
                    )
                else:
                    if len(routes) == 1:
                        route = routes[0]
                    else:
                        double_color: Optional[Color] = cmd.double_color
                        filtered_routes = list(
                            route for route in routes if route.color == double_color
                        )
                        if len(filtered_routes) == 0:
                            raise ParserException(
                                f"No route from '{cities_list[0]}' to '{cities_list[1]}' exists with {'gray' if double_color is None else double_color.name} color"
                            )
                        route = filtered_routes[0]
                    if (
                        route.color is None
                        and color is None
                        and cmd.wildcards < route.length
                    ):
                        raise ParserException(
                            "Route is gray; a color must be specified to build with if not using all wildcards"
                        )
                    return gaction.BuildAction(
                        route,
                        TrainCards(
                            {color: route.length - cmd.wildcards, None: cmd.wildcards}
                        ),
                    )

            elif cmd.action_name == "draw":
                if cmd.color is False:
                    return gaction.TrainCardPickAction(
                        draw_known=False, selected_card_if_known=None
                    )
                else:
                    return gaction.TrainCardPickAction(
                        draw_known=True, selected_card_if_known=cmd.color
                    )
            elif cmd.action_name == "draw-destinations":
                return gaction.DestinationCardPickAction()
            elif cmd.action_name == "pick":
                cards = []
                cities_to_card = {
                    card.cities: card
                    for card in self.state.hand.unselected_destination_cards
                }
                for cities in cmd.destination_cards:
                    if cities in cities_to_card:
                        cards.append(cities_to_card[cities])
                    else:
                        cities_list = list(cities)
                        raise ParserException(
                            f"No destination card from '{cities_list[0].name}' to '{cities_list[1].name}' in hand to pick from"
                        )
                return gaction.DestinationCardSelectionAction(frozenset(cards))
            elif cmd.action_name == "pass":
                return gaction.PassAction()
            else:
                raise ParserException(f"Unexpected action '{cmd.action_name}'")

        return parse

    def parse_action(self, action_str: str) -> Action:
        return self._parser(action_str)


class RandomActor(PlayerActor):
    def get_action(self) -> Action:
        all_actions = list(self.state.get_legal_actions())
        return random.choice(all_actions).action


class AIActor(PlayerActor):
    pass
