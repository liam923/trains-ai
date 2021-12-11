import csv
import sys
from multiprocessing import Pool
from typing import Dict, Callable, List, Mapping, Tuple, Generator, Any

from tqdm import tqdm

from trains.ai.actor import AiActor
from trains.ai.mc_expectiminimax import McExpectiminimaxActor
from trains.ai.mcts import UfMctsActor, BasicMctsActor
from trains.ai.route_building_probability_calculator import DummyRbpc
from trains.ai.utility_function import (
    RelativeUf,
    ImprovedExpectedScoreUf,
    ExpectedScoreUf,
)
from trains.game.box import Player, Box
from trains.game.game_actor import play_game, SimulatedGameActor
from trains.game.observed_state import ObservedState


AiMaker = Callable[[Box, Player], AiActor]
BoxMaker = Callable[[List[Player]], Box]


def ais_sets_to_compare() -> Generator[Tuple[str, Dict[str, AiMaker]], None, None]:
    yield "relative-improved", {
        "Expectiminimax (depth=3)": lambda box, player: McExpectiminimaxActor.make(
            box,
            player,
            utility_function=RelativeUf(ImprovedExpectedScoreUf()),
            route_building_probability_calculator=DummyRbpc,
            depth=5,
            print_state=False,
        ),
        "McExpectiminimax (depth=6, breadth=10)": lambda box, player: McExpectiminimaxActor.make(
            box,
            player,
            utility_function=RelativeUf(ImprovedExpectedScoreUf()),
            route_building_probability_calculator=DummyRbpc,
            depth=5,
            breadth=lambda _: 10,
            print_state=False,
        ),
        "UF-MCTS (iterations=1000)": lambda box, player: UfMctsActor.make(
            box,
            player,
            iterations=1000,
            utility_function=RelativeUf(ImprovedExpectedScoreUf()),
            print_state=False,
            show_progress=False,
        ),
        "Basic-MCTS (iterations=200)": lambda box, player: BasicMctsActor.make(
            box,
            player,
            iterations=200,
            print_state=False,
            show_progress=False,
        ),
    }

    yield "uf-compare", {
        "ExpectedScoreUf": lambda box, player: UfMctsActor.make(
            box,
            player,
            iterations=1000,
            utility_function=ExpectedScoreUf(),
            print_state=False,
            show_progress=False,
        ),
        "ImprovedExpectedScoreUf": lambda box, player: UfMctsActor.make(
            box,
            player,
            iterations=1000,
            utility_function=ImprovedExpectedScoreUf(),
            print_state=False,
            show_progress=False,
        ),
        "Relative ImprovedExpectedScoreUf": lambda box, player: UfMctsActor.make(
            box,
            player,
            iterations=1000,
            utility_function=RelativeUf(ImprovedExpectedScoreUf()),
            print_state=False,
            show_progress=False,
        ),
    }


def play_ais(
    ai_a: AiMaker, ai_b: AiMaker, games: int, box_maker: BoxMaker, pbar: Any
) -> Tuple[float, float, float, float]:
    p1 = Player("one")
    p2 = Player("two")
    box = box_maker([p1, p2])

    a_wins: float = 0
    b_wins: float = 0
    a_scores: float = 0
    b_scores: float = 0

    for i in range(games):
        game = SimulatedGameActor.make(box)
        if i % 2 == 0:
            ai1 = ai_a(box, p1)
            ai2 = ai_b(box, p2)
            a_player = p1
            b_player = p2
        else:
            ai1 = ai_b(box, p1)
            ai2 = ai_a(box, p2)
            a_player = p2
            b_player = p1

        winner = play_game({p1: ai1, p2: ai2}, game, print_result=False)
        if winner is None:
            a_wins += 0.5
            b_wins += 0.5
        elif winner == a_player:
            a_wins += 1
        elif winner == b_player:
            b_wins += 1

        a_scores += game.scores[a_player]
        b_scores += game.scores[b_player]

        pbar.update()  # type: ignore

    return a_wins / games, a_scores / games, b_wins / games, b_scores / games


def compare_ais(
    ais: Mapping[str, AiMaker], games: int, box_maker: BoxMaker
) -> Tuple[List[List[str]], List[List[str]]]:
    listed_ais = list(ais.items())
    win_results = [[""] + [ai_name for ai_name, _ in listed_ais]]
    win_results += [[ai_name] for ai_name, _ in listed_ais]
    score_results = [[""] + [ai_name for ai_name, _ in listed_ais]]
    score_results += [[ai_name] for ai_name, _ in listed_ais]

    with tqdm(total=(len(listed_ais) * (len(listed_ais) + 1)) * games // 2) as pbar:
        for i, (name_b, ai_b) in list(enumerate(listed_ais)):
            for j, (name_a, ai_a) in list(enumerate(listed_ais))[i:]:
                a_wins, a_score, b_wins, b_score = play_ais(
                    ai_a, ai_b, games, box_maker, pbar
                )
                win_results[i + 1].append(f"{a_wins}")
                if i != j:
                    win_results[j + 1].append(f"{b_wins}")
                score_results[i + 1].append(f"{a_score}")
                if i != j:
                    score_results[j + 1].append(f"{b_score}")

    return win_results, score_results


if __name__ == "__main__":
    box_maker = Box.small
    for ai_set_name, ais_to_compare in ais_sets_to_compare():
        win_data, score_data = compare_ais(ais_to_compare, 20, box_maker)

        with open(f"wins-{ai_set_name}.csv", "w") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerows(win_data)

        with open(f"scores-{ai_set_name}.csv", "w") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerows(score_data)
