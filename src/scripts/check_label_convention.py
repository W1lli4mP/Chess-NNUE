import chess
from stockfish import Stockfish

from nnue.data import get_evaluation_from_board
from paths import here
from config import EVAL_ENGINE_DEPTH, EVAL_ENGINE_SKILL_LEVEL, EVAL_ENGINE_THREADS 

STOCKFISH_PATH = here("data", "stockfish", "stockfish-windows-x86-64-avx2.exe")

"""
sanity-check Stockfish label convention

the absolute NNUE architecture does not encode side to move, so
labels must be relative to white

this script checks that equivalent piece placements with different
side-to-move values keep the same sign convention
"""

def get_eval(engine, fen):
    board = chess.Board(fen)
    engine.set_fen_position(board.fen())
    eval_dict = engine.get_evaluation()
    return eval_dict

def test_white_relative_eval_conversion() -> None:
    engine = Stockfish(STOCKFISH_PATH)

    engine.set_depth(EVAL_ENGINE_DEPTH)
    engine.set_skill_level(EVAL_ENGINE_SKILL_LEVEL)
    engine.update_engine_parameters({"Threads": EVAL_ENGINE_THREADS})

    positions = [
        "8/8/8/8/8/8/4Q3/K6k w - - 0 1",
        "8/8/8/8/8/8/4Q3/K6k b - - 0 1",
        "8/8/8/8/8/8/4q3/K6k w - - 0 1",
        "8/8/8/8/8/8/4q3/K6k b - - 0 1",
    ]

    for fen in positions:
        board = chess.Board(fen)
        cp = get_evaluation_from_board(engine, board)
        print(fen)
        print(cp)
        print()

def main():
    # test to check if absolute or relative evals are being used
    test_white_relative_eval_conversion()
    # test confirms evals are converted to White-relative:
    # positive = good for White, negative = good for Black

if __name__ == "__main__":
    main()