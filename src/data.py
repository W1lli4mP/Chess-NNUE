import zstandard as zstd
import io
import chess.pgn
from stockfish import Stockfish
import numpy as np
from pathlib import Path

from paths import here
from config import EVAL_ENGINE_DEPTH, EVAL_ENGINE_SKILL_LEVEL, EVAL_ENGINE_THREADS, SHARD_SIZE, MAX_GAMES

STOCKFISH_PATH = here("data", "stockfish", "stockfish-windows-x86-64-avx2.exe")
MAX_PIECES = 32  # maximum pieces on a chess board

def calculate_index(piece: chess.Piece, square: int) -> int:
    """
    ENCODING
    piece colour
    piece type (from 0 to 5): p, n, b, r, q, k
    square (0 to 63)
    """
    return (not piece.color) * 64 * 6 + (piece.piece_type - 1) * 64 + square

def extract_training_data() -> None:
    print("STARTING")

    global features_buffer, evals_buffer, shard_idx
    features_buffer = []
    evals_buffer = []
    
    # resume from existing shards
    shard_dir = Path(here("data", "nnue_shards"))
    existing = list(shard_dir.glob("shard_*.npz"))
    if existing:
        shard_idx = max(int(p.stem.split('_')[1]) for p in existing) + 1
        print(f"Resuming from shard {shard_idx}")
    else:
        shard_idx = 0

    # for statistics/tracking
    games_processed = 0
    games_skipped = 0
    positions_count = 0

    data_path = here("data", "lichess_db_standard_rated_2022-07.pgn.zst")

    # initialise engine for evaluating
    engine = Stockfish(STOCKFISH_PATH)
    
    engine.set_depth(EVAL_ENGINE_DEPTH)
    engine.set_skill_level(EVAL_ENGINE_SKILL_LEVEL)
    engine.update_engine_parameters({"Threads": EVAL_ENGINE_THREADS})

    # decompress .zst
    with open(data_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()

        # read file in streams
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            # iterate through the games until all games exhausted
            while True:
                game = chess.pgn.read_game(text_stream)

                # skip non-existing games (EOF)
                if game is None:
                    break

                # skip variants
                if game.headers.get("Variant", "Standard") != "Standard":
                    games_skipped += 1
                    continue

                # skip abandoned games
                termination = game.headers.get("Termination", "")
                if "Abandoned" in termination:
                    games_skipped += 1
                    continue

                # the game is a tree of moves
                board = game.board()
                move_count = 0
                for node in game.mainline():
                    board.push(node.move)
                    move_count += 1
                    
                    # only evaluate every 4th position (75% faster and more efficient)
                    if move_count % 4 != 0:
                        continue

                    # construct features by iterating over the board's current position
                    features = get_features_from_board(board)

                    # get eval from stockfish to attach a label to the features
                    eval = get_evaluation_from_board(engine, board)
                    
                    # skip position if no evaluation given
                    if eval is None:
                        continue

                    # push to buffer
                    features_buffer.append(features)
                    evals_buffer.append(eval)

                    positions_count += 1

                    # save features and eval when buffers exceed shard size
                    if len(features_buffer) >= SHARD_SIZE:
                        save_shard(positions_count)
                
                games_processed += 1

                if MAX_GAMES and games_processed >= MAX_GAMES:
                    print(f"Reached {MAX_GAMES} games limit, stopping...")
                    break

                if games_processed % 100 == 0:
                    print(f"Games processed: {games_processed} Games skipped: {games_skipped}")
            
            # save leftover data
            save_shard(positions_count)
            print(f"Complete! {positions_count} positions extracted")

def get_features_from_board(board: chess.Board) -> np.ndarray:
    features = []
    for square in range(64):
        piece = board.piece_at(square)
        if piece:
            # add index encoding and append to features - sparse features
            idx = calculate_index(piece, square)
            features.append(idx)
    
    # pad to fixed length with -1
    padded = np.full(MAX_PIECES, -1, dtype=np.int16)
    padded[:len(features)] = features
    return padded

def get_evaluation_from_board(engine: Stockfish, board: chess.Board) -> int | None:
    engine.set_fen_position(board.fen())
    eval_dict = engine.get_evaluation()

    # standardise into centipawns metric
    if eval_dict["type"] == "cp":
        return eval_dict["value"]
    elif eval_dict["type"] == "mate":
        # restrict checkmate scores to +-10000 centipawns
        return 10000 if eval_dict["value"] > 0 else -10000
    else:
        return None
    
def save_shard(total_positions: int = 0) -> None:
    global features_buffer, evals_buffer, shard_idx

    if not features_buffer:
        return
    
    fp = here("data", "nnue_shards", f"shard_{shard_idx:04d}.npz")
    
    # convert to numpy arrays with proper dtypes
    features_arr = np.array(features_buffer, dtype=np.int16)
    evals_arr = np.array(evals_buffer, dtype=np.int16)
    
    np.savez_compressed(fp, features=features_arr, evals=evals_arr)
    
    print(f"Saved {fp} ({len(features_buffer)} positions)")
    
    # save progress for reference
    with open(here("data", "nnue_shards", "progress.txt"), "w") as f:
        f.write(f"Last shard: {shard_idx}\nTotal positions: {total_positions}\n")

    features_buffer = []
    evals_buffer = []
    shard_idx += 1

# extract_training_data()
# FINAL STATS:
# ~ 1 hr
# 1618957 positions evaluated
# 100k games processed, 386 games skipped
# 17 .npz shards created

def sparse_to_dense_batch(sparse_features: np.ndarray) -> np.ndarray:
    # converts (N, 32) sparse indices to (N, 768) dense binary vectors
    n = sparse_features.shape[0]
    dense = np.zeros((n, 768), dtype=np.float32)

    for i in range(n):
        for idx in sparse_features[i]:
            if idx >= 0: # skip "-1" padding                
                dense[i, idx] = 1.0
    return dense

def get_data(shard_indices: list[int] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shard_dir = Path(here("data", "nnue_shards"))
    all_shards = sorted(shard_dir.glob("shard_*.npz"))

    shards = [all_shards[i] for i in shard_indices] if shard_indices is not None else all_shards

    all_features = []
    all_evals = []

    for shard_path in shards:
        data = np.load(shard_path)
        all_features.append(data["features"])
        all_evals.append(data["evals"])

    # combine all shards
    features = np.vstack(all_features) # (N, 32) sparse
    evals = np.concatenate(all_evals) # (N,) centipawns

    # convert sparse to dense
    X = sparse_to_dense_batch(features)

    # normalise evals to [-1.0, 1.0]
    y = np.tanh(evals / 400).astype(np.float32)

    # train/test split (90/10)
    n = len(X)
    split = int(n * 0.9)
    perm = np.random.permutation(n)

    train_X = X[perm[:split]]
    train_y = y[perm[:split]]

    test_X = X[perm[split:]]
    test_y = y[perm[split:]]

    return train_X, train_y, test_X, test_y