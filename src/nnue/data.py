import zstandard as zstd
import io
import chess.pgn
from stockfish import Stockfish
import numpy as np
from pathlib import Path

from paths import here
from config import EVAL_ENGINE_DEPTH, EVAL_ENGINE_SKILL_LEVEL, EVAL_ENGINE_THREADS, SHARD_SIZE, MAX_GAMES, DATA_SPLIT_SEED

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
    shard_dir.mkdir(parents=True, exist_ok=True) # create the folder beforehand if it does not exist already

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
    #! stockfish returns scores from side-to-move perspective, NOT from white's perspective
    if eval_dict["type"] == "cp":
        cp = eval_dict["value"]

    elif eval_dict["type"] == "mate":
        # restrict checkmate scores to +-10000 centipawns
        cp = 10000 if eval_dict["value"] > 0 else -10000
    else:
        return None

    # convert side-to-move-relative eval to white-relative eval
    # AFTER:
    # positive = good for White
    # negative = good for Black

    if board.turn == chess.BLACK:
        cp = -cp

    return cp
    
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

def sparse_to_dense_batch(sparse_features: np.ndarray) -> np.ndarray:
    # converts (N, 32) sparse indices to (N, 768) dense binary vectors
    n = sparse_features.shape[0]
    dense = np.zeros((n, 768), dtype=np.float32)

    for i in range(n):
        for idx in sparse_features[i]:
            if idx >= 0: # skip "-1" padding                
                dense[i, idx] = 1.0
    return dense

class DataLoader:
    # memory efficient data loader that keeps data sparse until batch time

    def __init__(self, sparse_features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool = True):
        self.sparse_features = sparse_features # (N, 32) int16
        self.labels = labels # (N,) float32
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(labels)
        self.indices = np.arange(self.n)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start in range(0, self.n, self.batch_size):
            batch_idx = self.indices[start:start + self.batch_size]
            sparse_batch = self.sparse_features[batch_idx]
            labels_batch = self.labels[batch_idx].reshape(-1, 1) # (batch, 1) to match network output

            # convert to dense only for this batch
            dense_batch = sparse_to_dense_batch(sparse_batch)
            yield dense_batch, labels_batch

def get_data(shard_indices: list[int] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # loads data and returns sparse features and normalised labels
    shard_dir = Path(here("data", "nnue_shards"))
    all_shards = sorted(shard_dir.glob("shard_*.npz"))

    shards = [all_shards[i] for i in shard_indices] if shard_indices is not None else all_shards

    all_features = []
    all_evals = []

    print(f"Loading {len(shards)} shards...")
    for shard_path in shards:
        data = np.load(shard_path)
        all_features.append(data["features"])
        all_evals.append(data["evals"])

    # combine all shards
    features = np.vstack(all_features) # (N, 32) sparse int16
    evals = np.concatenate(all_evals) # (N,) int16 centipawns

    # normalise evals to [-1.0, 1.0]
    y = np.tanh(evals.astype(np.float32) / 400.0).astype(np.float32)

    # train/test split (90/10)
    n = len(features)
    split = int(n * 0.9)

    # make the split more deterministic
    rng = np.random.default_rng(DATA_SPLIT_SEED)
    perm = rng.permutation(n)

    train_features = features[perm[:split]]
    train_y = y[perm[:split]]

    test_features = features[perm[split:]]
    test_y = y[perm[split:]]

    print(f"Loaded {n:,} positions ({split:,} train, {n-split:,} test)")
    return train_features, train_y, test_features, test_y