import struct
import chess
import numpy as np

from nnue.data import calculate_index

"""
numpy inference helpers for exported NNUE binaries

mirrors the future C-side implementation of the NNUE by:
- reading the raw .bin format
- building dense inputs from FENs for validation
- running dense or sparse forward propagation
"""

def load_binary_model(path):
    """
    load C-exported binary model
    """
    with open(path, "rb") as f:
        num_layers = struct.unpack("<i", f.read(4))[0]
        dims = [struct.unpack("<i", f.read(4))[0] for _ in range(num_layers + 1)]

        weights = []
        biases = []

        for i in range(num_layers):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            w_count = in_dim * out_dim
            w = np.frombuffer(f.read(w_count * 4), dtype="<f4").reshape(in_dim, out_dim)
            b = np.frombuffer(f.read(out_dim * 4), dtype="<f4")

            weights.append(w)
            biases.append(b)

        remaining = f.read()
        if remaining:
            raise ValueError(f"Extra bytes at end of file: {len(remaining)}")

    return dims, weights, biases

def dense_from_fen(fen: str) -> np.ndarray:
    """
    convert a FEN into a dense (1, 768) binary input vector

    mainly used to compare saved Python model output against
    exported binary model output
    """
    if fen == "startpos":
        fen = chess.STARTING_FEN

    board = chess.Board(fen)
    x = np.zeros((1, 768), dtype=np.float32)

    for square, piece in board.piece_map().items():
        idx = calculate_index(piece, square)
        x[0, idx] = 1.0

    return x

def forward_binary_dense(x: np.ndarray, weights, biases) -> np.ndarray:
    """
    run dense forward propagation for current architecture:
        768 -> ReLU(128) -> tanh(1)
    """
    hidden = x @ weights[0] + biases[0]
    hidden = np.maximum(hidden, 0.0)

    y = hidden @ weights[1] + biases[1]
    y = np.tanh(y)

    return y

def forward_binary_sparse(features: np.ndarray, weights, biases) -> float:
    """
    run sparse forward propagation from active feature indices

    avoids constructing a dense 768-vector
    """
    hidden = biases[0].astype(np.float32).copy()

    for idx in features:
        if idx >= 0:
            hidden += weights[0][idx]

    hidden = np.maximum(hidden, 0.0)

    y = hidden @ weights[1][:, 0] + biases[1][0]
    y = np.tanh(y)

    return float(y)