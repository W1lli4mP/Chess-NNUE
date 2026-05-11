import numpy as np

"""
simple material baseline for NNUE validation

provides a sanity-check baseline so the NNUE can be compared
against a basic handcrafted evaluator

!not used for training
"""

def material_from_sparse_features(features: np.ndarray) -> int:
    """
    calculates white-relative material score from sparse absolute
    feature indices

    positive -> white is ahead in material
    negative -> black is ahead in material
    """
    score = 0

    for idx in features:
        if idx < 0:
            continue

        colour_block = idx // 384
        within_colour = idx % 384
        piece_type_index = within_colour // 64

        if piece_type_index == 0:
            value = 100
        elif piece_type_index == 1:
            value = 320
        elif piece_type_index == 2:
            value = 330
        elif piece_type_index == 3:
            value = 500
        elif piece_type_index == 4:
            value = 900
        elif piece_type_index == 5:
            value = 0
        else:
            raise ValueError(f"Invalid piece type index from feature {idx}")

        # feature layout uses block 0 for white and 1 for black (weird)
        if colour_block == 0:
            score += value
        else:
            score -= value

    return score