import numpy as np
from pathlib import Path

from nnue.data import get_data
from paths import here

def verify_if_new_shards_load_correctly():

    train_features, train_y, test_features, test_y = get_data()

    print(train_features.shape)
    print(train_y.shape)
    print(test_features.shape)
    print(test_y.shape)

    print("train_y min/max/mean:", train_y.min(), train_y.max(), train_y.mean())
    print("test_y min/max/mean:", test_y.min(), test_y.max(), test_y.mean())

def verify_raw_eval_distribution():
    shard_dir = Path(here("data", "nnue_shards"))
    shards = sorted(shard_dir.glob("shard_*.npz"))

    all_evals = []

    for shard in shards:
        data = np.load(shard)
        evals = data["evals"]
        all_evals.append(evals)

        print(
            shard.name,
            "positions:", len(evals),
            "min:", evals.min(),
            "max:", evals.max(),
            "mean:", evals.mean()
        )

    evals = np.concatenate(all_evals)

    print()
    print("TOTAL:", len(evals))
    print("min:", evals.min())
    print("max:", evals.max())
    print("mean:", evals.mean())
    print("median:", np.median(evals))
    print("std:", evals.std())

    print()
    print("positive:", np.mean(evals > 0))
    print("negative:", np.mean(evals < 0))
    print("zero:", np.mean(evals == 0))

    print()
    for limit in [100, 300, 500, 1000, 3000, 9000]:
        print(f"|eval| <= {limit}:", np.mean(np.abs(evals) <= limit))

    print()
    print("mate-capped +10000:", np.sum(evals == 10000))
    print("mate-capped -10000:", np.sum(evals == -10000))

def verify_feature_indices():
    shard_dir = Path(here("data", "nnue_shards"))
    shards = sorted(shard_dir.glob("shard_*.npz"))

    bad_count = 0
    max_piece_count = 0
    min_piece_count = 32

    for shard in shards:
        data = np.load(shard)
        features = data["features"]

        bad = features[(features != -1) & ((features < 0) | (features >= 768))]
        if len(bad):
            print(shard.name, "bad features:", bad[:20])
            bad_count += len(bad)

        piece_counts = np.sum(features != -1, axis=1)
        max_piece_count = max(max_piece_count, int(piece_counts.max()))
        min_piece_count = min(min_piece_count, int(piece_counts.min()))

    print("total bad features:", bad_count)
    print("min piece count:", min_piece_count)
    print("max piece count:", max_piece_count)

def main():
    verify_feature_indices()

if __name__ == "__main__":
    main()