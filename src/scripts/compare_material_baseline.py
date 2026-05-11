import numpy as np

from nnue.data import get_data
from nnue.inference import load_binary_model, forward_binary_sparse
from nnue.baseline import material_from_sparse_features
from paths import here
from config import MODEL_NAME

"""
compare the exported NNUE against a simple material evaluator on the
test split

for validation, NOT testing; checks whether NNUE has learned useful
signal beyond count (like patterns in positions, etc)
"""

BIN_MODEL_NAME = MODEL_NAME.replace(".npz", ".bin")

def main():
    _, weights, biases = load_binary_model(here("models", BIN_MODEL_NAME))

    _, _, test_features, test_y = get_data()

    nn_errors = []
    material_errors = []

    for features, target in zip(test_features, test_y):
        nn_pred = forward_binary_sparse(features, weights, biases)

        material_cp = material_from_sparse_features(features)
        material_pred = float(np.tanh(material_cp / 400.0))

        nn_errors.append(abs(nn_pred - float(target)))
        material_errors.append(abs(material_pred - float(target)))

    nn_errors = np.array(nn_errors)
    material_errors = np.array(material_errors)

    print("TEST NNUE MAE normalised:", nn_errors.mean())
    print("TEST material MAE normalised:", material_errors.mean())

    print()
    print("TEST NNUE median error:", np.median(nn_errors))
    print("TEST material median error:", np.median(material_errors))

    print()
    print("TEST NNUE 90th percentile error:", np.percentile(nn_errors, 90))
    print("TEST material 90th percentile error:", np.percentile(material_errors, 90))

    print()
    print("TEST NNUE better fraction:", np.mean(nn_errors < material_errors))


if __name__ == "__main__":
    main()