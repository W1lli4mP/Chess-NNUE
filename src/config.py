# metadata
MODEL_NAME = "prototype_3.npz"
MODEL_RESULTS_NAME = "prototype_3.json"
MODEL_EXPORT_NAME = "prototype_3.npz"

# hyperparameters for the neural network
LAYER_DIMS = [768, 128, 1]
INPUT_UNITS = LAYER_DIMS[0]
HIDDEN_UNITS = sum(LAYER_DIMS[1:-1])
OUTPUT_UNITS = LAYER_DIMS[-1]

LEARNING_RATE = 0.005
BATCH_SIZE = 64
EPOCHS = 100
L2_LAMBDA = 1e-4
DROPOUT_RATE = 0
MOMENTUM = 0.9

# data labelling
EVAL_ENGINE_DEPTH = 8
EVAL_ENGINE_SKILL_LEVEL = 20
EVAL_ENGINE_THREADS = 8

SHARD_SIZE = 100000
MAX_GAMES = 100000  # set to None for unlimited

# selectable seeds for reproducibility
DATA_SPLIT_SEED = 33
TRAINING_SEED = 33