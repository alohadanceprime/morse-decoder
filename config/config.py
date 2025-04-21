class DataConfig:
    CSV_FILE = "train.csv"
    ROOT_DIR = "data/spectrogramms"
    SYMBOLS = "0123456789АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ# "
    BLANK_CHAR = "&"
    TRAIN_SIZE = 29000
    TEST_SIZE = 1000


class ModelConfig:
    HIDDEN_SIZE = 256
    NUM_RNN_LAYERS = 3
    CNN_OUTPUT_CHANNELS = 256
    RNN_HIDDEN_SIZE = 128


class TrainConfig:
    BATCH_SIZE = 128
    NUM_EPOCHS = 1000
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 1e-6
    GRAD_CLIP_MAX_NORM = 5
    RANDOM_STATE = 42
    CHECKPOINT_DIR = "model_params"


class CTCConfig:
    BLANK_INDEX = 0
