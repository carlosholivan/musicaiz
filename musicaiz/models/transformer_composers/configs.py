from pathlib import Path
from datetime import datetime


class GPTConfigs:
    """
    ...
    The vocabulary size is given by the ``vocabulary.txt`` file that
    must be placed in the dataset path (this file is generated when
    tokenizing).
    """
    N_DECODERS = 2
    SEQ_LEN = 512
    EMBED_DIM = 32  # also d_model
    N_HEADS = 4  # must be divisor of embed dim
    DROPOUT = 0.1


class TrainConfigs:
    TRAIN_SPLIT = 0.8
    IS_SPLITTED = False  # if dataset is not already splitted in 2 dirs: train and validation
    CHECKPOINT_PATH = Path("results", str(datetime.now().strftime("%Y-%m-%d_%H-%M")))
    MODEL_NAME = "gpt"
    WEIGHT_DECAY = 0.01
    EPOCHS = 250
    BATCH_SIZE = 64
    LR = 5e-3
    ADAM_EPSILON = 1e-6
    CKPT_STEPS = 100  # steps to save checkpoint
    LOG_STEPS = 1
    LOG_DIR = Path("results", str(datetime.now().strftime("%Y-%m-%d_%H-%M")))
    GRAD_ACUM_STEPS = 1
    FP16 = True
    FP16_OPT_LEVEL = "O2"
