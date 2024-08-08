import os
from distutils.util import strtobool
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TRAINING_INPUT_PATH = os.getenv('TRAINING_INPUT_PATH', 'input')
TRAINING_TARGET_PATH = os.getenv('TRAINING_TARGET_PATH', 'target')
NORMALIZATION = bool(strtobool(os.getenv('NORMALIZATION', 'True')))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 4))

EPOCHS = int(os.getenv('EPOCHS', 30))
SAVE_EACH_EPOCH = bool(strtobool(os.getenv('SAVE_EACH_EPOCH', 'True')))

VALIDATION = bool(strtobool(os.getenv('VALIDATION', 'True')))
VALIDATION_INPUT_PATH = v_input
VALIDATION_TARGET_PATH = v_target

EARLY_STOPPING = bool(strtobool(os.getenv('EARLY_STOPPING', 'True')))
EARLY_STOPPING_LOSS = os.getenv('EARLY_STOPPING_LOSS', 'loss')
EARLY_STOPPING_PATIENCE = int(os.getenv('EARLY_STOPPING_PATIENCE', 5))