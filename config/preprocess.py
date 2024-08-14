import os

from dotenv import load_dotenv

load_dotenv()

PREPROCESSING_CLEAN_PATH = os.getenv('PREPROCESSING_CLEAN_PATH', 'clean_raw')
PREPROCESSING_NOISE_PATH = os.getenv('PREPROCESSING_NOISE_PATH', 'noise_raw')
PREPROCESSING_INPUT_PATH = os.getenv('PREPROCESSING_INPUT_PATH', 'input')
PREPROCESSING_TARGET_PATH = os.getenv('PREPROCESSING_TARGET_PATH', 'target')
TARGET_RMS = float(os.getenv('TARGET_RMS', 0.1))