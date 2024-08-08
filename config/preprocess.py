import os

from dotenv import load_dotenv

load_dotenv()

PREPROCESSING_INPUT_PATH = os.getenv('PREPROCESSING_INPUT_PATH', 'raw')
PREPROCESSING_OUTPUT_PATH = os.getenv('PREPROCESSING_OUTPUT_PATH', 'preprocessed')
TARGET_RMS = float(os.getenv('TARGET_RMS', 0.1))