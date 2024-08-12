import os

from dotenv import load_dotenv

load_dotenv()

PADDING_INPUT_PATH = os.getenv('PADDING_INPUT_PATH', 'noise_unprocessed')
PADDING_OUTPUT_PATH = os.getenv('PADDING_OUTPUT_PATH', 'noise_processed')
