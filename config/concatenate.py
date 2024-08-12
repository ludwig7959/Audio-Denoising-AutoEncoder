import os

from dotenv import load_dotenv

load_dotenv()

CONCATENATING_INPUT_PATH = os.getenv('CONCATENATING_INPUT_PATH', 'segments')
CONCATENATING_OUTPUT_PATH = os.getenv('CONCATENATING_OUTPUT_PATH', 'concatenated')
CONCATENATING_COUNT = int(os.getenv('CONCATENATING_COUNT', 100))
