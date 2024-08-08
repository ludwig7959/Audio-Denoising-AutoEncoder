import os

from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv('MODEL_PATH')
NOISY_PATH = os.getenv('NOISY_PATH')
OUTPUT_PATH = os.getenv('OUTPUT_PATH')