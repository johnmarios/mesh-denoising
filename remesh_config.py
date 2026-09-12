
from pathlib import Path

NAME = "bouncing"

INPUT_FOLDER = Path("resources/dynamic/" + NAME)
OUTPUT_FOLDER = Path("resources/dynamic/" + NAME + "_nocorr")

MIN_FACE_RATIO = 0.60
MAX_FACE_RATIO = 0.95

NOISE_TYPE = "NORMAL"
NOISE_LEVEL = 0.20

RANDOM_SEED = 100

# None for all frames
MAX_FRAMES = None
