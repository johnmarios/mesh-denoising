
import config as project_config


# Follow the same single sequence selection used by q8.py and bonus.py.
NAME = project_config.DYNAMIC_MESH_NAME
INPUT_FOLDER = project_config.QUESTION_8_DYNAMIC_FOLDER
OUTPUT_FOLDER = (
    project_config.RESOURCE_DIR / "dynamic" / f"{NAME}_nocorr"
)

MIN_FACE_RATIO = 0.60
MAX_FACE_RATIO = 0.95

NOISE_TYPE = "NORMAL"
NOISE_LEVEL = 0.20

RANDOM_SEED = 100

# None for all frames
MAX_FRAMES = None

