from pathlib import Path

WIDTH = 1400
HEIGHT = 900
DEFAULT_MODEL = "BUNNY"
DEFAULT_SEED = 7

NOISE_LEVEL_MIN = 0.02
NOISE_LEVEL_MAX = 0.20
DEFAULT_NOISE_LEVEL = 0.101

SMOOTHING_STRENGTH_MAX = 0.80
DEFAULT_SMOOTHING_STRENGTH = 0.288
ITERATIONS_PER_PRESS = 10

ROOT = Path(__file__).resolve().parent
RESOURCE_DIR = ROOT / "resources"
OUTPUT_DIR = ROOT / "outputs"

# Dynamic mesh dataset. Change only this name to select another sequence.
# Available coursework sequences: "bouncing" and "swing".
DYNAMIC_MESH_NAME = "swing"

# Question 8 uses the original sequence with vertex correspondence.
QUESTION_8_DYNAMIC_FOLDER = RESOURCE_DIR / "dynamic" / DYNAMIC_MESH_NAME

# The Bonus uses the same sequence after independent remeshing.
BONUS_DYNAMIC_FOLDER = (
    RESOURCE_DIR
    / "dynamic"
    / f"{DYNAMIC_MESH_NAME}_nocorr"
    / "clean_remeshed"
)

# Common-topology sequence produced by CPD and consumed by the Bonus viewer.
# The sequence name is part of the output path so cached results from different
# dynamic meshes can never be mixed accidentally.
BONUS_REGISTERED_FOLDER = (
    OUTPUT_DIR / f"cpd_registered_sequence_{DYNAMIC_MESH_NAME}"
)

# Default source (moving) and target (fixed) frames for `python bonus.py`.
CPD_DEMO_SOURCE_FRAME = 0
CPD_DEMO_TARGET_FRAME = 35
