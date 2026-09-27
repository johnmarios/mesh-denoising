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

DYNAMIC_MESH_NAMES = ("bouncing", "swing")


def question_8_dynamic_folder(model_name: str) -> Path:
    """Return the original sequence, whose frames have correspondence."""
    return RESOURCE_DIR / "dynamic" / model_name


def bonus_dynamic_folder(model_name: str) -> Path:
    """Return the independently remeshed sequence used by the Bonus."""
    return (
        RESOURCE_DIR
        / "dynamic"
        / f"{model_name}_nocorr"
        / "clean_remeshed"
    )


def bonus_registered_folder(model_name: str) -> Path:
    """Return the CPD output folder for one sequence."""
    return OUTPUT_DIR / f"cpd_registered_sequence_{model_name}"
