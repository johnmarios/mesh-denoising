"""Run the complete correspondence-free dynamic-mesh Bonus pipeline."""

import config
from questions.bonus_cpd.sequence_registration import (
    register_canonical_sequence,
)
from questions.q8_dynamic.viewer import DynamicMeshApp


def registered_sequence_is_ready() -> bool:
    """Check that CPD produced one output mesh for every remeshed input."""
    input_frames = list(config.BONUS_DYNAMIC_FOLDER.glob("*.obj"))
    registered_frames = list(config.BONUS_REGISTERED_FOLDER.glob("*.obj"))

    return (
        len(input_frames) > 0
        and len(registered_frames) == len(input_frames)
    )


def ensure_registered_sequence() -> None:
    """Compute full-sequence CPD only when no complete result exists."""
    if registered_sequence_is_ready():
        print(f"Using saved CPD sequence: {config.BONUS_REGISTERED_FOLDER}")
        return

    print("No complete CPD sequence was found.")
    print("Registering frame 0 onto every independently remeshed frame...")
    register_canonical_sequence(
        input_folder=config.BONUS_DYNAMIC_FOLDER,
        output_folder=config.BONUS_REGISTERED_FOLDER,
    )


def main() -> None:
    ensure_registered_sequence()

    app = DynamicMeshApp(
        folder=config.BONUS_DYNAMIC_FOLDER,
        require_correspondence=False,
        registered_folder=config.BONUS_REGISTERED_FOLDER,
    )
    app.mainLoop()


if __name__ == "__main__":
    main()
