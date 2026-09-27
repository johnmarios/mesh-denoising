"""Run the complete correspondence-free dynamic-mesh Bonus pipeline."""

import argparse
from pathlib import Path

import config
from questions.bonus_cpd.sequence_registration import (
    register_canonical_sequence,
)
from questions.q8_dynamic.viewer import DynamicMeshApp


def parse_model_name() -> str:
    parser = argparse.ArgumentParser(
        description="Bonus: correspondence-free dynamic mesh denoising"
    )
    parser.add_argument(
        "model",
        choices=config.DYNAMIC_MESH_NAMES,
        help="dynamic mesh sequence to load",
    )
    return parser.parse_args().model


def registered_sequence_is_ready(
    input_folder: Path,
    registered_folder: Path,
) -> bool:
    """Check that CPD produced one output mesh for every remeshed input."""
    input_frames = list(input_folder.glob("*.obj"))
    registered_frames = list(registered_folder.glob("*.obj"))

    return (
        len(input_frames) > 0
        and len(registered_frames) == len(input_frames)
    )


def ensure_registered_sequence(
    input_folder: Path,
    registered_folder: Path,
) -> None:
    """Compute full-sequence CPD only when no complete result exists."""
    if registered_sequence_is_ready(input_folder, registered_folder):
        print(f"Using saved CPD sequence: {registered_folder}")
        return

    print("No complete CPD sequence was found.")
    print("Registering frame 0 onto every independently remeshed frame...")
    register_canonical_sequence(
        input_folder=input_folder,
        output_folder=registered_folder,
    )


def main() -> None:
    model_name = parse_model_name()
    input_folder = config.bonus_dynamic_folder(model_name)
    registered_folder = config.bonus_registered_folder(model_name)

    ensure_registered_sequence(input_folder, registered_folder)

    app = DynamicMeshApp(
        folder=input_folder,
        require_correspondence=False,
        registered_folder=registered_folder,
    )
    app.mainLoop()


if __name__ == "__main__":
    main()
