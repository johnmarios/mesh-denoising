"""Show how CPD moves one independently remeshed frame onto another."""

import argparse
import sys

import config
from questions.bonus_cpd.viewer import CPDRegistrationApp


def parse_arguments(arguments: list[str]) -> tuple[str, int, int]:
    """Read the model and two frames, also accepting the old `--1 --35` form."""
    parser = argparse.ArgumentParser(
        description="Show CPD registration between two dynamic-mesh frames."
    )
    parser.add_argument(
        "model",
        choices=config.DYNAMIC_MESH_NAMES,
        help="dynamic mesh sequence to load",
    )
    parser.add_argument("source_frame", type=int, help="moving source frame")
    parser.add_argument("target_frame", type=int, help="fixed target frame")

    tokens = " ".join(arguments).replace(",", " ").split()
    normalized = [
        token.removeprefix("--")
        if token.removeprefix("--").isdigit()
        else token
        for token in tokens
    ]
    parsed = parser.parse_args(normalized)

    if parsed.source_frame < 0 or parsed.target_frame < 0:
        parser.error("frame numbers must be zero or positive")
    if parsed.source_frame == parsed.target_frame:
        parser.error("source and target must be different frames")

    return parsed.model, parsed.source_frame, parsed.target_frame


def run_demo(
    model_name: str,
    source_frame: int,
    target_frame: int,
) -> None:
    app = CPDRegistrationApp(
        folder=config.bonus_dynamic_folder(model_name),
        source_frame=source_frame,
        target_frame=target_frame,
    )
    app.mainLoop()


def main() -> None:
    try:
        model_name, source_frame, target_frame = parse_arguments(sys.argv[1:])
        run_demo(model_name, source_frame, target_frame)
    except (FileNotFoundError, ValueError) as error:
        print(f"CPD demo error: {error}")
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
