"""Show how CPD moves one independently remeshed frame onto another."""

import sys

import config
from questions.bonus_cpd.viewer import CPDRegistrationApp


def parse_frame_numbers(arguments: list[str]) -> tuple[int, int]:
    """Read `--1 --35`, `--1, --35`, or simply `1 35`."""
    text = " ".join(arguments).replace(",", " ")
    tokens = text.split()

    if len(tokens) != 2:
        raise ValueError(
            "Give exactly two frames, for example: "
            "python cpd_demo.py --1 --35"
        )

    frame_numbers = []
    for token in tokens:
        number = token.removeprefix("--")
        if not number.isdigit():
            raise ValueError(
                f"'{token}' is not a valid non-negative frame number."
            )
        frame_numbers.append(int(number))

    source_frame, target_frame = frame_numbers
    if source_frame == target_frame:
        raise ValueError("Source and target must be different frames.")

    return source_frame, target_frame


def run_demo(source_frame: int, target_frame: int) -> None:
    app = CPDRegistrationApp(
        folder=config.BONUS_DYNAMIC_FOLDER,
        source_frame=source_frame,
        target_frame=target_frame,
    )
    app.mainLoop()


def main() -> None:
    try:
        source_frame, target_frame = parse_frame_numbers(sys.argv[1:])
        run_demo(source_frame, target_frame)
    except (FileNotFoundError, ValueError) as error:
        print(f"CPD demo error: {error}")
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
