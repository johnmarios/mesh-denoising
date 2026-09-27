"""Run Question 8 for one named dynamic mesh."""

import argparse

import config
from questions.q8_dynamic.viewer import DynamicMeshApp


def parse_model_name() -> str:
    parser = argparse.ArgumentParser(description="Question 8: dynamic mesh denoising")
    parser.add_argument(
        "model",
        choices=config.DYNAMIC_MESH_NAMES,
        help="dynamic mesh sequence to load",
    )
    return parser.parse_args().model


def main() -> None:
    model_name = parse_model_name()
    folder = config.question_8_dynamic_folder(model_name)
    app = DynamicMeshApp(folder)
    app.mainLoop()


if __name__ == "__main__":
    main()
