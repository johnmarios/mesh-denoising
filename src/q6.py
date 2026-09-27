"""Run the existing denoising application on a Question 6 mesh."""

import argparse

from app import ProjectApp


Q6_MODELS = {
    "reference": "BUNNY_Q6_REFERENCE",
    "remeshed": "BUNNY_Q6_REMESHED",
    "holes": "BUNNY_Q6_HOLES",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Question 6 mesh experiments")
    parser.add_argument(
        "variant",
        choices=Q6_MODELS,
        nargs="?",
        default="remeshed",
        help="Question 6 mesh to load (default: remeshed)",
    )
    args = parser.parse_args()

    app = ProjectApp(Q6_MODELS[args.variant])
    app.mainLoop()


if __name__ == "__main__":
    main()
