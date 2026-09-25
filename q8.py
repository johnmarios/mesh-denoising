"""Run Question 8 with the dynamic mesh selected in config.py."""

import config
from questions.q8_dynamic.viewer import DynamicMeshApp


def main() -> None:
    app = DynamicMeshApp(config.QUESTION_8_DYNAMIC_FOLDER)
    app.mainLoop()


if __name__ == "__main__":
    main()
