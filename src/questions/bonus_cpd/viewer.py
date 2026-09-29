"""Interactive VVR viewer for two-frame non-rigid CPD registration."""

from pathlib import Path

import numpy as np

import config as project_config
from core.visualization import set_axes_visible
from questions.bonus_cpd import config
from questions.bonus_cpd.cpd import (
    gaussian_kernel,
    nonrigid_cpd,
)
from questions.bonus_cpd.sampling import (
    sample_mesh_surface_evenly,
)
from questions.q8_dynamic.sequence import load_frames
from vvrpywork.constants import Color, Key
from vvrpywork.scene import Scene3D_
from vvrpywork.shapes import PointSet3D


class CPDRegistrationApp(Scene3D_):
    """Inspect one independently remeshed frame registering onto another."""

    def __init__(
        self,
        folder: str | Path,
        source_frame: int,
        target_frame: int,
    ):
        folder = Path(folder)

        if source_frame < 0 or target_frame < 0:
            raise ValueError("Frame numbers must be zero or positive.")
        if source_frame == target_frame:
            raise ValueError("Source and target must be different frames.")

        self.source_frame = source_frame
        self.target_frame = target_frame

        super().__init__(
            project_config.WIDTH,
            project_config.HEIGHT,
            f"CPD: frame {source_frame} -> frame {target_frame}",
            output=True,
        )

        self.axes_visible = True

        frames = load_frames(folder, require_correspondence=False)

        required_frames = max(source_frame, target_frame) + 1

        if len(frames) < required_frames:
            raise ValueError(
                f"The folder contains {len(frames)} frames, but frame "
                f"{required_frames - 1} was requested."
            )

        source_mesh = frames[source_frame]
        target_mesh = frames[target_frame]

        self.print(
            f"Registering frame {source_frame} onto frame {target_frame}."
        )

        self.source_vertices = np.asarray(source_mesh.vertices, dtype=float)
        self.target_vertices = np.asarray(target_mesh.vertices, dtype=float)

        self.print("Sampling the two mesh surfaces...")

        source_control = sample_mesh_surface_evenly(
            source_mesh.vertices,
            source_mesh.triangles,
            config.SOURCE_CONTROL_POINTS,
            config.RANDOM_SEED,
            config.CONTROL_CANDIDATE_MULTIPLIER,
        )
        target_control = sample_mesh_surface_evenly(
            target_mesh.vertices,
            target_mesh.triangles,
            config.TARGET_CONTROL_POINTS,
            config.RANDOM_SEED + 1,
            config.CONTROL_CANDIDATE_MULTIPLIER,
        )

        self.print(
            f"Running CPD with {len(source_control)} moving and "
            f"{len(target_control)} target control points..."
        )

        result = nonrigid_cpd(
            target_points=target_control,
            moving_points=source_control,
            beta=config.BETA,
            regularization=config.REGULARIZATION,
            outlier_weight=config.OUTLIER_WEIGHT,
            max_iterations=config.MAX_ITERATIONS,
            tolerance=config.TOLERANCE,
        )

        self.weight_history = result.weight_history
        # This kernel does not change while inspecting the CPD iterations.
        self.display_kernel = gaussian_kernel(
            self.source_vertices,
            source_control,
            config.BETA,
        )

        self.iteration = 0

        self.initial_points = PointSet3D(
            self.source_vertices,
            size=0.55 * config.POINT_SIZE,
            color=(0.35, 0.35, 0.35, 0.25),
        )
        self.target_points = PointSet3D(
            self.target_vertices,
            size=config.POINT_SIZE,
            color=Color.RED,
        )
        self.moving_points = PointSet3D(
            self.source_vertices,
            size=config.POINT_SIZE,
            color=Color.BLUE,
        )

        self.addShape(self.initial_points, "initial")
        self.addShape(self.target_points, "target")
        self.addShape(self.moving_points, "moving")

        self.print_help()
        self.show_iteration(0)

    def transformed_source_vertices(self, iteration):
        """Apply the deformation of one CPD iteration to the full source mesh.
        Τ(Υ) = Υ + G(Υ, Y) W"""
        weights = self.weight_history[iteration]
        return self.source_vertices + self.display_kernel @ weights

    def show_iteration(self, iteration):
        """Update the blue points for one CPD iteration."""
        self.iteration = int(np.clip(iteration, 0, len(self.weight_history) - 1))
        transformed_vertices = self.transformed_source_vertices(self.iteration)
        self.moving_points.points = transformed_vertices
        self.updateShape("moving")

        self.print(
            f"CPD iteration {self.iteration:02d}/"
            f"{len(self.weight_history) - 1:02d}"
        )

    def on_key_press(self, symbol, modifiers):
        if symbol == Key.RIGHT:
            self.show_iteration(self.iteration + 1)
        elif symbol == Key.LEFT:
            self.show_iteration(self.iteration - 1)
        elif symbol == Key.R:
            self.show_iteration(0)
        elif symbol == Key.A:
            self.toggle_axes()
        elif symbol == Key.SLASH:
            self.print_help()

    def toggle_axes(self):
        new_state = not self.axes_visible
        if not set_axes_visible(self._plotter, new_state):
            self.print("The coordinate axes could not be found.")
            return

        self.axes_visible = new_state
        state = "shown" if self.axes_visible else "hidden"
        self.print(f"Coordinate axes {state}.")

    def print_help(self):
        self.print(
            "\n"
            "================ BONUS CPD =================\n"
            f"Frames       : {self.source_frame} -> {self.target_frame}\n"
            "Red points   : fixed target frame\n"
            "Blue points  : moving source frame\n"
            "Gray points  : original source position\n"
            "LEFT / RIGHT : previous / next CPD iteration\n"
            "R            : return to iteration 0\n"
            "A            : coordinate axes on/off\n"
            "?            : show this menu\n"
            "============================================\n"
        )
