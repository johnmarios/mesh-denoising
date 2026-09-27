"""Interactive VVR viewer for two-frame non-rigid CPD registration."""

from pathlib import Path

import numpy as np

import config as project_config
from core.mesh import copy_mesh, vertex_normals
from core.visualization import set_axes_visible
from questions.bonus_cpd import config
from questions.bonus_cpd.cpd import (
    gaussian_kernel,
    nonrigid_cpd,
)
from questions.bonus_cpd.evaluation import (
    create_target_surface,
    error_heatmap_colors,
    point_to_surface_distances,
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
        self.source_triangles = np.asarray(source_mesh.triangles, dtype=np.int64)
        self.target_vertices = np.asarray(target_mesh.vertices, dtype=float)
        self.target_triangles = np.asarray(target_mesh.triangles, dtype=np.int64)

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
        # The target is a surface, not an ordered list of corresponding vertices.
        self.target_surface = create_target_surface(
            self.target_vertices,
            self.target_triangles,
        )

        final_vertices = self.transformed_source_vertices(
            len(self.weight_history) - 1
        )
        initial_errors = point_to_surface_distances(
            self.source_vertices,
            self.target_surface,
        )
        final_errors = point_to_surface_distances(
            final_vertices,
            self.target_surface,
        )

        all_endpoint_errors = np.concatenate((initial_errors, final_errors))
        self.heatmap_color_limit = float(
            np.percentile(
                all_endpoint_errors,
                config.HEATMAP_LIMIT_PERCENTILE,
            )
        )

        self.iteration = 0
        self.heatmap_visible = False

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
        self.heatmap_mesh = copy_mesh(source_mesh)

        self.addShape(self.initial_points, "initial")
        self.addShape(self.target_points, "target")
        self.addShape(self.moving_points, "moving")

        self.print_help()
        self.show_iteration(0)

    def transformed_source_vertices(self, iteration):
        """Apply the deformation of one CPD iteration to the full source mesh."""
        weights = self.weight_history[iteration]
        return self.source_vertices + self.display_kernel @ weights

    def show_iteration(self, iteration):
        """Update the blue points or the heatmap for one CPD iteration."""
        self.iteration = int(np.clip(iteration, 0, len(self.weight_history) - 1))
        transformed_vertices = self.transformed_source_vertices(self.iteration)

        if self.heatmap_visible:
            surface_errors = point_to_surface_distances(
                transformed_vertices,
                self.target_surface,
            )
            self.update_heatmap_mesh(transformed_vertices, surface_errors)
            self.updateShape("heatmap")
        else:
            self.moving_points.points = transformed_vertices
            self.updateShape("moving")

        self.print(
            f"CPD iteration {self.iteration:02d}/"
            f"{len(self.weight_history) - 1:02d}"
        )

    def update_heatmap_mesh(self, vertices, errors):
        """Put the current deformation and its error colors on the source mesh."""
        self.heatmap_mesh.vertices = vertices
        self.heatmap_mesh.vertex_normals = vertex_normals(
            vertices,
            self.source_triangles,
        )
        self.heatmap_mesh.vertex_colors = error_heatmap_colors(
            errors,
            self.heatmap_color_limit,
        )

    def toggle_heatmap(self):
        """Switch between blue registration points and the error-colored mesh."""
        transformed_vertices = self.transformed_source_vertices(self.iteration)
        surface_errors = point_to_surface_distances(
            transformed_vertices,
            self.target_surface,
        )

        if self.heatmap_visible:
            self.removeShape("heatmap")
            self.moving_points.points = transformed_vertices
            self.addShape(self.moving_points, "moving")
            self.heatmap_visible = False
            self.print("Blue CPD points shown.")
        else:
            self.removeShape("moving")
            self.update_heatmap_mesh(transformed_vertices, surface_errors)
            self.addShape(self.heatmap_mesh, "heatmap")
            self.heatmap_visible = True
            self.print(
                "Point-to-surface error heatmap shown. "
                f"Blue=0, red>={self.heatmap_color_limit:.6g}."
            )

    def on_key_press(self, symbol, modifiers):
        if symbol == Key.RIGHT:
            self.show_iteration(self.iteration + 1)
        elif symbol == Key.LEFT:
            self.show_iteration(self.iteration - 1)
        elif symbol == Key.R:
            self.show_iteration(0)
        elif symbol == Key.E:
            self.toggle_heatmap()
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
            "E            : blue points / error heatmap\n"
            "A            : coordinate axes on/off\n"
            "?            : show this menu\n"
            "============================================\n"
        )
