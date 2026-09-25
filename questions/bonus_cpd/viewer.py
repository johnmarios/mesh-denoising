"""Interactive VVR viewer for two-frame non-rigid CPD registration."""

from pathlib import Path
import time

import numpy as np

import config as project_config
from core.mesh import copy_mesh, vertex_normals
from core.visualization import set_axes_visible
from questions.bonus_cpd import config
from questions.bonus_cpd.cpd import (
    gaussian_kernel,
    nonrigid_cpd,
    symmetric_chamfer,
)
from questions.bonus_cpd.evaluation import (
    create_target_surface,
    error_heatmap_colors,
    error_statistics,
    point_to_surface_distances,
)
from questions.bonus_cpd.sampling import (
    sample_mesh_surface,
    sample_mesh_surface_evenly,
)
from questions.q8_dynamic.sequence import load_frames
from vvrpywork.constants import Color, Key
from vvrpywork.scene import Scene3D_
from vvrpywork.shapes import PointSet3D


class CPDRegistrationApp(Scene3D_):
    """Animate one independently remeshed frame registering onto another."""

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

        source_evaluation = sample_mesh_surface(
            source_mesh.vertices,
            source_mesh.triangles,
            config.EVALUATION_POINTS,
            config.RANDOM_SEED + 2,
        )
        target_evaluation = sample_mesh_surface(
            target_mesh.vertices,
            target_mesh.triangles,
            config.EVALUATION_POINTS,
            config.RANDOM_SEED + 3,
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
        self.variance_history = result.variance_history

        # These kernels do not change during the animation.
        self.display_kernel = gaussian_kernel(
            self.source_vertices,
            source_control,
            config.BETA,
        )
        evaluation_kernel = gaussian_kernel(
            source_evaluation,
            source_control,
            config.BETA,
        )

        self.chamfer_history = []
        for weights in self.weight_history:
            warped_evaluation = source_evaluation + evaluation_kernel @ weights
            self.chamfer_history.append(
                symmetric_chamfer(warped_evaluation, target_evaluation)
            )

        initial_chamfer = self.chamfer_history[0]
        final_chamfer = self.chamfer_history[-1]
        improvement = 100.0 * (initial_chamfer - final_chamfer) / initial_chamfer
        self.print(
            f"Chamfer before CPD: {initial_chamfer:.6g}\n"
            f"Chamfer after CPD : {final_chamfer:.6g}\n"
            f"Improvement       : {improvement:.2f}%"
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

        initial_statistics = error_statistics(initial_errors)
        final_statistics = error_statistics(final_errors)
        self.print_surface_comparison(initial_statistics, final_statistics)

        self.iteration = 0
        self.playing = False
        self.heatmap_visible = False
        self.last_animation_step = time.perf_counter()

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
        """Update the blue points and print the corresponding measurements."""
        self.iteration = int(np.clip(iteration, 0, len(self.weight_history) - 1))
        transformed_vertices = self.transformed_source_vertices(self.iteration)
        surface_errors = point_to_surface_distances(
            transformed_vertices,
            self.target_surface,
        )
        statistics = error_statistics(surface_errors)

        if self.heatmap_visible:
            self.update_heatmap_mesh(transformed_vertices, surface_errors)
            self.updateShape("heatmap")
        else:
            self.moving_points.points = transformed_vertices
            self.updateShape("moving")

        initial_chamfer = self.chamfer_history[0]
        current_chamfer = self.chamfer_history[self.iteration]

        if initial_chamfer > 0.0:
            improvement = 100.0 * (initial_chamfer - current_chamfer) / initial_chamfer
        else:
            improvement = 0.0

        self.print(
            f"Iteration {self.iteration:02d}/{len(self.weight_history) - 1:02d} | "
            f"Chamfer={current_chamfer:.6g} | "
            f"improvement={improvement:.2f}% | "
            f"mean surface error={statistics['mean']:.6g} | "
            f"P95={statistics['p95']:.6g} | "
            f"max={statistics['maximum']:.6g} | "
            f"sigma^2={self.variance_history[self.iteration]:.6g}"
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

    def print_surface_comparison(self, before, after):
        """Print the point-to-surface measurements before and after CPD."""
        self.print(
            "\nPoint-to-surface error (source vertices -> target surface):\n"
            "                    before CPD       after CPD\n"
            f"mean                {before['mean']:<16.6g} {after['mean']:.6g}\n"
            f"95th percentile     {before['p95']:<16.6g} {after['p95']:.6g}\n"
            f"maximum             {before['maximum']:<16.6g} {after['maximum']:.6g}\n"
        )

    def on_key_press(self, symbol, modifiers):
        if symbol == Key.SPACE:
            if self.iteration == len(self.weight_history) - 1:
                self.show_iteration(0)
            self.playing = not self.playing
            self.last_animation_step = time.perf_counter()
            self.print("Animation playing." if self.playing else "Animation paused.")
        elif symbol == Key.RIGHT:
            self.playing = False
            self.show_iteration(self.iteration + 1)
        elif symbol == Key.LEFT:
            self.playing = False
            self.show_iteration(self.iteration - 1)
        elif symbol == Key.HOME or symbol == Key.R:
            self.playing = False
            self.show_iteration(0)
        elif symbol == Key.END:
            self.playing = False
            self.show_iteration(len(self.weight_history) - 1)
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

    def on_idle(self):
        if not self.playing:
            return

        now = time.perf_counter()
        if now - self.last_animation_step < config.SECONDS_PER_ITERATION:
            return

        self.last_animation_step = now

        if self.iteration == len(self.weight_history) - 1:
            self.playing = False
            self.print("Animation finished.")
            return

        self.show_iteration(self.iteration + 1)

    def print_help(self):
        self.print(
            "\n"
            "================ BONUS CPD =================\n"
            f"Frames       : {self.source_frame} -> {self.target_frame}\n"
            "Red points   : fixed target frame\n"
            "Blue points  : moving source frame\n"
            "Gray points  : original source position\n"
            "SPACE        : play / pause animation\n"
            "LEFT / RIGHT : previous / next CPD iteration\n"
            "HOME or R    : return to iteration 0\n"
            "END          : show final registration\n"
            "E            : blue points / error heatmap\n"
            "A            : coordinate axes on/off\n"
            "?            : show this menu\n"
            "============================================\n"
        )
