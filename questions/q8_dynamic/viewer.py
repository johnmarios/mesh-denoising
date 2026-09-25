"""Small interactive viewer for Question 8 dynamic sequences."""

from pathlib import Path
import csv
import re
import time

import numpy as np

from vvrpywork.constants import Key
from vvrpywork.scene import Scene3D_
from vvrpywork.shapes import LineSet3D

import config as project_config
from core.visualization import gray_colors, set_axes_visible
from questions.q8_dynamic import config
from questions.q8_dynamic.evaluation import error_heatmap_colors, meshes_from_vertices, sequence_metrics, sequence_maximum_normal_error
from questions.q8_dynamic.sequence import DynamicMeshSession
from questions.q8_dynamic.spectral import (
    compute_graph_eigenvectors,
    denoise_dynamic_vertices,
)


NOISE_KEYS = {
    Key.G: "GAUSSIAN",
    Key.D: "NORMAL",
    Key.U: "UNIFORM",
    Key.I: "IMPULSE",
    Key.Q: "QUANTIZATION",
}


class DynamicMeshApp(Scene3D_):
    """Display one frame at a time while the complete sequence stays in memory."""

    def __init__(
        self,
        folder: str | Path,
        require_correspondence: bool = True,
        registered_folder: str | Path | None = None,
    ):
        folder = Path(folder)
        super().__init__(project_config.WIDTH, project_config.HEIGHT, f"Dynamic Mesh Denoising - {folder.name}", output=True, n_sliders=1,)

        self.input_has_correspondence = require_correspondence
        self.require_correspondence = require_correspondence
        self.raw_session = DynamicMeshSession(folder, require_correspondence)
        self.registered_folder = (
            None if registered_folder is None else Path(registered_folder)
        )
        self.registered_session = None
        self.session = self.raw_session
        self.selected_noise = "NORMAL"
        self.noise_level = config.DEFAULT_DYNAMIC_NOISE_LEVEL
        self.wireframe_visible = True
        self.axes_visible = True
        self.eigenvectors = None # eigenbasis cache 
        self.heatmap_maximum_error = None

        self.addShape(self.session.current, "mesh")
        self.wireframe = LineSet3D.create_from_mesh(self.session.current)
        self.addShape(self.wireframe, "wireframe")

        span = project_config.NOISE_LEVEL_MAX - project_config.NOISE_LEVEL_MIN
        slider_value = (self.noise_level - project_config.NOISE_LEVEL_MIN) / span
        self.set_slider_value(0, slider_value)

        self.print_help()
        self._print_frame()

    def _replace_mesh(self):
        self.removeShape("mesh")
        self.removeShape("wireframe")
        self.addShape(self.session.current, "mesh")

        if self.wireframe_visible:
            self.wireframe = LineSet3D.create_from_mesh(self.session.current)
            self.addShape(self.wireframe, "wireframe")

    def _print_frame(self):
        if self.require_correspondence:
            stage = "CORRESPONDENCE AVAILABLE"
        else:
            stage = "RAW REMESHED / NO CORRESPONDENCE"

        self.print(
            f"Q8 frame {self.session.frame_index + 1}/{self.session.frame_count} "
            f"| {stage} | view={self.session.mode}"
        )

    def on_slider_change(self, slider_id, value):
        if slider_id == 0:
            span = project_config.NOISE_LEVEL_MAX - project_config.NOISE_LEVEL_MIN
            self.noise_level = project_config.NOISE_LEVEL_MIN + span * value
            self.print(f"Dynamic noise alpha = {self.noise_level:.3f}")

    def on_key_press(self, symbol, modifiers):
        if symbol in NOISE_KEYS:
            self.selected_noise = NOISE_KEYS[symbol]
            self.print(f"Selected dynamic noise: {self.selected_noise}")
            return

        if symbol == Key.RIGHT:
            self.session.next_frame()
            self._replace_mesh()
            self._print_frame()
        elif symbol == Key.LEFT:
            self.session.previous_frame()
            self._replace_mesh()
            self._print_frame()
        elif symbol == Key._1:
            self.add_noise()
        elif symbol == Key._8:
            self.denoise_sequence()
        elif symbol == Key._2:
            self.print_metrics()
        elif symbol == Key.E:
            self.show_error_heatmap()
        elif symbol == Key.R:
            self.show_registered_sequence()
        elif symbol == Key.B:
            self.show_raw_sequence()
        elif symbol == Key.O:
            self.session.show_original()
            self._reset_colors()
        elif symbol == Key.N:
            if self.session.show_noisy():
                self._reset_colors()
            else:
                self.print("Create dynamic noise first with key 1.")
        elif symbol == Key.P:
            if self.session.show_denoised():
                self._reset_colors()
            else:
                self.print("Run Q8 spectral denoising first with key 8.")
        elif symbol == Key.C:
            self._reset_colors()
        elif symbol == Key.W:
            self.wireframe_visible = not self.wireframe_visible
            self._replace_mesh()
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

    def _reset_colors(self):
        self.session.current.vertex_colors = gray_colors(len(self.session.current.vertices))
        self._replace_mesh()
        self._print_frame()

    def add_noise(self):
        if not self.require_correspondence:
            self.print(
                "Noise is disabled because the remeshed frames do not have "
                "correspondence. Press R to load the CPD result first."
            )
            return

        self.session.set_noise(self.selected_noise, self.noise_level, config.NOISE_SEED,)
        self.heatmap_maximum_error = sequence_maximum_normal_error(self.session.original, self.session.noisy,)

        self._reset_colors()

        self.print(
            f"Q8: added {self.selected_noise} noise to all frames, "
            f"alpha={self.noise_level:.3f}."
        )
        self.print(
            "Heatmap common scale: "
            f"blue=0°, red={self.heatmap_maximum_error:.3f}°."
        )

    def denoise_sequence(self):
        if not self.require_correspondence:
            self.print(
                "Denoising is disabled because the remeshed frames do not "
                "have correspondence. Press R to load the CPD result first."
            )
            return

        if self.session.noisy is None:
            self.print("Create dynamic noise first with key 1.")
            return

        vertices_sequence = np.stack([np.asarray(frame.vertices, dtype=float) for frame in self.session.noisy])
        triangles = np.asarray(self.session.noisy[0].triangles, dtype=np.int64)

        if self.eigenvectors is None:
            self.print(
                "Q8: computing the graph eigenbasis. "
                "For this mesh it takes about 25 seconds..."
            )
            start_time = time.perf_counter()
            self.eigenvectors = compute_graph_eigenvectors(
                vertices_sequence[0],
                triangles,
            )
            elapsed = time.perf_counter() - start_time
            self.print(f"Q8: eigenbasis ready in {elapsed:.2f} seconds.")

        self.print("Q8: applying temporal low-rank denoising...")
        denoised_vertices = denoise_dynamic_vertices(vertices_sequence, triangles, eigenvectors=self.eigenvectors)
        denoised_frames = meshes_from_vertices(self.session.original, denoised_vertices)
        self.session.set_denoised(denoised_frames)
        self._reset_colors()

        noisy_metrics = sequence_metrics(
            self.session.original,
            self.session.noisy,
        )
        denoised_metrics = sequence_metrics(
            self.session.original,
            self.session.denoised,
        )

        self.print("Q8 finished. Mean sequence errors:")
        for name in ("MSAE_deg", "L2_surface", "Hausdorff", "RMSE"):
            self.print(
                f"{name:12s}: {noisy_metrics[name]:.6g} -> "
                f"{denoised_metrics[name]:.6g}"
            )

    def print_metrics(self):
        if self.session.mode == "original":
            self.print("Select noisy (N) or denoised (P) frames first.")
            return

        values = sequence_metrics(self.session.original, self.session.current_frames)
        self.print("Q8 mean sequence metrics:")
        for name, value in values.items():
            self.print(f"{name:14s}: {value:.6g}")

    def show_error_heatmap(self):
        if self.session.mode == "original":
            self.print("Error heatmap requires noisy or denoised view.")
            return

        reference = self.session.original[self.session.frame_index]
        current = self.session.current
        current.vertex_colors = error_heatmap_colors(reference, current, self.heatmap_maximum_error)
        self.removeShape("mesh")
        self.addShape(current, "mesh")
        self.print("Q8 angular normal-error heatmap shown for current frame.")

    def show_registered_sequence(self):
        """Replace raw remeshed input with the common-topology CPD result."""
        if self.input_has_correspondence:
            self.print("This sequence already has vertex correspondence.")
            return

        if self.registered_folder is None:
            self.print("No CPD-registered sequence was configured.")
            return

        if self.registered_session is None:
            if not self.registered_folder.exists():
                self.print(f"CPD result not found: {self.registered_folder}")
                return

            self.print("Loading the CPD-registered sequence...")
            try:
                self.registered_session = DynamicMeshSession(
                    self.registered_folder,
                    require_correspondence=True,
                )
            except (FileNotFoundError, ValueError) as error:
                self.print(f"Could not load the CPD sequence: {error}")
                self.registered_session = None
                return

            self._print_cpd_summary()

        current_index = self.session.frame_index
        self.session = self.registered_session
        self.session.frame_index = min(
            current_index,
            self.session.frame_count - 1,
        )
        self.require_correspondence = True
        self.eigenvectors = None
        self._replace_mesh()
        self._print_frame()
        self.print(
            "CPD correspondence loaded. Press 1 to add noise and then 8 "
            "to run temporal denoising."
        )

    def show_raw_sequence(self):
        """Return to the independently remeshed input sequence."""
        if self.input_has_correspondence:
            self.print("This Q8 run has no separate remeshed input.")
            return

        current_index = self.session.frame_index
        self.session = self.raw_session
        self.session.frame_index = min(
            current_index,
            self.session.frame_count - 1,
        )
        self.require_correspondence = False
        self.eigenvectors = None
        self._replace_mesh()
        self._print_frame()
        self.print("Raw remeshed frames shown. Press R to apply CPD correspondence.")

    def _print_cpd_summary(self):
        """Print aggregate before/after registration measurements."""
        metrics_path = self.registered_folder / "metrics.csv"
        if not metrics_path.exists():
            return

        try:
            with metrics_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

            before = np.array(
                [float(row["surface_mean_before"]) for row in rows]
            )
            after = np.array(
                [float(row["surface_mean_after"]) for row in rows]
            )
        except (OSError, KeyError, TypeError, ValueError) as error:
            self.print(f"Could not read CPD metrics: {error}")
            return

        if not rows:
            return

        improved = int(np.sum(after < before))
        self.print(
            "CPD mean point-to-surface error: "
            f"{np.mean(before):.6f} -> {np.mean(after):.6f}."
        )
        self.print(f"CPD improved {improved}/{len(rows)} target frames.")

    def print_help(self):
        self.print(
            "\n"
            "================ DYNAMIC MESH Q8 ================\n"
            "LEFT / RIGHT : previous / next frame\n"
            "G/D/U/I/Q    : choose dynamic noise type (D = Gaussian along normals)\n"
            "R            : load the CPD-registered sequence (Bonus)\n"
            "B            : return to raw remeshed frames (Bonus)\n"
            "1            : add selected noise to every frame\n"
            "8            : spectral dynamic denoising (paper Algorithm 1)\n"
            "2            : print mean Q2 metrics over the sequence\n"
            "E            : error heatmap on current frame\n"
            "O / N / P    : original / noisy / denoised sequence\n"
            "C            : return current frame to gray\n"
            "W            : wireframe on/off\n"
            "A            : coordinate axes on/off\n"
            "?            : show this menu\n"
            "Slider 1     : dynamic noise alpha\n"
            "Bonus order  : R, then 1, then 8\n"
            "=================================================\n"
        )
