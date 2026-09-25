import numpy as np

from vvrpywork.constants import Color, Key
from vvrpywork.scene import Scene3D_
from vvrpywork.shapes import LineSet3D, PointSet3D

import config
from core.geometry import average_edge_length
from core.mesh import MeshSession
from core.visualization import gray_colors, set_axes_visible
from questions import q1_noise, q2_metrics, q3_analysis, q4_smoothing, q5_feature_denoising, q7_pointnet

NOISE_KEYS = {
    Key.G: "GAUSSIAN",
    Key.D: "NORMAL",
    Key.U: "UNIFORM",
    Key.I: "IMPULSE",
    Key.Q: "QUANTIZATION",
}

Q4_BRANCHES = ("LAPLACIAN", "TAUBIN")
Q5_BRANCH = "FEATURE_AWARE"
Q7_BRANCH = "POINTNET"


class ProjectApp(Scene3D_):
    def __init__(self, model_name: str):
        super().__init__(
            config.WIDTH,
            config.HEIGHT,
            f"Mesh Denoising - {model_name}",
            output=True,
            n_sliders=2,
        )

        self.session = MeshSession(model_name)
        self.selected_noise = "GAUSSIAN"
        self.selected_denoiser = "LAPLACIAN"
        self.noise_level = config.DEFAULT_NOISE_LEVEL
        self.smoothing_strength = config.DEFAULT_SMOOTHING_STRENGTH
        self.wireframe_visible = True
        self.normals_visible = False
        self.candidates_visible = False 
        self.point_cloud_visible = False
        self.axes_visible = True

        self.q3_cache = None
        self.candidate_regions_cache = None
        self.delta_cache = None
        self.normal_lines_cache = None
        self.cuboids_cache = None
        self.point_cloud_cache = None

        self.pointnet_model = None
        self.pointnet_device = None
        self.pointnet_checkpoint_path = None

        self.addShape(self.session.current, "mesh")
        self.wireframe = LineSet3D.create_from_mesh(self.session.current)
        self.addShape(self.wireframe, "wireframe")

        self.set_slider_value(0, (self.noise_level - config.NOISE_LEVEL_MIN)/ (config.NOISE_LEVEL_MAX - config.NOISE_LEVEL_MIN),)
        self.set_slider_value(1, self.smoothing_strength / config.SMOOTHING_STRENGTH_MAX)
        self.print_startup_mesh_information()
        self.print_help()

    def print_startup_mesh_information(self):
        """Print the loaded mesh and noise scale used by Question 1."""
        vertices = np.asarray(self.session.original.vertices, dtype=float)
        triangles = np.asarray(self.session.original.triangles, dtype=np.int64)
        mean_edge_length = average_edge_length(vertices, triangles)
        sigma = self.noise_level * mean_edge_length

        print(
            "\n"
            "================ MESH INFORMATION ================\n"
            f"Model            : {self.session.model_name}\n"
            f"N (vertices)     : {len(vertices)}\n"
            f"F (faces)        : {len(triangles)}\n"
            f"l_mean           : {mean_edge_length:.12g}\n"
            f"alpha            : {self.noise_level:.6g}\n"
            f"sigma            : {sigma:.12g}\n"
            "==================================================\n"
        )

    def _invalidate_analysis_cache(self):
        """Call whenever the current mesh geometry changes."""
        self._hide_q3_overlays()
        self.q3_cache = None
        self.candidate_regions_cache = None
        self.delta_cache = None
        self.normal_lines_cache = None
        self.cuboids_cache = None
        self.point_cloud_cache = None
        self.wireframe = None

    def _hide_q3_overlays(self):
        """Remove all temporary overlays when the mesh geometry changes."""
        self._hide_normals()
        self._hide_candidate_regions()

    def _hide_normals(self):
        """Remove normal lines from the scene but keep their cached geometry."""
        self.removeShape("normals")
        self.normals_visible = False

    def _hide_candidate_regions(self):
        """Remove Q3 cuboids from the scene but keep them cached."""
        if self.cuboids_cache is not None:
            for index in range(len(self.cuboids_cache)):
                self.removeShape(f"q3_cuboid_{index}")

        self.candidates_visible = False

    def _get_q3_analysis(self):
        """Compute Q3 analysis only once for the current mesh."""

        if self.q3_cache is None:
            self.q3_cache = q3_analysis.analyze_q3(self.session.current, k=20,)

        return self.q3_cache

    def _get_candidate_regions(self):
        """Compute candidate connected regions only when needed."""

        if self.candidate_regions_cache is None:
            analysis = self._get_q3_analysis()
            self.candidate_regions_cache = (q3_analysis.analyze_candidate_regions(self.session.current, analysis, roughness_percentile=80.0, min_region_size=5,))

        return self.candidate_regions_cache

    def _show_heatmap(self, colors: np.ndarray, name: str):
        """Apply heatmap colors to the current mesh."""
        self._hide_candidate_regions()
        self.session.current.vertex_colors = colors
        self._refresh_mesh_colors()
        self.print(f"{name} heatmap shown.")

    def show_feature_heatmap(self):
        analysis = self._get_q3_analysis()
        colors = q3_analysis.show_feature_heatmap(analysis["vertex_feature_scores"])
        self._show_heatmap(colors, "Feature",)

    def show_roughness_heatmap(self):
        analysis = self._get_q3_analysis()
        colors = q3_analysis.show_roughness_heatmap(analysis["vertex_roughness"])
        self._show_heatmap(colors, "Roughness",)

    def show_delta_heatmap(self):
        if self.delta_cache is None:
            self.delta_cache = q3_analysis.delta_coordinate_values(
                self.session.current
            )
        colors = q3_analysis.show_delta_coordinates_heatmap(self.delta_cache)
        self._show_heatmap(colors, "Delta-coordinate",)
        
    def _gray(self):
        self.session.current.vertex_colors = gray_colors(len(self.session.current.vertices))

    def _replace_mesh(self):
        """Replace the displayed geometry after the current mesh changes."""

        self.removeShape("mesh")
        self.removeShape("wireframe")
        self.removeShape("point_cloud")

        if self.point_cloud_visible:
            self._show_point_cloud()
        else:
            self.addShape(self.session.current, "mesh")
            if self.wireframe_visible:
                if self.wireframe is None:
                    self.wireframe = LineSet3D.create_from_mesh(self.session.current)
                self.addShape(self.wireframe, "wireframe")

    def _show_point_cloud(self):
        if self.point_cloud_cache is None:
            self.point_cloud_cache = PointSet3D(
                np.asarray(self.session.current.vertices, dtype=float),
                size=2.0,
                color=Color.GRAY,
            )
            self._update_point_cloud_colors()
        self.addShape(self.point_cloud_cache, "point_cloud")

    def _update_point_cloud_colors(self):
        """Copy current mesh colors to the cached point-cloud object."""
        if self.point_cloud_cache is None:
            return

        colors = np.asarray(self.session.current.vertex_colors, dtype=float)
        alpha = np.ones((len(colors), 1), dtype=float)
        self.point_cloud_cache.colors = np.hstack((colors[:, :3], alpha))

    def _refresh_mesh_colors(self):
        """Update only colors, without rebuilding mesh edges or Q3 analysis."""
        if self.point_cloud_visible:
            self._update_point_cloud_colors()
            self.updateShape("point_cloud")
        else:
            self.updateShape("mesh")

    def _need_noisy(self) -> bool:
        if self.session.noisy is None:
            self.print("Q1 first: select G/D/U/I/Q and press 1.")
            return False
        return True

    def _display_changed_mesh(self):
        """Refresh display objects and caches after current geometry changes."""
        self._invalidate_analysis_cache()
        self._gray()
        self._replace_mesh()

    def select_denoiser(self, method: str):
        """Select the independent Laplacian or Taubin Q4 branch."""
        if method not in Q4_BRANCHES:
            raise ValueError(f"Unknown Q4 denoiser: {method}")

        self.selected_denoiser = method
        total = self.session.iterations.get(method, 0)
        self.print(
            f"Selected Q4 denoiser: {method}. Saved iterations: {total}."
        )

    def restore_original(self):
        self.session.restore_original()
        self._display_changed_mesh()
        self.print("Clean ground-truth mesh restored; saved results were kept.")

    def restore_noisy(self):
        if not self._need_noisy():
            return
        self.session.restore_noisy()
        self._display_changed_mesh()
        self.print("Saved noisy input restored.")

    def restore_result(self, branch: str, label: str, create_key: str):
        """Restore a saved result branch without modifying the other branches."""
        if not self._need_noisy():
            return
        if not self.session.restore_branch(branch):
            self.print(f"No saved {label} result. Press {create_key} to create it.")
            return

        self._display_changed_mesh()
        total = self.session.iterations.get(branch, 0)
        self.print(f"Restored {label} result at {total} total iterations.")

    def reset_experiment(self):
        self.session.reset()
        self._display_changed_mesh()
        self.print("Experiment reset; noise and all result branches were cleared.")

    def show_gray_mesh(self):
        self._hide_candidate_regions()
        self._gray()
        self._refresh_mesh_colors()
        self.print("Current mesh returned to gray.")

    def _todo(self, question: str, filename: str):
        self.print(f"{question} is not implemented yet: {filename}")


    def on_slider_change(self, slider_id, value):
        if slider_id == 0:
            span = config.NOISE_LEVEL_MAX - config.NOISE_LEVEL_MIN
            self.noise_level = config.NOISE_LEVEL_MIN + span * value
            self.print(f"Noise alpha = {self.noise_level:.3f}")
        elif slider_id == 1:
            self.smoothing_strength = config.SMOOTHING_STRENGTH_MAX * value
            self.print(f"Smoothing lambda = {self.smoothing_strength:.3f}")

    def on_key_press(self, symbol, modifiers):
        if symbol in NOISE_KEYS:
            self.selected_noise = NOISE_KEYS[symbol]
            self.print(f"Selected noise: {self.selected_noise}")
            return

        if symbol == Key.X:
            self.select_denoiser("LAPLACIAN")
            return
        if symbol == Key.T:
            self.select_denoiser("TAUBIN")
            return

        if symbol == Key._1:
            self.question_1()
        elif symbol == Key._2:
            self.question_2()
        elif symbol == Key._3:
            self.question_3()
        elif symbol == Key._4:
            self.question_4(bool(modifiers & Key.MOD_CTRL))
        elif symbol == Key._5:
            self.question_5(bool(modifiers & Key.MOD_CTRL))
        elif symbol == Key._7:
            self.question_7(bool(modifiers & Key.MOD_CTRL))
        elif symbol == Key.O:
            self.restore_original()
        elif symbol == Key.N:
            self.restore_noisy()
        elif symbol == Key.B:
            self.restore_result(self.selected_denoiser, self.selected_denoiser, "4",)
        elif symbol == Key.F:
            self.restore_result(Q5_BRANCH, "feature-aware", "5")
        elif symbol == Key.P:
            self.restore_result(Q7_BRANCH, "PointNet", "7")
        elif symbol == Key.R:
            self.reset_experiment()
        elif symbol == Key.W:
            self.toggle_wireframe()
        elif symbol == Key.A:
            self.toggle_axes()
        elif symbol == Key.V:
            self.toggle_point_cloud()
        elif symbol == Key.SLASH:
            self.print_help()
        elif symbol == Key.S:
            self.show_feature_heatmap()
        elif symbol == Key.H:
            self.show_roughness_heatmap()
        elif symbol == Key.L:
            self.show_delta_heatmap()
        elif symbol == Key.M:
            self.toggle_normals()
        elif symbol == Key.C:
            self.show_gray_mesh()
    
    # Question routing 

    def question_1(self):

        vertices = q1_noise.add_noise(
            self.session.original.vertices,
            self.session.original.triangles,
            self.session.original.vertex_normals,
            self.selected_noise,
            self.noise_level,
            config.DEFAULT_SEED,
        )

        self.session.set_noisy_vertices(vertices)
        self._display_changed_mesh()
        self.print(f"Q1: added {self.selected_noise} noise, alpha={self.noise_level:.3f}")

    def question_2(self):
        metrics = q2_metrics.compute_metrics(reference_mesh=self.session.original, current_mesh=self.session.current)
        output_metrics = q2_metrics.format_metrics(metrics)
        self.print(output_metrics)

    def question_3(self):
        if not self._need_noisy():
            return

        # Question 3 always analyses the saved noisy input, never the original
        # or a denoised result branch.
        if not self.session.current_is_noisy:
            self.session.restore_noisy()
            self._display_changed_mesh()

        analysis = self._get_q3_analysis()
        colors = q3_analysis.show_candidate_heatmap(analysis["vertex_candidate_scores"])
        self._show_heatmap(colors, "Candidate",)

        region_analysis = self._get_candidate_regions()
        if self.cuboids_cache is None:
            self.cuboids_cache = q3_analysis.show_denoising_candidate_regions(
                region_analysis["aabbs"]
            )

        for index, cuboid in enumerate(self.cuboids_cache):
            self.addShape(cuboid, f"q3_cuboid_{index}")

        self.candidates_visible = True
        candidate_count = int(np.sum(region_analysis["candidate_mask"]))
        region_count = len(region_analysis["regions"])
        self.print(
            "Q3 noisy input: "
            f"{candidate_count} candidate faces in {region_count} regions."
        )

    def question_4(self, restart: bool):
        if not self._need_noisy(): return

        source = self.session.start_branch(self.selected_denoiser, restart)
        if source is None: return 

        if self.selected_denoiser == "LAPLACIAN":
            vertices = q4_smoothing.laplacian_smoothing(source, self.smoothing_strength, config.ITERATIONS_PER_PRESS,)
        else:
            vertices = q4_smoothing.taubin_smoothing(source, self.smoothing_strength, config.ITERATIONS_PER_PRESS,)

        self.session.save_branch(
            self.selected_denoiser,
            vertices,
            added_iterations=config.ITERATIONS_PER_PRESS,
            restart=restart,
        )
        self._display_changed_mesh()

        total = self.session.iterations[self.selected_denoiser]
        action = "restarted" if restart else "continued"
        self.print(
            f"Q4 {self.selected_denoiser} {action}: "
            f"added {config.ITERATIONS_PER_PRESS}, total {total}, "
            f"lambda={self.smoothing_strength:.4f}."
        )

    def question_5(self, restart: bool):
        if not self._need_noisy():
            return

        source = self.session.start_branch(Q5_BRANCH, restart)
        if source is None:
            return

        vertices, vertex_feature_scores = (q5_feature_denoising.feature_aware_denoising(source.vertices, source.triangles, strength=self.smoothing_strength, iterations=config.ITERATIONS_PER_PRESS,))

        self.session.save_branch(
            Q5_BRANCH,
            vertices,
            added_iterations=config.ITERATIONS_PER_PRESS,
            restart=restart,
        )
        self._display_changed_mesh()

        total = self.session.iterations[Q5_BRANCH]
        action = "restarted" if restart else "continued"
        feature_vertices = int(np.sum(vertex_feature_scores >= 0.5))
        self.print(
            f"Q5 feature-aware {action}: "
            f"added {config.ITERATIONS_PER_PRESS}, total {total}, "
            f"step={self.smoothing_strength:.4f}, "
            f"feature vertices={feature_vertices}."
        )

    def question_7(self, restart: bool):
        if not self._need_noisy():
            return

        from questions.q7_pointnet import config as pointnet_config
        from questions.q7_pointnet import inference as pointnet_inference
        from questions.q7_pointnet.checkpoint import checkpoint_path

        selected_checkpoint = checkpoint_path(pointnet_config.OUTPUT_DIR, pointnet_config.INFERENCE_CHECKPOINT,)
        if selected_checkpoint is None:
            self.print(
                "Unknown PointNet checkpoint choice: "
                f"{pointnet_config.INFERENCE_CHECKPOINT}."
            )
            return

        selected_checkpoint = selected_checkpoint.resolve()
        if not selected_checkpoint.exists():
            self.print(f"PointNet checkpoint not found: {selected_checkpoint}")
            self.print("Train it first with: python project.py --train-pointnet")
            return

        if (
            self.pointnet_model is None
            or self.pointnet_checkpoint_path != selected_checkpoint
        ):
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                model = pointnet_inference.load_trained_model(
                    output_directory=str(pointnet_config.OUTPUT_DIR),
                    checkpoint_choice=pointnet_config.INFERENCE_CHECKPOINT,
                    device=device,
                )
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                self.print(f"Could not load PointNet checkpoint: {error}")
                return

            self.pointnet_model = model
            self.pointnet_device = device
            self.pointnet_checkpoint_path = selected_checkpoint
            self.print(
                f"Loaded PointNet checkpoint on {device}: {selected_checkpoint}"
            )

        source = self.session.start_branch(Q7_BRANCH, restart)
        if source is None:
            return

        try:
            vertices = pointnet_inference.denoise_mesh_with_model(
                mesh=source,
                model=self.pointnet_model,
                device=self.pointnet_device,
                k=pointnet_config.PATCH_SIZE,
            )
        except (RuntimeError, ValueError) as error:
            self.print(f"PointNet inference failed: {error}")
            return

        self.session.save_branch(
            Q7_BRANCH,
            vertices,
            added_iterations=1,
            restart=restart,
        )
        self._display_changed_mesh()

        total = self.session.iterations[Q7_BRANCH]
        action = "restarted" if restart else "continued"
        total_label = "pass" if total == 1 else "passes"
        self.print(
            f"Q7 PointNet {action}: added 1 pass, total {total} {total_label}."
        )
        
  
    # Display-only 

    def toggle_normals(self):
        """Show/hide cached vertex-normal lines for the current mesh."""
        if self.normals_visible:
            self._hide_normals()
            self.print("Normals hidden.")
            return

        if self.normal_lines_cache is None:
            self.normal_lines_cache = q3_analysis.show_normals(
                current_mesh=self.session.current
            )
        self.addShape(self.normal_lines_cache, "normals")
        self.normals_visible = True
        self.print("Normals shown.")

    def toggle_wireframe(self):
        self.wireframe_visible = not self.wireframe_visible
        if not self.point_cloud_visible:
            if self.wireframe_visible:
                if self.wireframe is None:
                    self.wireframe = LineSet3D.create_from_mesh(self.session.current)
                self.addShape(self.wireframe, "wireframe")
            else:
                self.removeShape("wireframe")
        state = "enabled" if self.wireframe_visible else "disabled"
        self.print(f"Wireframe {state}.")

    def toggle_point_cloud(self):
        self.point_cloud_visible = not self.point_cloud_visible
        self._replace_mesh()
        mode = "point cloud" if self.point_cloud_visible else "surface mesh"
        self.print(f"View: {mode}.")

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
            "================ MESH DENOISING MENU ================\n"
            "Noise selection for Question 1:\n"
            "G : Select Gaussian noise\n"
            "D : Select normal-direction noise\n"
            "U : Select uniform noise\n"
            "I : Select impulse noise\n"
            "Q : Select quantization noise\n"
            "1 : Add selected noise; clear old denoising branches\n"
            "\n"
            "Project questions:\n"
            "2 : Print metrics for the mesh currently on screen\n"
            "3 : Restore noisy input and show candidate regions\n"
            "X : Select Laplacian Q4 branch\n"
            "T : Select Taubin Q4 branch\n"
            "4 : Add 10 iterations to selected Q4 branch\n"
            "CTRL+4 : Restart selected Q4 branch from noisy, then add 10\n"
            "5 : Add 10 iterations to Q5 feature-aware branch\n"
            "CTRL+5 : Restart Q5 branch from noisy, then add 10\n"
            "7 : Add one PointNet pass to the Q7 branch\n"
            "CTRL+7 : Restart Q7 from noisy, then apply one PointNet pass\n"
            "\n"
            "Restore saved states:\n"
            "O : Show clean original / ground truth without clearing results\n"
            "N : Show saved noisy input\n"
            "B : Show saved selected Q4 baseline branch\n"
            "F : Show saved Q5 feature-aware branch\n"
            "P : Show saved Q7 PointNet branch\n"
            "\n"
            "Display and reset:\n"
            "V : Toggle mesh / vertex point-cloud view\n"
            "S : Show feature heatmap of current mesh\n"
            "H : Show roughness heatmap of current mesh\n"
            "L : Show delta heatmap of current mesh\n"
            "C : Return current mesh to gray\n"
            "W : Toggle wireframe\n"
            "A : Toggle coordinate axes\n"
            "M : show normal lines on/off\n"
            "R : Reset clean mesh and clear the whole experiment\n"
            "? : Show this menu\n"
            "\n"
            "Slider 1: alpha in noise = alpha * average edge [0.02, 0.20]\n"
            "Slider 2: smoothing lambda [0, 0.80]\n"
            "=======================================================\n"
        )
