"""Create one common-topology sequence from independently remeshed frames."""

import csv
from pathlib import Path
import time

import numpy as np

from questions.bonus_cpd import config
from questions.bonus_cpd.cpd import (
    nonrigid_cpd,
    symmetric_chamfer,
    transform_points,
)
from questions.bonus_cpd.evaluation import (
    create_target_surface,
    point_to_surface_distances,
)
from questions.bonus_cpd.sampling import (
    sample_mesh_surface,
    sample_mesh_surface_evenly,
)
from questions.q8_dynamic.sequence import load_frames


METRIC_FIELDS = (
    "canonical_frame",
    "target_frame",
    "chamfer_before",
    "chamfer_after",
    "surface_mean_before",
    "surface_mean_after",
    "edge_distortion_mean",
    "edge_distortion_p95",
    "cpd_iterations",
    "initial_sigma2",
    "final_sigma2",
    "seconds",
)


def save_mesh(vertices, triangles, output_path: str | Path) -> None:
    """Save vertices and one-based triangle indices as a simple OBJ file."""
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as file:
        for x, y, z in np.asarray(vertices, dtype=float):
            file.write(f"v {x:.10g} {y:.10g} {z:.10g}\n")
        for a, b, c in np.asarray(triangles, dtype=np.int64) + 1:
            file.write(f"f {a} {b} {c}\n")


def unique_edges(triangles) -> np.ndarray:
    """Return every undirected mesh edge once."""
    triangles = np.asarray(triangles, dtype=np.int64)
    edges = np.concatenate(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ),
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def write_metrics(rows, output_path: str | Path) -> None:
    """Write one comparable registration row for every target frame."""
    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def register_canonical_sequence(
    input_folder: str | Path,
    output_folder: str | Path,
) -> None:
    """Register remeshed frame 0 independently onto all remaining frames."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    frames = load_frames(input_folder, require_correspondence=False)
    canonical_mesh = frames[0]
    canonical_vertices = np.asarray(canonical_mesh.vertices, dtype=float)
    canonical_triangles = np.asarray(canonical_mesh.triangles, dtype=np.int64)

    source_control = sample_mesh_surface_evenly(
        canonical_vertices,
        canonical_triangles,
        config.SEQUENCE_CONTROL_POINTS,
        config.RANDOM_SEED,
        config.CONTROL_CANDIDATE_MULTIPLIER,
    )
    source_evaluation = sample_mesh_surface(
        canonical_vertices,
        canonical_triangles,
        config.EVALUATION_POINTS,
        config.RANDOM_SEED + 1,
    )

    edges = unique_edges(canonical_triangles)
    original_edge_lengths = np.linalg.norm(
        canonical_vertices[edges[:, 0]] - canonical_vertices[edges[:, 1]],
        axis=1,
    )

    save_mesh(
        canonical_vertices,
        canonical_triangles,
        output_folder / "mesh_0000.obj",
    )

    rows = []
    number_of_targets = len(frames) - 1

    for frame_index, target_mesh in enumerate(frames[1:], start=1):
        start_time = time.perf_counter()
        target_vertices = np.asarray(target_mesh.vertices, dtype=float)
        target_triangles = np.asarray(target_mesh.triangles, dtype=np.int64)

        target_control = sample_mesh_surface_evenly(
            target_vertices,
            target_triangles,
            config.SEQUENCE_CONTROL_POINTS,
            config.RANDOM_SEED + 10 * frame_index,
            config.CONTROL_CANDIDATE_MULTIPLIER,
        )
        target_evaluation = sample_mesh_surface(
            target_vertices,
            target_triangles,
            config.EVALUATION_POINTS,
            config.RANDOM_SEED + 10 * frame_index + 1,
        )

        result = nonrigid_cpd(
            target_points=target_control,
            moving_points=source_control,
            beta=config.SEQUENCE_BETA,
            regularization=config.SEQUENCE_REGULARIZATION,
            outlier_weight=config.OUTLIER_WEIGHT,
            max_iterations=config.MAX_ITERATIONS,
            tolerance=config.TOLERANCE,
        )

        registered_vertices = transform_points(
            canonical_vertices,
            source_control,
            result.weights,
            config.SEQUENCE_BETA,
        )
        registered_evaluation = transform_points(
            source_evaluation,
            source_control,
            result.weights,
            config.SEQUENCE_BETA,
        )

        target_surface = create_target_surface(
            target_vertices,
            target_triangles,
        )
        surface_before = point_to_surface_distances(
            canonical_vertices,
            target_surface,
        )
        surface_after = point_to_surface_distances(
            registered_vertices,
            target_surface,
        )

        registered_edge_lengths = np.linalg.norm(
            registered_vertices[edges[:, 0]]
            - registered_vertices[edges[:, 1]],
            axis=1,
        )
        edge_ratios = np.maximum(registered_edge_lengths, 1e-12) / np.maximum(
            original_edge_lengths,
            1e-12,
        )
        edge_distortion = np.abs(np.log(edge_ratios))

        save_mesh(
            registered_vertices,
            canonical_triangles,
            output_folder / f"mesh_{frame_index:04d}.obj",
        )

        row = {
            "canonical_frame": 0,
            "target_frame": frame_index,
            "chamfer_before": symmetric_chamfer(
                source_evaluation,
                target_evaluation,
            ),
            "chamfer_after": symmetric_chamfer(
                registered_evaluation,
                target_evaluation,
            ),
            "surface_mean_before": float(np.mean(surface_before)),
            "surface_mean_after": float(np.mean(surface_after)),
            "edge_distortion_mean": float(np.mean(edge_distortion)),
            "edge_distortion_p95": float(np.percentile(edge_distortion, 95.0)),
            "cpd_iterations": len(result.weight_history) - 1,
            "initial_sigma2": result.variance_history[0],
            "final_sigma2": result.variance_history[-1],
            "seconds": time.perf_counter() - start_time,
        }
        rows.append(row)
        write_metrics(rows, output_folder / "metrics.csv")

        print(
            f"CPD frame {frame_index}/{number_of_targets}: "
            f"Chamfer {row['chamfer_before']:.6f} -> "
            f"{row['chamfer_after']:.6f}"
        )

    print(f"Registered sequence saved in: {output_folder}")
