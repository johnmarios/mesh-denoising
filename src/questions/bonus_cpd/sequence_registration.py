"""Create one common-topology sequence from independently remeshed frames."""

from pathlib import Path

import numpy as np

from questions.bonus_cpd import config
from questions.bonus_cpd.cpd import (
    nonrigid_cpd,
    transform_points,
)
from questions.bonus_cpd.sampling import (
    sample_mesh_surface_evenly,
)
from questions.q8_dynamic.sequence import load_frames


def save_mesh(vertices, triangles, output_path: str | Path) -> None:
    """Save vertices and one-based triangle indices as a simple OBJ file."""
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as file:
        for x, y, z in np.asarray(vertices, dtype=float):
            file.write(f"v {x:.10g} {y:.10g} {z:.10g}\n")
        for a, b, c in np.asarray(triangles, dtype=np.int64) + 1:
            file.write(f"f {a} {b} {c}\n")


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
    save_mesh(
        canonical_vertices,
        canonical_triangles,
        output_folder / "mesh_0000.obj",
    )

    number_of_targets = len(frames) - 1

    for frame_index, target_mesh in enumerate(frames[1:], start=1):
        target_vertices = np.asarray(target_mesh.vertices, dtype=float)
        target_triangles = np.asarray(target_mesh.triangles, dtype=np.int64)

        target_control = sample_mesh_surface_evenly(
            target_vertices,
            target_triangles,
            config.SEQUENCE_CONTROL_POINTS,
            config.RANDOM_SEED + 10 * frame_index,
            config.CONTROL_CANDIDATE_MULTIPLIER,
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
        save_mesh(
            registered_vertices,
            canonical_triangles,
            output_folder / f"mesh_{frame_index:04d}.obj",
        )

        print(f"CPD frame {frame_index}/{number_of_targets} saved.")

    print(f"Registered sequence saved in: {output_folder}")
