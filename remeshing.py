from pathlib import Path

import numpy as np
import open3d as o3d

from vvrpywork.shapes import Mesh3D

from core.mesh import copy_mesh, vertex_normals
from core.visualization import gray_colors
from questions import q1_noise
from questions.q8_dynamic.sequence import load_frames
from remesh_config import (
    INPUT_FOLDER,
    OUTPUT_FOLDER,
    MIN_FACE_RATIO,
    MAX_FACE_RATIO,
    NOISE_TYPE,
    NOISE_LEVEL,
    RANDOM_SEED,
    MAX_FRAMES
)


# convertion Mesh3D <-> Open3D
def to_open3d(mesh: Mesh3D) -> o3d.geometry.TriangleMesh:
    """Convert a vvrpywork Mesh3D to an Open3D mesh."""
    result = o3d.geometry.TriangleMesh()

    result.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    result.triangles = o3d.utility.Vector3iVector(mesh.triangles)

    result.compute_vertex_normals()

    return result


def from_open3d(mesh: o3d.geometry.TriangleMesh,) -> Mesh3D:
    """Convert an Open3D mesh to a vvrpywork Mesh3D."""
    result = Mesh3D()

    result.vertices = np.asarray(mesh.vertices, dtype=float,)
    result.triangles = np.asarray(mesh.triangles, dtype=np.int64,)

    result.vertex_normals = vertex_normals(result.vertices, result.triangles,)
    result.vertex_colors = gray_colors(len(result.vertices))

    return result


# remeshing
def reduce_mesh(mesh: Mesh3D, face_ratio: float,) -> Mesh3D:
    """
    Reduce the number of triangles.

    face_ratio=0.70 keeps approximately 70% of the original triangles.
    """
    open3d_mesh = to_open3d(mesh)
    original_face_count = len(mesh.triangles)
    target_face_count = int(face_ratio * original_face_count)

    simplified = (open3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_face_count))

    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_duplicated_vertices()
    simplified.remove_unreferenced_vertices()
    simplified.compute_vertex_normals()

    return from_open3d(simplified)


# noise addition
def add_noise(clean_mesh: Mesh3D, seed: int,) -> Mesh3D:
    """Add noise to clean mesh. """
    noisy_mesh = copy_mesh(clean_mesh)

    noisy_mesh.vertices = q1_noise.add_noise(
        vertices=clean_mesh.vertices,
        triangles=clean_mesh.triangles,
        vertex_normals=clean_mesh.vertex_normals,
        noise_type=NOISE_TYPE,
        level=NOISE_LEVEL,
        seed=seed,
    )

    noisy_mesh.vertex_normals = vertex_normals(noisy_mesh.vertices, noisy_mesh.triangles,)

    return noisy_mesh


# save
def save_mesh(mesh: Mesh3D, path: Path,) -> None:
    """Save a vvrpywork Mesh3D as OBJ."""
    path.parent.mkdir(parents=True, exist_ok=True,)
    open3d_mesh = to_open3d(mesh)

    success = o3d.io.write_triangle_mesh(str(path), open3d_mesh, write_ascii=True,)

    if not success:
        raise OSError(f"Could not save {path}")


# dataset creation
def create_no_correspondence_sequence() -> None:
    """
    Create independently face reduced and noisy frames. 
    Every frame gets a different target triangle count.
    Therefore, vertex correspondence between frames is lost.
    """
    frames = load_frames(INPUT_FOLDER)

    if MAX_FRAMES is not None:
        frames = frames[:MAX_FRAMES]

    clean_output = OUTPUT_FOLDER / "clean_remeshed"
    noisy_output = OUTPUT_FOLDER / "noisy"

    rng = np.random.default_rng(RANDOM_SEED)

    for frame_index, original_mesh in enumerate(frames):
        # Different ratio for every frame.
        face_ratio = rng.uniform(MIN_FACE_RATIO, MAX_FACE_RATIO,)
        clean_remeshed = reduce_mesh(original_mesh, face_ratio,)

        noisy_remeshed = add_noise(clean_remeshed, seed=RANDOM_SEED + frame_index,)

        filename = f"mesh_{frame_index:04d}.obj"

        save_mesh(clean_remeshed, clean_output / filename,)

        save_mesh(noisy_remeshed, noisy_output / filename,)

        print(
            f"Frame {frame_index:03d}: "
            f"vertices {len(original_mesh.vertices)} "
            f"-> {len(clean_remeshed.vertices)}, "
            f"faces {len(original_mesh.triangles)} "
            f"-> {len(clean_remeshed.triangles)}"
        )

    print()
    print("Dataset creation completed.")
    print(f"Clean meshes: {clean_output}")
    print(f"Noisy input:  {noisy_output}")


if __name__ == "__main__":
    create_no_correspondence_sequence()