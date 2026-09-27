import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from vvrpywork.shapes import Mesh3D

import config
from core.mesh import vertex_normals
from core.visualization import gray_colors
from questions.q8_dynamic.sequence import load_frames
from remesh_config import (
    MIN_FACE_RATIO,
    MAX_FACE_RATIO,
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


# save
def save_mesh(mesh: Mesh3D, path: Path,) -> None:
    """Save a vvrpywork Mesh3D as OBJ."""
    path.parent.mkdir(parents=True, exist_ok=True,)
    open3d_mesh = to_open3d(mesh)

    success = o3d.io.write_triangle_mesh(str(path), open3d_mesh, write_ascii=True,)

    if not success:
        raise OSError(f"Could not save {path}")


# dataset creation
def create_no_correspondence_sequence(model_name: str) -> None:
    """
    Create independently face-reduced frames.
    Every frame gets a different target triangle count.
    Therefore, vertex correspondence between frames is lost.
    """
    input_folder = config.question_8_dynamic_folder(model_name)
    output_folder = (
        config.RESOURCE_DIR / "dynamic" / f"{model_name}_nocorr"
    )
    frames = load_frames(input_folder)

    if MAX_FRAMES is not None:
        frames = frames[:MAX_FRAMES]

    clean_output = output_folder / "clean_remeshed"

    rng = np.random.default_rng(RANDOM_SEED)

    for frame_index, original_mesh in enumerate(frames):
        # Different ratio for every frame.
        face_ratio = rng.uniform(MIN_FACE_RATIO, MAX_FACE_RATIO,)
        clean_remeshed = reduce_mesh(original_mesh, face_ratio,)

        filename = f"mesh_{frame_index:04d}.obj"

        save_mesh(clean_remeshed, clean_output / filename,)

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


def parse_model_name() -> str:
    parser = argparse.ArgumentParser(
        description="Create an independently remeshed dynamic sequence"
    )
    parser.add_argument(
        "model",
        choices=config.DYNAMIC_MESH_NAMES,
        help="dynamic mesh sequence to remesh",
    )
    return parser.parse_args().model


if __name__ == "__main__":
    create_no_correspondence_sequence(parse_model_name())
