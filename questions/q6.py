"""Create the normalized reference mesh used in Question 6."""

from pathlib import Path
import sys

import numpy as np


# Allow this file to be executed directly with: python questions/q6.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry import average_edge_length
from core.mesh import load_mesh_from_path, prepare_mesh


INPUT_PATH = PROJECT_ROOT / "resources" / "bunny_low.obj"
OUTPUT_PATH = PROJECT_ROOT / "resources" / "q6" / "bunny_low_normalized.obj"


def save_obj(vertices: np.ndarray, triangles: np.ndarray, output_path: Path) -> None:
    """Save vertices and triangular faces in the simple OBJ format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as obj_file:
        obj_file.write("# Normalized reference mesh for Question 6\n")

        for x, y, z in vertices:
            obj_file.write(f"v {x:.12g} {y:.12g} {z:.12g}\n")

        # OBJ indices start at 1, whereas NumPy indices start at 0.
        for i, j, k in triangles:
            obj_file.write(f"f {i + 1} {j + 1} {k + 1}\n")


def print_mesh_information(label: str, vertices: np.ndarray, triangles: np.ndarray) -> None:
    center = np.mean(vertices, axis=0)
    maximum_radius = np.max(np.linalg.norm(vertices - center, axis=1))
    mean_edge_length = average_edge_length(vertices, triangles)

    print(label)
    print(f"  vertices          : {len(vertices)}")
    print(f"  faces             : {len(triangles)}")
    print(f"  centroid          : {center}")
    print(f"  maximum radius    : {maximum_radius:.9f}")
    print(f"  mean edge length  : {mean_edge_length:.9f}")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input mesh was not found: {INPUT_PATH}")

    mesh = load_mesh_from_path(INPUT_PATH)

    original_vertices = np.asarray(mesh.vertices, dtype=float)
    original_triangles = np.asarray(mesh.triangles, dtype=np.int64)
    print_mesh_information("Original bunny_low.obj", original_vertices, original_triangles)

    normalized_mesh = prepare_mesh(mesh)
    normalized_vertices = np.asarray(normalized_mesh.vertices, dtype=float)
    normalized_triangles = np.asarray(normalized_mesh.triangles, dtype=np.int64)

    save_obj(normalized_vertices, normalized_triangles, OUTPUT_PATH)
    print_mesh_information("\nNormalized Question 6 reference", normalized_vertices, normalized_triangles)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
