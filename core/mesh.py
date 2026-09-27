from pathlib import Path
import shutil
import tempfile

import numpy as np

from vvrpywork.constants import Color
from vvrpywork.shapes import Mesh3D

from config import RESOURCE_DIR


MODEL_SOURCES = {
    "BUNNY": None,
    "ARMADILLO": None,
    "BUNNY_LOW": RESOURCE_DIR / "bunny_low.obj",
    "CUBE": RESOURCE_DIR / "cube.obj",
    "ICOSAHEDRON": RESOURCE_DIR / "icosahedron.obj",
    "POLYHEDRON": RESOURCE_DIR / "polyhedron.obj",
    "TEAPOT": RESOURCE_DIR / "teapotMultiMesh.obj",
    "SUZANNE": RESOURCE_DIR / "suzanne_clean.obj",
    "DRAGON": RESOURCE_DIR / "dragon_low_low.obj",
    "DOLPHIN": RESOURCE_DIR / "dolphin.obj",
    "HAND": RESOURCE_DIR / "hand_clean.obj",
    "FLASHLIGHT": RESOURCE_DIR / "flashlight.obj",
    "PHONE": RESOURCE_DIR / "Phone_v02.obj",
    "UNICORN": RESOURCE_DIR / "unicorn_low.obj",
    "BUNNY_Q6_REFERENCE": RESOURCE_DIR / "q6" / "bunny_low_normalized.obj",
    "BUNNY_Q6_REMESHED": RESOURCE_DIR / "q6" / "bunny_low_normalized_remeshed.obj",
    "BUNNY_Q6_HOLES": RESOURCE_DIR / "q6" / "bunny_low_normalized_holes.obj",
}

MODEL_NAMES = tuple(MODEL_SOURCES)

# These meshes were all created from the same normalized Question 6 reference.
# They must keep that common coordinate system, so MeshSession must not center
# and scale each one independently when it is loaded.
PRENORMALIZED_MODEL_NAMES = {
    "BUNNY_Q6_REFERENCE",
    "BUNNY_Q6_REMESHED",
    "BUNNY_Q6_HOLES",
}


def vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.int64) # (n_triangles, 3)

    tri = np.asarray(vertices, dtype=float)[triangles] # (n_triangles, 3, 3)
    face_vectors = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])

    normals = np.zeros_like(vertices, dtype=float) #area weighted vertex normals
    # at each face vector is added in the three vertices of the triangle
    np.add.at(normals, triangles[:, 0], face_vectors)
    np.add.at(normals, triangles[:, 1], face_vectors)
    np.add.at(normals, triangles[:, 2], face_vectors)

    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    return normals


def copy_mesh(mesh: Mesh3D) -> Mesh3D:
    copied = Mesh3D()
    copied.vertices = np.array(mesh.vertices, dtype=float, copy=True)
    copied.triangles = np.array(mesh.triangles, dtype=np.int64, copy=True)
    copied.color = Color.GRAY
    copied.vertex_normals = np.array(mesh.vertex_normals, dtype=float, copy=True)
    copied.vertex_colors = np.full((len(copied.vertices), 3), 0.65)
    return copied


def load_mesh_from_path(path: str | Path) -> Mesh3D:
    """Load one mesh file without changing its topology or vertex indices."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(suffix=path.suffix, delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        shutil.copyfile(path, temp_path)
        return Mesh3D(str(temp_path), color=Color.GRAY)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def create_mesh(name: str) -> Mesh3D:
    name = name.upper()
    if name == "BUNNY":
        return Mesh3D.create_bunny(color=Color.GRAY)
    if name == "ARMADILLO":
        return Mesh3D.create_armadillo(color=Color.GRAY)

    path = MODEL_SOURCES.get(name)
    if path is None:
        raise ValueError(f"Unknown model: {name}")
    return load_mesh_from_path(path)


def prepare_mesh(mesh: Mesh3D, normalize: bool = True) -> Mesh3D:
    """Clean a mesh and optionally center it inside the unit sphere."""
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    if normalize:
        vertices -= vertices.mean(axis=0)
        radius = np.max(np.linalg.norm(vertices, axis=1))
        if radius > 0.0:
            vertices /= radius

    mesh.vertices = vertices
    mesh.triangles = np.asarray(mesh.triangles, dtype=np.int64)
    mesh.vertex_normals = vertex_normals(mesh.vertices, mesh.triangles)
    mesh.vertex_colors = np.full((len(vertices), 3), 0.65)
    return mesh


class MeshSession:
    """Keep clean/noisy/current meshes and named independent result branches."""

    def __init__(self, model_name: str):
        self.model_name = model_name.upper()
        normalize = self.model_name not in PRENORMALIZED_MODEL_NAMES
        self.original = prepare_mesh(create_mesh(self.model_name), normalize=normalize)
        self.noisy = None
        self.current = copy_mesh(self.original)
        self.current_source = "original"
        self.results: dict[str, Mesh3D] = {}
        self.iterations: dict[str, int] = {}

    @property
    def current_is_noisy(self) -> bool:
        """Whether the current mesh is the stored noisy input."""
        return self.current_source == "noisy"

    def reset(self) -> None:
        self.noisy = None
        self.current = copy_mesh(self.original)
        self.current_source = "original"
        self.results.clear()
        self.iterations.clear()

    def set_noisy_vertices(self, vertices: np.ndarray) -> None:
        noisy = copy_mesh(self.original)
        noisy.vertices = np.asarray(vertices, dtype=float)
        noisy.vertex_normals = vertex_normals(noisy.vertices, noisy.triangles)
        self.noisy = copy_mesh(noisy)
        self.current = copy_mesh(noisy)
        self.current_source = "noisy"
        self.results.clear()
        self.iterations.clear()

    def restore_original(self) -> None:
        self.current = copy_mesh(self.original)
        self.current_source = "original"

    def restore_noisy(self) -> bool:
        if self.noisy is None:
            return False
        self.current = copy_mesh(self.noisy)
        self.current_source = "noisy"
        return True

    def start_branch(self, name: str, restart: bool) -> Mesh3D | None:
        if self.noisy is None:
            return None
        if restart or name not in self.results:
            return copy_mesh(self.noisy)
        return copy_mesh(self.results[name])

    def save_branch(self, name: str, vertices: np.ndarray, added_iterations: int = 0, restart: bool = False) -> None:
        if self.noisy is None:
            raise RuntimeError("A noisy mesh is required before saving a denoising branch.")

        result = copy_mesh(self.noisy)
        result.vertices = np.asarray(vertices, dtype=float)
        result.vertex_normals = vertex_normals(result.vertices, result.triangles)
        self.results[name] = copy_mesh(result)
        self.current = copy_mesh(result)
        self.current_source = f"result:{name}"

        if restart:
            self.iterations[name] = added_iterations
        else:
            self.iterations[name] = self.iterations.get(name, 0) + added_iterations

    def restore_branch(self, name: str) -> bool:
        result = self.results.get(name)
        if result is None:
            return False
        self.current = copy_mesh(result)
        self.current_source = f"result:{name}"
        return True
