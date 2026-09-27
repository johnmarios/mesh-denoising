import numpy as np
from core.geometry import delta_coordinates_sparse
from vvrpywork.shapes import Mesh3D


def laplacian_smoothing(mesh: Mesh3D, strength: float, iterations: int,) -> np.ndarray:

    result = np.array(mesh.vertices, dtype=float, copy=True)
    triangles = np.array(mesh.triangles, dtype=np.int64)
    for _ in range(iterations):
        delta = delta_coordinates_sparse(result, triangles)
        result = result - strength * delta

    return result


def taubin_smoothing(mesh: Mesh3D, strength: float, iterations: int, mu_ratio: float = -1.06,) -> np.ndarray:

    result = np.array(mesh.vertices, dtype=float, copy=True)
    triangles = np.array(mesh.triangles, dtype=np.int64)
    mu = mu_ratio * strength

    for _ in range(iterations):
        delta = delta_coordinates_sparse(result, triangles)
        result = result - strength * delta
        delta = delta_coordinates_sparse(result, triangles)
        result = result - mu * delta

    return result
