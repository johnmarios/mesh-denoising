import numpy as np
from matplotlib import colormaps as cm

from core.mesh import copy_mesh, vertex_normals


def vertex_normal_angle_error(reference_mesh, current_mesh) -> np.ndarray:
    """Paper metric heatmap: angular normal error in degrees at every vertex."""
    reference_normals = vertex_normals(reference_mesh.vertices, reference_mesh.triangles)
    current_normals = vertex_normals(current_mesh.vertices, current_mesh.triangles)

    dot = np.sum(reference_normals * current_normals, axis=1)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def sequence_maximum_normal_error(reference_frames, noisy_frames,) -> float:
    """Maximum angular normal error across the complete noisy sequence."""
    
    maximum_error = 0.0

    for reference, noisy in zip(reference_frames, noisy_frames):
        errors = vertex_normal_angle_error(reference, noisy)

        if len(errors):
            maximum_error = max(maximum_error, float(np.max(errors)),)

    return maximum_error

def error_heatmap_colors(reference_mesh, current_mesh, maximum_error: float) -> np.ndarray:
    """Blue = small angular error, red = large error. Maximum error is computed from noisy sequence
    and used to normalize the heatmap for all frames."""

    values = vertex_normal_angle_error(reference_mesh, current_mesh)
    scale = max(float(maximum_error), 1e-12)
    normalized = np.clip(values / scale, 0.0, 1.0)
    return cm["turbo"](normalized)[:, :3]


def meshes_from_vertices(reference_frames, vertices_sequence):
    """Create display meshes while keeping the common connectivity."""
    result = []
    for reference, vertices in zip(reference_frames, vertices_sequence):
        mesh = copy_mesh(reference)
        mesh.vertices = np.asarray(vertices, dtype=float)
        mesh.vertex_normals = vertex_normals(mesh.vertices, mesh.triangles)
        result.append(mesh)
    return result
