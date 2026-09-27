"""Point-to-surface values used only to color correspondence-free heatmaps."""

import numpy as np
import open3d as o3d
from matplotlib import colormaps


def create_target_surface(vertices, triangles):
    """Build an Open3D scene used for closest-point queries."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(vertices, dtype=float)
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(triangles, dtype=np.int32)
    )

    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    return scene


def point_to_surface_distances(points, target_surface):
    """Return the unsigned distance of every point from the target surface."""
    points = np.asarray(points, dtype=float)
    query = o3d.core.Tensor(points.astype(np.float32))
    result = target_surface.compute_closest_points(query)
    closest_points = result["points"].numpy()
    return np.linalg.norm(points - closest_points, axis=1)


def error_heatmap_colors(distances, color_limit):
    """Map zero to blue and color_limit (or more) to red."""
    distances = np.asarray(distances, dtype=float)
    color_limit = max(float(color_limit), 1e-12)
    normalized = np.clip(distances / color_limit, 0.0, 1.0)
    return colormaps["coolwarm"](normalized)[:, :3]
