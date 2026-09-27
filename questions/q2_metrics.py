import numpy as np

from core.geometry import (
    face_normals_and_areas,
    one_ring_area_weights,
    point_to_surface_distances,
    delta_coordinates_sparse,
    surface_area,
)
from vvrpywork.shapes import Mesh3D

def mean_face_normal_angular_error(reference_vertices: np.ndarray, current_vertices: np.ndarray, triangles: np.ndarray,) -> float:
    """Return the angle in degrees."""
    reference_normals, reference_areas = face_normals_and_areas(reference_vertices, triangles)
    current_normals, current_areas = face_normals_and_areas(current_vertices, triangles)

    # A degenerate triangle has no well-defined normal in either mesh.
    tolerance = np.finfo(float).eps
    valid_faces = (reference_areas > tolerance) & (current_areas > tolerance)
    if not np.any(valid_faces):
        raise ValueError("No valid non-degenerate corresponding faces were found.")

    dot_products = np.sum(reference_normals[valid_faces] * current_normals[valid_faces], axis=1)
    angles_radians = np.arccos(np.clip(dot_products, -1.0, 1.0))
    return float(np.degrees(np.mean(angles_radians)))


def area_weighted_l2_vertex_to_surface_distance(current_vertices: np.ndarray, current_triangles: np.ndarray, distances: np.ndarray, reference_area: float,) -> float:

    if reference_area <= np.finfo(float).eps:
        raise ValueError("The reference mesh has zero surface area.")

    vertex_area_weights = one_ring_area_weights(current_vertices, current_triangles) # Ai
    squared_distances = distances**2 # dist(pi,T)^2
    weighted_squared_error = np.sum(vertex_area_weights * squared_distances)
    return float(np.sqrt(weighted_squared_error / (3.0 * reference_area)))



def vertex_based_hausdorff_distance(reference_vertices: np.ndarray, reference_triangles: np.ndarray, current_vertices: np.ndarray, distances: np.ndarray,) -> float:
   
    if len(current_vertices) == 0:
        raise ValueError("The current mesh has no vertices.")

    return float(np.max(distances))


def corresponding_vertex_rmse(reference_vertices: np.ndarray, current_vertices: np.ndarray) -> float:
    
    if reference_vertices.shape != current_vertices.shape:
        raise ValueError("Corresponding-vertex RMSE requires equal vertex-array shapes.")
    if len(reference_vertices) == 0:
        raise ValueError("Cannot compute RMSE for empty vertex arrays.")

    displacement = current_vertices - reference_vertices
    squared_distance_per_vertex = np.sum(displacement**2, axis=1)
    return float(np.sqrt(np.mean(squared_distance_per_vertex)))


def high_frequency_energy(mesh: Mesh3D) -> float:
    if len(mesh.vertices) == 0:
        raise ValueError("Cannot compute high-frequency energy for an empty mesh.")
    delta_coordinates = delta_coordinates_sparse(mesh.vertices, mesh.triangles)
    squared_magnitudes = np.sum(delta_coordinates**2, axis=1)
    return float(np.mean(squared_magnitudes))


def mean_radius(vertices: np.ndarray) -> float:

    if len(vertices) == 0:
        raise ValueError("Cannot compute mean radius for an empty mesh.")

    centroid = np.mean(vertices, axis=0)
    distances_from_centroid = np.linalg.norm(vertices - centroid, axis=1)
    return float(np.mean(distances_from_centroid))


def bounding_box_volume(vertices: np.ndarray) -> float:

    if len(vertices) == 0:
        raise ValueError("Cannot compute an AABB for an empty mesh.")

    side_lengths = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    return float(np.prod(side_lengths))


def _ratio(numerator: float, denominator: float, metric_name: str) -> float:
    if abs(denominator) <= np.finfo(float).eps:
        raise ValueError(f"Cannot compute {metric_name}: reference value is zero.")
    return float(numerator / denominator)


def compute_metrics(reference_mesh, current_mesh) -> dict[str, float]:

    reference_vertices = np.asarray(reference_mesh.vertices, dtype=float)
    reference_triangles = np.asarray(reference_mesh.triangles, dtype=np.int64)

    current_vertices = np.asarray(current_mesh.vertices, dtype=float)
    current_triangles = np.asarray(current_mesh.triangles, dtype=np.int64)

    if reference_vertices.shape != current_vertices.shape:
        raise ValueError("Q2 requires the same number of corresponding vertices.")
    if not np.array_equal(reference_triangles, current_triangles):
        raise ValueError("Q2 requires identical triangle connectivity.")

    # These closest-point distances are shared by both surface-distance metrics.
    distances = point_to_surface_distances(query_vertices=current_vertices, surface_vertices=reference_vertices, surface_triangles=reference_triangles)

    reference_area = surface_area(reference_vertices, reference_triangles)
    current_area = surface_area(current_vertices, current_triangles)

    reference_energy = high_frequency_energy(reference_mesh)
    current_energy = high_frequency_energy(current_mesh)

    reference_radius = mean_radius(reference_vertices)
    current_radius = mean_radius(current_vertices)

    reference_box_volume = bounding_box_volume(reference_vertices)
    current_box_volume = bounding_box_volume(current_vertices)

    return {
        "MSAE_deg": mean_face_normal_angular_error(reference_vertices, current_vertices, reference_triangles),
        "L2_surface": area_weighted_l2_vertex_to_surface_distance(current_vertices, current_triangles, distances, reference_area),
        "Hausdorff": vertex_based_hausdorff_distance(reference_vertices, reference_triangles, current_vertices, distances),
        "RMSE": corresponding_vertex_rmse(reference_vertices, current_vertices),
        "HF_ratio": _ratio(current_energy, reference_energy, "HF ratio"),
        "Radius_ratio": _ratio(current_radius, reference_radius, "radius ratio"),
        "AABB_ratio": _ratio(current_box_volume, reference_box_volume, "AABB ratio"),
        "Area_ratio": _ratio(current_area, reference_area, "area ratio"),
    }


def format_metrics(values: dict[str, float]) -> str:
    """Format one metric per line for terminal or GUI output."""
    return "\n".join(f"{name:14s}: {value:.6g}" for name, value in values.items())
