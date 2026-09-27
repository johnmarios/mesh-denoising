"""Simple area-uniform sampling of points on a triangle mesh."""

import numpy as np


def sample_mesh_surface(vertices, triangles, number_of_points, seed):
    """Return points sampled uniformly with respect to triangle area."""
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)

    if number_of_points <= 0:
        raise ValueError("number_of_points must be positive.")
    if len(triangles) == 0:
        raise ValueError("The mesh has no triangles.")

    triangle_vertices = vertices[triangles]
    edge_1 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    edge_2 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    triangle_areas = 0.5 * np.linalg.norm(np.cross(edge_1, edge_2), axis=1)

    total_area = float(np.sum(triangle_areas))
    if total_area <= 0.0:
        raise ValueError("The mesh has zero surface area.")

    generator = np.random.default_rng(seed)

    # Large triangles are selected more often than small triangles.
    triangle_probabilities = triangle_areas / total_area
    chosen_triangles = generator.choice(
        len(triangles),
        size=number_of_points,
        p=triangle_probabilities,
    )

    selected = triangle_vertices[chosen_triangles]

    # The square root makes the samples uniform inside every triangle.
    random_1 = np.sqrt(generator.random(number_of_points))
    random_2 = generator.random(number_of_points)

    weight_a = 1.0 - random_1
    weight_b = random_1 * (1.0 - random_2)
    weight_c = random_1 * random_2

    points = (
        weight_a[:, None] * selected[:, 0]
        + weight_b[:, None] * selected[:, 1]
        + weight_c[:, None] * selected[:, 2]
    )

    return points


def farthest_point_subset(points, number_of_points):
    """Choose a well-spread subset from a larger point set."""
    points = np.asarray(points, dtype=float)

    if number_of_points <= 0:
        raise ValueError("number_of_points must be positive.")
    if number_of_points > len(points):
        raise ValueError("The subset cannot be larger than the input.")

    selected = np.empty((number_of_points, points.shape[1]), dtype=float)
    closest_squared_distance = np.full(len(points), np.inf)

    center = np.mean(points, axis=0)
    next_index = int(np.argmax(np.linalg.norm(points - center, axis=1)))

    for index in range(number_of_points):
        selected[index] = points[next_index]

        squared_distance = np.sum(
            (points - points[next_index]) ** 2,
            axis=1,
        )
        closest_squared_distance = np.minimum(
            closest_squared_distance,
            squared_distance,
        )
        next_index = int(np.argmax(closest_squared_distance))

    return selected


def sample_mesh_surface_evenly(
    vertices,
    triangles,
    number_of_points,
    seed,
    candidate_multiplier=8,
):
    """Sample many surface candidates and keep a well-spread subset."""
    number_of_candidates = candidate_multiplier * number_of_points
    candidates = sample_mesh_surface(
        vertices,
        triangles,
        number_of_candidates,
        seed,
    )
    return farthest_point_subset(candidates, number_of_points)
