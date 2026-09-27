import numpy as np
from core.geometry import face_geometry, delta_coordinates_sparse
from core.visualization import heatmap_colors
from scipy.spatial import cKDTree
from vvrpywork.shapes import LineSet3D, Mesh3D
from vvrpywork.constants import Color
from scipy.cluster.vq import kmeans2

def knn_face_patches(face_centroids: np.ndarray, k: int = 20) -> tuple[np.ndarray, np.ndarray]:

    centroids = np.asarray(face_centroids, dtype=float)
    k = min(k, len(centroids))
    tree = cKDTree(centroids)

    distances, indices = tree.query(centroids, k=k, workers=-1)
    # distances: (F, k) array of distances to the k nearest neighbors for each face
    # indices: (F, k) array of indices of the k nearest neighbors for each face
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    return indices, distances

def min_max_normalize(values: np.ndarray) -> np.ndarray:
    """Return a normalized version of the input array, with values in [0, 1]."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or values.max() == values.min():
        return np.zeros_like(values)
    return (values - values.min()) / (values.max() - values.min())

def normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a normalized version of the input array of vectors, with unit length."""
    vectors = np.asarray(vectors, dtype=float)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > eps
    normalized = np.zeros_like(vectors, dtype=float)
    normalized[valid] = vectors[valid] / lengths[valid, None]
    return normalized


def filtered_face_normals(indices: np.ndarray, distances: np.ndarray, face_normals: np.ndarray, sigma_n: float = 0.35, iterations: int = 1, batch_size: int = 10000,) -> np.ndarray:

    normals = np.asarray(face_normals, dtype=float)
    patches = np.asarray(indices, dtype=np.int64)
    distances = np.asarray(distances, dtype=float)

    eps = 1e-12

    # Start from unit face normals.
    current = normalize_vectors(normals)

    # Spatial scale = median of positive k-NN centroid distances.
    positive_distances = distances[distances > eps]

    if len(positive_distances) == 0:
        return current

    sigma_c = np.median(positive_distances)

    for _ in range(iterations):

        # store filtered normals
        updated = current.copy()

        for start in range(0, len(current), batch_size):

            end = min(start + batch_size, len(current))

            batch_indices = patches[start:end]
            batch_distances = distances[start:end]

            # (B, k, 3)
            neighbour_normals = current[batch_indices]

            # (B, 1, 3)
            central_normals = current[start:end, None, :]

            # wc
            spatial_weights = np.exp(-(batch_distances ** 2) / (2 * sigma_c ** 2))

            normal_differences = neighbour_normals - central_normals
            normal_distances_squared = np.sum(normal_differences ** 2, axis=2,)

            # wn
            normal_similarity_weights = np.exp(-normal_distances_squared / (2 * sigma_n ** 2))

            # Bilateral weight.
            weights = spatial_weights * normal_similarity_weights

            # Degenerate neighbours have zero normal and should not contribute.
            valid_neighbours = (np.linalg.norm(neighbour_normals, axis=2) > eps)

            weights *= valid_neighbours

            # Weighted sum of neighbouring normals. (nominator of the bilateral filter)
            weighted_normals = np.sum(weights[:, :, None] * neighbour_normals, axis=1,)

            # Convert weighted sums back to unit normals.
            filtered_normals = normalize_vectors(weighted_normals, eps)

            # Keep the original value for degenerate central faces.
            valid_central_faces = (np.linalg.norm(current[start:end], axis=1) > eps)

            block = current[start:end].copy()

            block[valid_central_faces] = (filtered_normals[valid_central_faces])

            updated[start:end] = block

        current = updated

    return current

def local_normal_covariance_eigenvalues(face_normals, patch_indices) -> np.ndarray:
    """Return the normalized eigenvalues of N_i.T @ N_i for every face patch """
    normals = normalize_vectors(face_normals) # (F, 3)
    patches = np.asarray(patch_indices, dtype=np.int64) # (F, k) array of indices of the k nearest neighbors for each face
    patch_normals = normals[patches] # (F,k,3)
    # For every face: N_i.T @ N_i
    # (F, 3, k) @ (F, k, 3) -> (F, 3, 3)
    scatter_matrices = np.matmul(patch_normals.transpose(0, 2, 1), patch_normals,)

    # returns : lambda0 <= lambda1 <= lambda2
    eigenvalues = np.linalg.eigvalsh(scatter_matrices)

    # Normalize each eigenvalue triple so that: lambda0 + lambda1 + lambda2 = 1
    sums = np.sum(eigenvalues, axis=1, keepdims=True,)

    normalized_eigenvalues = np.divide(eigenvalues, sums, out=np.zeros_like(eigenvalues), where=sums > 1e-12,)

    return normalized_eigenvalues

   

def classify_face_features(normalized_eigenvalues) -> np.ndarray:
    """Classify faces as 0=flat, 1=edge or 2=corner."""
    normalized_eigenvalues = np.asarray(normalized_eigenvalues, dtype=float)
    if normalized_eigenvalues.ndim != 2 or normalized_eigenvalues.shape[1] != 3:
        raise ValueError("normalized_eigenvalues must have shape (F, 3).")

    lambda0 = normalized_eigenvalues[:, 0]
    lambda1 = normalized_eigenvalues[:, 1]
    lambda2 = normalized_eigenvalues[:, 2]

    d_flat = lambda2 - lambda1
    d_edge = lambda1 - lambda0
    d_corner = lambda0
    feature_vectors = np.stack((d_flat, d_edge, d_corner), axis=1)

    # Three meaningful clusters cannot be formed from fewer than three or
    # completely identical descriptors. Such a mesh is treated as flat.
    if len(feature_vectors) < 3 or np.allclose(feature_vectors, feature_vectors[0]):
        return np.zeros(len(feature_vectors), dtype=np.int64)

    try:
        cluster_centers, cluster_labels = kmeans2(feature_vectors, 3, minit="++", iter=50, missing="raise", seed=0,)
    except (ValueError, np.linalg.LinAlgError):
        # Deterministic fallback: use the strongest descriptor directly.
        return np.argmax(feature_vectors, axis=1).astype(np.int64)

    # Identify what each cluster represents
    # Cluster with strongest flat descriptor.
    flat_cluster = int(np.argmax(cluster_centers[:, 0]))

    # From the remaining two clusters, choose the one with strongest corner descriptor.
    remaining_clusters = [cluster for cluster in range(3) if cluster != flat_cluster]

    corner_cluster = remaining_clusters[int(np.argmax(cluster_centers[remaining_clusters, 2]))]
    
    # The only cluster left is the edge cluster.
    edge_cluster = next(cluster for cluster in remaining_clusters if cluster != corner_cluster)

    labels = np.zeros(len(cluster_labels), dtype=np.int64,)

    labels[cluster_labels == flat_cluster] = 0
    labels[cluster_labels == edge_cluster] = 1
    labels[cluster_labels == corner_cluster] = 2

    return labels

def face_feature_score(normalized_eigenvalues: np.ndarray) -> np.ndarray:
    """Return a feature score for every face """
    lambda2 = normalized_eigenvalues[:, 2]
    S = 1 - lambda2 # (F,) array of feature scores for each face
    return min_max_normalize(S)

def face_normal_roughness(raw_face_normals, patch_indices) -> np.ndarray:
    normals = np.asarray(raw_face_normals, dtype=float) # (F, 3)
    patches = np.asarray(patch_indices, dtype=np.int64) # (F, k)
    unit_normals = normalize_vectors(normals) # (F, 3)
    neighbor_normals = unit_normals[patches] # (F, k, 3)
    central_normals = unit_normals[:, None, :] # (F, 1, 3)

    dot_products = np.sum(neighbor_normals * central_normals, axis=2) # (F, k)
    angles = np.arccos(np.clip(dot_products, -1.0, 1.0)) # (F, k)
    if angles.shape[1] > 1:
        # we subtract the central face from the patch to avoid counting it in the mean roughness
        # because it's zero and we don't want to reduce the mean roughness 
        return min_max_normalize(np.mean(angles[:, 1:], axis=1))
    return min_max_normalize(np.mean(angles, axis=1))



def candidate_score(roughness, feature_score) -> np.ndarray:
    roughness = np.asarray(roughness, dtype=float)
    feature_score = np.asarray(feature_score, dtype=float)
    candidate_scores = min_max_normalize(roughness * (1 - feature_score)) # (F,) array of candidate scores for each face

    return candidate_scores

def candidate_mask(labels, candidate_scores, triangles, roughness_percentile: float = 80.0) -> np.ndarray:
    # Select candidates only among flat faces
    flat_mask = (labels == 0)
    if np.any(flat_mask):
        flat_scores = candidate_scores[flat_mask]
        # if we have measureable roughness
        if np.max(flat_scores) > 0.0:
            threshold = float(np.percentile(flat_scores, roughness_percentile))
            candidate_mask = (flat_mask & (candidate_scores >= threshold))
        else:
            threshold = np.inf
            candidate_mask = np.zeros(len(triangles), dtype=bool,)
    else:
        threshold = np.inf
        candidate_mask = np.zeros(len(triangles), dtype=bool,)
    return candidate_mask


def connected_candidate_face_regions(triangles, candidate_mask: np.ndarray, min_faces: int = 5) -> list[np.ndarray]:
    triangles = np.asarray(triangles, dtype=np.int64,)
    candidate_mask = np.asarray(candidate_mask, dtype=bool,) # (F,) boolean array indicating which faces are candidates
    # actual candidate faces
    candidate_faces = np.flatnonzero(candidate_mask) # (F,) array of indices of candidate faces

    # mesh edge -> candidate faces containing that edge
    edge_to_faces: dict[tuple[int, int], list[int],] = {}

    for face_index in candidate_faces:
        a, b, c = triangles[face_index]
        edges = ((int(a), int(b)), (int(b), int(c)), (int(c), int(a)),)

        for first, second in edges:
            # create an undirected edge representation (2, 7) == (7, 2)
            if first < second:
                edge = (first, second)
            else:
                edge = (second, first)

            edge_to_faces.setdefault(edge,[],).append(int(face_index))

    # Build candidate-face graph
    # used sets to avoid duplicate edges in the graph
    adjacency = {int(face_index): set() for face_index in candidate_faces}

    # connect all candidate faces that share an edge
    for faces in edge_to_faces.values(): 
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):

                first = faces[i]
                second = faces[j]

                adjacency[first].add(second)
                adjacency[second].add(first)

    # DFS connected components

    visited: set[int] = set()
    regions: list[np.ndarray] = []

    for start in candidate_faces:
        start = int(start)

        if start in visited:
            continue

        stack = [start]
        visited.add(start)

        region = [] # list of face indices in the current connected region

        while stack:

            current = stack.pop()
            region.append(current)

            for neighbour in adjacency[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)

        # Ignore very small isolated detections.
        if len(region) >= min_faces:
            regions.append(np.asarray(region, dtype=np.int64,))

    # Largest regions first.
    regions.sort(key=len, reverse=True,)

    return regions


def face_values_to_vertices_max(face_values, triangles, vertex_count: int) -> np.ndarray:
    """Return the maximum incident face value for every vertex."""
    values = np.asarray(face_values, dtype=float,)
    triangles = np.asarray(triangles, dtype=np.int64,)
    result = np.zeros(vertex_count, dtype=float,)

    # Every triangle contributes its face value to all three of its vertices.
    flat_vertices = triangles.reshape(-1) # (3*F,) array of vertex indices for all triangles
    repeated_values = np.repeat(values, 3, ) # (3*F,) array of face values repeated for each vertex of the triangle

    np.maximum.at(result, flat_vertices, repeated_values,)

    return result

def face_values_to_vertices_mean(face_values, triangles, vertex_count: int) -> np.ndarray:
    """Return the mean incident face value for every vertex."""
    values = np.asarray(face_values, dtype=float,)
    triangles = np.asarray(triangles, dtype=np.int64,)
    result = np.zeros(vertex_count, dtype=float,)

    counts = np.zeros(vertex_count, dtype=float,)

    flat_vertices = triangles.reshape(-1)
    repeated_values = np.repeat(values, 3,)

    np.add.at(result, flat_vertices, repeated_values,)
    np.add.at(counts, flat_vertices, 1.0,)

    valid = counts > 0

    result[valid] /= counts[valid]

    return result



def region_aabbs(vertices, triangles, regions) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute the axis-aligned bounding box (AABB) for each region of faces."""
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)

    aabbs: list[tuple[np.ndarray, np.ndarray]] = []

    for region in regions:
        # Get the vertex indices of the faces in the region
        face_vertex_indices = triangles[region] # (R, 3) array of vertex indices for each face in the region
        unique_vertex_indices = np.unique(face_vertex_indices) # (V,) array of unique vertex indices for the region

        # Get the coordinates of the vertices in the region
        region_vertices = vertices[unique_vertex_indices] # (V, 3) array of vertex coordinates for the region

        # Compute the AABB
        min_coords = np.min(region_vertices, axis=0) # (3,) array of minimum coordinates
        max_coords = np.max(region_vertices, axis=0) # (3,) array of maximum coordinates

        aabbs.append((min_coords, max_coords))

    return aabbs

def show_aabbs(aabbs: list[tuple[np.ndarray, np.ndarray]]) -> LineSet3D:
    """Draw all AABBs as one red line set."""
    points = []
    lines = []

    box_edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )

    for min_coords, max_coords in aabbs:
        x_min, y_min, z_min = min_coords
        x_max, y_max, z_max = max_coords

        first_point = len(points)
        points.extend(
            (
                (x_min, y_min, z_min),
                (x_max, y_min, z_min),
                (x_max, y_max, z_min),
                (x_min, y_max, z_min),
                (x_min, y_min, z_max),
                (x_max, y_min, z_max),
                (x_max, y_max, z_max),
                (x_min, y_max, z_max),
            )
        )

        for first, second in box_edges:
            lines.append((first_point + first, first_point + second))

    return LineSet3D(points=points, lines=lines, width=1, color=Color.RED)


def show_normals(current_mesh: Mesh3D, normal_length: float = 0.03, max_lines: int = 1000) -> LineSet3D:
    """Return a readable subset of face-normal lines, starting at face centroids."""
    vertices = np.asarray(current_mesh.vertices, dtype=float)
    triangles = np.asarray(current_mesh.triangles, dtype=np.int64)
    centroids, face_normals, _ = face_geometry(vertices, triangles)

    if max_lines <= 0:
        raise ValueError("max_lines must be positive.")

    step = max(1, int(np.ceil(len(centroids) / max_lines)))
    selected = np.arange(0, len(centroids), step)

    starts = centroids[selected]
    ends = starts + normal_length * face_normals[selected]

    points = np.vstack((starts, ends))
    line_count = len(starts)
    lines = np.column_stack((np.arange(line_count), np.arange(line_count) + line_count))

    return LineSet3D(points=points, lines=lines, width=1, color=Color.RED)


def show_denoising_candidate_regions(aabbs: list[tuple[np.ndarray, np.ndarray]]) -> LineSet3D:
    """Create one line set containing all detected Question 3 AABBs."""
    return show_aabbs(aabbs)

def show_candidate_heatmap(vertex_candidate_scores: np.ndarray) -> np.ndarray:
    """Return candidate heatmap colors for the current mesh."""
    return heatmap_colors(vertex_candidate_scores)

def show_feature_heatmap(vertex_feature_scores: np.ndarray) -> np.ndarray:
    """Return feature heatmap colors for the current mesh."""
    return heatmap_colors(vertex_feature_scores)

def show_delta_coordinates_heatmap(vertex_delta_coordinates: np.ndarray) -> np.ndarray:
    """Return delta-coordinate heatmap colors for the current mesh."""
    return heatmap_colors(vertex_delta_coordinates)

def show_roughness_heatmap(vertex_roughness: np.ndarray) -> np.ndarray:
    """Return roughness heatmap colors for the current mesh."""
    return heatmap_colors(vertex_roughness)

def delta_coordinate_values(current_mesh: Mesh3D,) -> np.ndarray:
    """Return ||delta_i|| for every vertex."""
    delta_coordinates = delta_coordinates_sparse(current_mesh.vertices, current_mesh.triangles)

    return np.linalg.norm(delta_coordinates, axis=1,)

def analyze_candidate_regions(current_mesh: Mesh3D, analysis: dict[str, object], roughness_percentile: float = 80.0, min_region_size: int = 5,) -> dict[str, object]:

    triangles = np.asarray(current_mesh.triangles, dtype=np.int64,)
    vertices = np.asarray(current_mesh.vertices, dtype=float,)

    labels = analysis["face_classes"]
    candidate_scores = analysis["candidate_scores"]
    candidate_face_mask = candidate_mask(labels, candidate_scores, triangles, roughness_percentile=roughness_percentile,)
    regions = connected_candidate_face_regions(triangles, candidate_face_mask, min_faces=min_region_size,) # list of arrays of face indices for each connected region
    aabbs = region_aabbs(vertices, triangles, regions,) # list of tuples of (min, max) coordinates for each region's AABB

    return {
        "candidate_mask": candidate_face_mask,
        "regions": regions,
        "aabbs": aabbs,
    }

def analyze_q3(current_mesh: Mesh3D, k: int = 20,) -> dict[str, object]:

    vertices = np.asarray(current_mesh.vertices, dtype=float,)
    triangles = np.asarray(current_mesh.triangles, dtype=np.int64,)
    face_centroids, raw_face_normals, _ = face_geometry(vertices, triangles,)

    indices, distances = knn_face_patches(face_centroids, k=k,)
    filtered_normals = filtered_face_normals(indices=indices, distances=distances, face_normals=raw_face_normals,)
    normalized_eigenvalues = local_normal_covariance_eigenvalues(filtered_normals, indices,)
    labels = classify_face_features(normalized_eigenvalues)
    feature_scores = face_feature_score(normalized_eigenvalues)
    roughness = face_normal_roughness(raw_face_normals, indices,)
    candidate_scores = candidate_score(roughness, feature_scores,)
    # Candidate heatmap: maximum incident candidate score
    vertex_candidate_scores = face_values_to_vertices_max(candidate_scores, triangles, len(vertices),)
    # Feature heatmap: mean incident feature score
    vertex_feature_scores = face_values_to_vertices_mean(feature_scores, triangles, len(vertices),)
    vertex_roughness = face_values_to_vertices_mean(roughness, triangles, len(vertices),)


    return {
        "centroids": face_centroids,
        "raw_face_normals": raw_face_normals,
        "classification_normals": filtered_normals,
        "patch_indices": indices,
        "patch_distances": distances,
        "eigenvalues": normalized_eigenvalues,
        "face_classes": labels,
        "feature_mask": labels != 0,
        "feature_scores": feature_scores,
        "roughness": roughness,
        "candidate_scores": candidate_scores,
        "vertex_candidate_scores": vertex_candidate_scores,
        "vertex_feature_scores": vertex_feature_scores,
        "vertex_roughness": vertex_roughness,
    }
