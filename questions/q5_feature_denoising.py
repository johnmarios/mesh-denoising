import numpy as np

from core.geometry import face_geometry
from questions import q3_analysis
from vvrpywork.shapes import Mesh3D

def _patch_centers_containing_each_face(patch_indices: np.ndarray,) -> list[list[int]]:
    """For every face, return the patches in which that face appears. """
    # patch_indices: (F,k)
    # containing_centers is a list of lists, where each inner list contains the indices of the patches that contain the corresponding face.
    containing_centers = [[] for _ in range(len(patch_indices))] # (F, number of patches that contain the face)

    for patch_center, patch in enumerate(patch_indices):
        # patch is the list of face indices that are in the patch which is centered at patch_center
        for face_index in patch:
            containing_centers[int(face_index)].append(patch_center)

    return containing_centers


def select_ideal_face_patches(face_normals: np.ndarray, patch_indices: np.ndarray, feature_mask: np.ndarray, eps: float = 1e-12,) -> np.ndarray:

    normals = q3_analysis.normalize_vectors(face_normals, eps)
    patches = np.asarray(patch_indices, dtype=np.int64) # (F, k)
    features = np.asarray(feature_mask, dtype=bool) # (F, ) 

    selected_patches = np.array(patches, copy=True) # (F, k)
    containing_centers = _patch_centers_containing_each_face(patches) # (F, number of patches that contain the face)

    for face_index in np.flatnonzero(features): # for every face that is classified as a feature, select the best patch 
        central_normal = normals[face_index]
        if np.linalg.norm(central_normal) <= eps: continue # skip faces with zero normals 

        # the best patch is the one with the highest score
        best_score = -np.inf
        best_patch = patches[face_index] # initialize the best patch to the current patch of the face
        
        for patch_center in containing_centers[face_index]: # for every candidate patch of the current face
            candidate_patch = patches[patch_center] # (k, ) indices of the faces in the candidate patch
            candidate_normals = normals[candidate_patch]

            scatter = candidate_normals.T @ candidate_normals
            eigenvalues = np.maximum(np.linalg.eigvalsh(scatter), 0.0) # (3, )

            tau = (eigenvalues[2] - eigenvalues[0]) / (eigenvalues[2] + eps) # τ

            # compute the average normal of the candidate patch ng
            average_normal = np.sum(candidate_normals, axis=0)
            average_length = np.linalg.norm(average_normal)
            if average_length <= eps: continue
            average_normal /= average_length

            alignment = max(float(np.dot(central_normal, average_normal)), 0.0) # ξ 
            normal_differences = central_normal - candidate_normals
            omega = float(np.max(np.sum(normal_differences**2, axis=1))) # ω
            score = tau * alignment / (omega + eps)

            if score > best_score:
                best_score = score
                best_patch = candidate_patch

        selected_patches[face_index] = best_patch

    return selected_patches




def update_vertices_from_face_normals(vertices: np.ndarray, triangles: np.ndarray, filtered_face_normals: np.ndarray, iterations: int, step_size: float,) -> np.ndarray:

    result = np.array(vertices, dtype=float, copy=True)
    triangles = np.asarray(triangles, dtype=np.int64)
    normals = q3_analysis.normalize_vectors(filtered_face_normals)

    flat_vertices = triangles.reshape(-1)
    repeated_faces = np.repeat(np.arange(len(triangles)), 3) # [0, 0, 0, 1, 1, 1, 2, 2, 2, ...] (F*3, )

    incident_face_counts = np.zeros(len(result), dtype=float) # we count for each vertex how many faces are incident to it
    np.add.at(incident_face_counts, flat_vertices, 1.0) # Ci
    referenced_vertices = incident_face_counts > 0.0

    for _ in range(iterations):
        face_centroids = np.mean(result[triangles], axis=1) # (F, 3), cj (at every iteration, centroids change so we recompute them)

        vertex_to_centroid = (face_centroids[repeated_faces] - result[flat_vertices]) # (F*3, 3), cj - vi
        incident_normals = normals[repeated_faces]

        signed_distance = np.sum(incident_normals * vertex_to_centroid, axis=1,) # (F*3, ), di,j = (cj - vi) · nj
        contributions = incident_normals * signed_distance[:, None] # (F*3, 3), di,j * nj

        correction = np.zeros_like(result)
        np.add.at(correction, flat_vertices, contributions)

        result[referenced_vertices] += step_size * (correction[referenced_vertices] / incident_face_counts[referenced_vertices, None])

    return result


def feature_aware_denoising(vertices: np.ndarray, triangles: np.ndarray, strength: float, iterations: int,  k: int = 20,) -> tuple[np.ndarray, np.ndarray]:

    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)

    face_centroids, raw_face_normals, _ = face_geometry(vertices, triangles)
    patch_indices, patch_distances = q3_analysis.knn_face_patches(face_centroids, k=k,)

    # one iteration of prefiltering is enough to get a good classification of the face normals
    classification_normals = q3_analysis.filtered_face_normals(indices=patch_indices, distances=patch_distances, face_normals=raw_face_normals, iterations=1,)

    eigenvalues = q3_analysis.local_normal_covariance_eigenvalues(classification_normals, patch_indices,)
    face_classes = q3_analysis.classify_face_features(eigenvalues)
    feature_mask = face_classes != 0
    face_feature_scores = q3_analysis.face_feature_score(eigenvalues)

    # select the best patch for each face
    selected_patches = select_ideal_face_patches(classification_normals, patch_indices, feature_mask,) # (F, k)

    new_indices = np.array(selected_patches, copy=True) # (F, k)
    # compute new distances for the selected patches, to be used in the bilateral filter
    new_centroids = face_centroids[new_indices] # (F, k, 3)
    central_centroids = face_centroids[:, None, :] # (F, 1, 3)
    new_distances = np.linalg.norm(new_centroids - central_centroids, axis=2) # (F, k)

    # bilateral filter 2 times to get a good smoothing of the face normals, while preserving features
    filtered_normals = q3_analysis.filtered_face_normals(indices=new_indices, distances=new_distances, face_normals=classification_normals, iterations=2,)

    denoised_vertices = update_vertices_from_face_normals(vertices, triangles, filtered_normals, iterations=iterations, step_size=strength,)

    vertex_feature_scores = q3_analysis.face_values_to_vertices_mean(face_feature_scores, triangles, len(vertices),)

    return denoised_vertices, vertex_feature_scores
