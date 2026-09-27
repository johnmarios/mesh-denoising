"""Apply a trained Q7 PointNet to every vertex of a mesh."""

import numpy as np
import torch
from vvrpywork.shapes import Mesh3D

from questions.q7_pointnet.checkpoint import checkpoint_path, load_checkpoint
from questions.q7_pointnet.data import knn_vertex_patches
from questions.q7_pointnet.model import PointNetDenoiser


def normalized_inference_patches(
    noisy_vertices: np.ndarray,
    patch_indices: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Use exactly the same centering/scaling convention as during training."""
    noisy_vertices = np.asarray(noisy_vertices, dtype=float)
    patch_indices = np.asarray(patch_indices, dtype=np.int64)

    patches = noisy_vertices[patch_indices]
    centered_patches = patches - noisy_vertices[:, None, :]

    radii = np.max(np.linalg.norm(centered_patches, axis=2), axis=1)
    radii = np.maximum(radii, eps)

    normalized_patches = centered_patches / radii[:, None, None]
    return normalized_patches, radii


def load_trained_model(
    output_directory: str,
    checkpoint_choice: str,
    device: torch.device,
) -> PointNetDenoiser:
    """Load either the best or latest Q7 checkpoint."""
    model = PointNetDenoiser().to(device)
    path = checkpoint_path(output_directory, checkpoint_choice)
    load_checkpoint(path, model, device)
    model.eval()
    return model


def denoise_mesh_with_model(
    mesh: Mesh3D,
    model: PointNetDenoiser,
    device: torch.device,
    k: int = 64,
    batch_size: int = 256,
) -> np.ndarray:
    """Predict one inverse-noise displacement for every mesh vertex."""
    noisy_vertices = np.asarray(mesh.vertices, dtype=float)

    patch_indices = knn_vertex_patches(noisy_vertices, k=k)
    normalized_patches, radii = normalized_inference_patches(
        noisy_vertices,
        patch_indices,
    )

    predictions = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(normalized_patches), batch_size):
            batch = torch.from_numpy(
                normalized_patches[start : start + batch_size].astype(np.float32)
            ).to(device)
            predictions.append(model(batch).cpu().numpy())

    normalized_displacements = np.concatenate(predictions, axis=0)
    displacements = normalized_displacements * radii[:, None]

    return noisy_vertices + displacements
