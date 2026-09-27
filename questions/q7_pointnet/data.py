"""Dataset construction for Question 7.

Each sample is a local k-NN point-cloud patch around one mesh vertex. The
network target is the normalized correction that moves the noisy center back
to its clean position.
"""

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
from vvrpywork.shapes import Mesh3D

from questions.q1_noise import add_noise


def knn_vertex_patches(vertices: np.ndarray, k: int = 64) -> np.ndarray:
    """Return the k nearest vertex indices for every mesh vertex."""
    vertices = np.asarray(vertices, dtype=float)
    k = min(k, len(vertices))

    tree = cKDTree(vertices)
    _, indices = tree.query(vertices, k=k, workers=-1)

    if k == 1:
        indices = indices[:, None]

    return np.asarray(indices, dtype=np.int64)


def normalize_patches_and_targets(
    noisy_vertices: np.ndarray,
    clean_vertices: np.ndarray,
    patch_indices: np.ndarray,
    center_indices: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center and scale selected patches and their inverse-noise targets."""
    noisy_vertices = np.asarray(noisy_vertices, dtype=float)
    clean_vertices = np.asarray(clean_vertices, dtype=float)
    patch_indices = np.asarray(patch_indices, dtype=np.int64)
    center_indices = np.asarray(center_indices, dtype=np.int64)

    selected_patch_indices = patch_indices[center_indices]
    noisy_centers = noisy_vertices[center_indices]
    clean_centers = clean_vertices[center_indices]

    patches = noisy_vertices[selected_patch_indices]
    centered_patches = patches - noisy_centers[:, None, :]

    # Use the farthest point in each local patch as a scale factor.
    radii = np.max(np.linalg.norm(centered_patches, axis=2), axis=1)
    radii = np.maximum(radii, eps)

    normalized_patches = centered_patches / radii[:, None, None]

    # Ground truth inverse noise: where the noisy center must move to become clean.
    corrections = clean_centers - noisy_centers
    normalized_targets = corrections / radii[:, None]

    return normalized_patches, normalized_targets, radii


def build_noisy_variants(
    clean_vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_normals: np.ndarray,
    noise_types: tuple[str, ...],
    noise_levels: tuple[float, ...],
    seed: int,
) -> list[dict]:
    """Create one clean variant and every requested noisy variant.

    The clean sample is added explicitly instead of using noise level 0.0.
    Therefore add_noise() is called only with strictly positive levels.
    """
    variants = [
        {
            "noisy_vertices": np.array(clean_vertices, dtype=float, copy=True),
            "noise_type": "CLEAN",
            "noise_level": 0.0,
        }
    ]

    variant_seed = seed
    for noise_type in noise_types:
        for noise_level in noise_levels:
            noisy_vertices = add_noise(
                clean_vertices,
                triangles,
                vertex_normals,
                noise_type,
                noise_level,
                variant_seed,
            )
            variants.append(
                {
                    "noisy_vertices": noisy_vertices,
                    "noise_type": noise_type,
                    "noise_level": noise_level,
                }
            )
            variant_seed += 1

    return variants


def choose_center_indices(
    clean_vertices: np.ndarray,
    noisy_vertices: np.ndarray,
    count: int,
    seed: int,
    balance_impulse: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    """Choose patch centers.

    Normally centers are sampled uniformly. For impulse-noise TRAINING only,
    half the centers are chosen from displaced vertices and half from unchanged
    vertices. This prevents the rare impulse corruptions from disappearing in
    a dataset dominated by zero-displacement targets.
    """
    rng = np.random.default_rng(seed)
    vertex_count = len(noisy_vertices)

    if not balance_impulse:
        return rng.choice(
            vertex_count,
            size=count,
            replace=count > vertex_count,
        )

    displacement = np.linalg.norm(clean_vertices - noisy_vertices, axis=1)
    displaced = np.flatnonzero(displacement > eps)
    unchanged = np.flatnonzero(displacement <= eps)

    # Fall back to uniform sampling if a balanced split is impossible.
    if len(displaced) == 0 or len(unchanged) == 0:
        return rng.choice(
            vertex_count,
            size=count,
            replace=count > vertex_count,
        )

    displaced_count = count // 2
    unchanged_count = count - displaced_count

    selected = np.concatenate(
        [
            rng.choice(
                displaced,
                size=displaced_count,
                replace=displaced_count > len(displaced),
            ),
            rng.choice(
                unchanged,
                size=unchanged_count,
                replace=unchanged_count > len(unchanged),
            ),
        ]
    )
    rng.shuffle(selected)
    return selected


def random_rotation_matrix() -> np.ndarray:
    """Generate one random proper 3D rotation using a unit quaternion."""
    quaternion = np.random.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion

    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ]
    )


class PatchDenoisingDataset(Dataset):
    """Precompute local patches and inverse-noise displacement targets."""

    def __init__(
        self,
        clean_mesh: Mesh3D,
        noise_types: tuple[str, ...],
        noise_levels: tuple[float, ...],
        samples_per_variant: int,
        k: int = 64,
        seed: int = 42,
        augment: bool = False,
        augment_probability: float = 0.75,
        balance_impulse: bool = False,
    ):
        self.augment = augment
        self.rotation_probability = augment_probability
        clean_vertices = np.asarray(clean_mesh.vertices, dtype=float)
        triangles = np.asarray(clean_mesh.triangles, dtype=np.int64)
        vertex_normals = np.asarray(clean_mesh.vertex_normals, dtype=float)

        variants = build_noisy_variants(
            clean_vertices,
            triangles,
            vertex_normals,
            noise_types,
            noise_levels,
            seed,
        )

        all_patches = []
        all_targets = []

        for variant_index, variant in enumerate(variants):
            noisy_vertices = variant["noisy_vertices"]
            noise_type = variant["noise_type"]

            patch_indices = knn_vertex_patches(noisy_vertices, k=k)
            center_indices = choose_center_indices(
                clean_vertices,
                noisy_vertices,
                count=samples_per_variant,
                seed=seed + variant_index,
                balance_impulse=(balance_impulse and noise_type == "IMPULSE"),
            )

            patches, targets, _ = normalize_patches_and_targets(
                noisy_vertices,
                clean_vertices,
                patch_indices,
                center_indices,
            )
            all_patches.append(patches)
            all_targets.append(targets)

        self.patches = np.concatenate(all_patches, axis=0).astype(np.float32)
        self.targets = np.concatenate(all_targets, axis=0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch = self.patches[index].copy()
        target = self.targets[index].copy()

        # Rotate the patch and its vector target by exactly the same rotation.
        # This augmentation is used only for the training dataset.
        if self.augment and np.random.random() < self.rotation_probability:
            rotation = random_rotation_matrix()
            patch = patch @ rotation.T
            target = target @ rotation.T

        return (
            torch.from_numpy(patch.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )
