"""Paper-faithful spectral steps for Question 8.

The functions follow Algorithm 1 of Arvanitis, Lalos and Moustakas (2019):
Laplacian -> GFT -> high-frequency coherent matrices -> low-rank RPCA -> IGFT.
"""

import numpy as np

from questions.q8_dynamic import config
from core.geometry import *

def graph_fourier_transform_sequence(vertices_sequence: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    frame_count, vertex_count, _ = vertices_sequence.shape
    gfts = np.empty((frame_count, vertex_count, 3), dtype=float)
    columns = vertices_sequence.transpose(1, 0, 2) # (V, T, 3) 
    columns = columns.reshape(vertex_count, frame_count * 3) # (V, T*3)
    transformed = eigenvectors.T @ columns # (V, T*3)
    gfts = transformed.reshape(vertex_count, frame_count, 3).transpose(1, 0, 2) # (T, V, 3)
    return gfts

def inverse_graph_fourier_transform_sequence(gfts: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    frame_count, vertex_count, _ = gfts.shape
    columns = gfts.transpose(1, 0, 2) # (V, T, 3)
    columns = columns.reshape(vertex_count, frame_count * 3) # (V, T*3)
    reconstructed = eigenvectors @ columns # (V, T*3)
    vertices_sequence = reconstructed.reshape(vertex_count, frame_count, 3).transpose(1, 0, 2) # (T, V, 3)
    return vertices_sequence

def graph_fourier_transform(vertices: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """V_hat = U^T V."""
    return eigenvectors.T @ np.asarray(vertices, dtype=float) # (V, 3)

def inverse_graph_fourier_transform(gft: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """V = U V_hat."""
    return eigenvectors @ gft

def compute_k_bars(gfts: np.ndarray, energy_fraction: float = config.HIGH_FREQUENCY_ENERGY_FRACTION,) -> tuple[np.ndarray, int]:
    gfts = np.asarray(gfts) # (T, V, 3)

    component_energy = np.sum(gfts.astype(np.float64) ** 2, axis=2,) # (T, V) energy of each vertex in each frame

    total_energy = np.sum(component_energy, axis=1) # (T,) energy of each frame
    targets = energy_fraction * total_energy
    cumulative = np.cumsum(component_energy, axis=1)

    reached = cumulative >= targets[:, None] # (T, V) boolean array indicating which vertices have reached the target energy

    if not np.all(np.any(reached, axis=1)):
        raise RuntimeError("The full GFT did not reach the requested energy threshold.")

    # argmax returns the first 1 (True) in each row 
    k_bars = np.argmax(reached, axis=1) + 1 # (T,) index of the first vertex that exceeds the target energy in each frame
    k_bar = int(np.max(k_bars)) # max(k_bar_i) over all frames

    return k_bars.astype(np.int64), k_bar


def compute_temporal_matrices(gfts: np.ndarray, k_bar: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Ex, Ey, Ez."""
    high = gfts[:, :k_bar, :] # (T, k_bar, 3)

    Ex = high[:, :, 0] # (T, k_bar)
    Ey = high[:, :, 1] # (T, k_bar)
    Ez = high[:, :, 2] # (T, k_bar)

    return Ex, Ey, Ez

def low_rank_rpca(E: np.ndarray, rank: int, threshold: float, iterations: int, seed: int,) -> np.ndarray:
    """Estimate the low-rank part S of a temporal matrix E."""
    E = np.asarray(E, dtype=float) # (T, k_bar)
    rows, columns = E.shape
    rank = max(1, min(rank, rows, columns)) # ensure rank is at least 1 and at most min(rows, columns)

    # Initial sparse-noise estimate.
    N = np.zeros_like(E)

    # Initial low-rank coefficient matrix T.
    rng = np.random.default_rng(seed)
    T = rng.standard_normal((rank, columns))

    for _ in range(iterations):
        # Find an orthonormal basis Q for the current low-rank temporal subspace.
        A = (E - N) @ T.T
        Q, R = np.linalg.qr(A, mode="reduced")

        # Coordinates of the data inside that subspace.
        T = Q.T @ (E - N)

        # Current low-rank estimate.
        S = Q @ T

        # What is not explained by the low-rank model.
        residual = E - S

        # Keep large residuals as sparse corruption.
        N = soft_threshold(residual, threshold)

    return S

def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return (np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0,))

def reconstruct_gfts(gfts: np.ndarray, Sx: np.ndarray, Sy: np.ndarray, Sz: np.ndarray, k_bars: np.ndarray,) -> np.ndarray:

    denoised_gfts = gfts.copy() # (T, V, 3)

    for frame_index in range(len(gfts)):
        k_i = k_bars[frame_index]

        denoised_gfts[frame_index, :k_i, 0] = Sx[frame_index, :k_i]
        denoised_gfts[frame_index, :k_i, 1] = Sy[frame_index, :k_i]
        denoised_gfts[frame_index, :k_i, 2] = Sz[frame_index, :k_i]

    return denoised_gfts

def reconstruct_vertices(original_vertices: np.ndarray, original_gft: np.ndarray, denoised_gft: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:

    spectral_correction = (denoised_gft - original_gft)
    vertex_correction = inverse_graph_fourier_transform(spectral_correction, eigenvectors)

    return (original_vertices + vertex_correction)

def denoise_dynamic_vertices(vertices_sequence: np.ndarray, triangles: np.ndarray, eigenvectors: np.ndarray = None) -> np.ndarray:
    """dynamic mesh denoising process."""
    vertices_sequence = np.asarray(vertices_sequence, dtype=float) # (T, V, 3)
    vertex_count = vertices_sequence.shape[1]
    if eigenvectors is None:
        # compute L from the first frame's triangles and vertices (same topology)
        L = graph_laplacian(vertices_sequence[0], triangles) 
        # compute the eignenvalues and eigenvectors of L
        eigenvalues, eigenvectors = eigendecomposition(L)

    # GFT 
    # gfts = np.stack([graph_fourier_transform(vertices, eigenvectors) for vertices in vertices_sequence]) # (T, V, 3)
    gfts = graph_fourier_transform_sequence(vertices_sequence, eigenvectors) # (T, V, 3)

    # find k_bar = max(k_bar_i), k_bar_i which is the number of high frequencies in each frame
    k_bars, k_bar = compute_k_bars(gfts)

    # Build Ex/Ey/Ez.
    Ex, Ey, Ez = compute_temporal_matrices(gfts, k_bar)

    # RPCA: denoise the temporal matrices.
    Sx = low_rank_rpca(Ex, rank=config.RPCA_RANK, threshold=config.RPCA_THRESHOLD_FACTOR * float(np.std(Ex)), iterations=config.RPCA_ITERATIONS, seed=config.RPCA_SEED)
    Sy = low_rank_rpca(Ey, rank=config.RPCA_RANK, threshold=config.RPCA_THRESHOLD_FACTOR * float(np.std(Ey)), iterations=config.RPCA_ITERATIONS, seed=config.RPCA_SEED + 1)
    Sz = low_rank_rpca(Ez, rank=config.RPCA_RANK, threshold=config.RPCA_THRESHOLD_FACTOR * float(np.std(Ez)), iterations=config.RPCA_ITERATIONS, seed=config.RPCA_SEED + 2)

    # Put the denoised high frequencies back into each frame's GFT.
    denoised_gfts = reconstruct_gfts(gfts, Sx, Sy, Sz, k_bars)

    # Inverse GFT -> XYZ vertices.
    denoised_vertices = inverse_graph_fourier_transform_sequence(denoised_gfts, eigenvectors) # (T, V, 3)

    return denoised_vertices