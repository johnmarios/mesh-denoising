import numpy as np

from core.geometry import average_edge_length

NOISE_TYPES = ("GAUSSIAN", "NORMAL", "UNIFORM", "IMPULSE", "QUANTIZATION")


def gaussian_noise(vertices: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return vertices + rng.normal(0.0, sigma, size=vertices.shape)


def normal_direction_noise(
    vertices: np.ndarray,
    vertex_normals: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Move every vertex only along its normal direction."""
    offsets = rng.normal(0.0, sigma, size=(len(vertices), 1))
    return vertices + offsets * vertex_normals


def uniform_noise(vertices: np.ndarray, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Add independent uniform displacement to each coordinate."""
    return vertices + rng.uniform(-amplitude, amplitude, size=vertices.shape)


def impulse_noise(vertices: np.ndarray, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    noisy = np.array(vertices, copy=True)
    count = max(1, int(0.02 * len(vertices)))
    indices = rng.choice(len(vertices), size=count, replace=False)

    directions = rng.normal(size=(count, 3))
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    noisy[indices] = noisy[indices] + 5.0 * amplitude * directions
    return noisy


def quantization_noise(vertices: np.ndarray, step: float) -> np.ndarray:
    return np.round(vertices / step) * step


def add_noise(
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_normals: np.ndarray,
    noise_type: str,
    level: float,
    seed: int = 7,
) -> np.ndarray:

    noise_type = noise_type.upper()

    scale = average_edge_length(vertices, triangles)
    amount = float(level) * scale
    rng = np.random.default_rng(seed)

    if noise_type == "GAUSSIAN":
        return gaussian_noise(vertices, amount, rng)
    if noise_type == "NORMAL":
        return normal_direction_noise(vertices, vertex_normals, amount, rng)
    if noise_type == "UNIFORM":
        return uniform_noise(vertices, amount, rng)
    if noise_type == "IMPULSE":
        return impulse_noise(vertices, amount, rng)
    if noise_type == "QUANTIZATION":
        return quantization_noise(vertices, amount)