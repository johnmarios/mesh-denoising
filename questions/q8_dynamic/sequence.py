"""Loading and storing a dynamic mesh as an ordered sequence of static meshes."""

from pathlib import Path
import re

import numpy as np

from core.mesh import copy_mesh, load_mesh_from_path, vertex_normals
from core.visualization import gray_colors
from questions import q1_noise


def _natural_key(path: Path):
    """Sort frame_2.obj before frame_10.obj."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def load_frames(folder: str | Path, require_correspondence: bool = True,):
    """Load OBJ frames in order."""

    folder = Path(folder)
    paths = sorted(folder.glob("*.obj"), key=_natural_key,)

    if not paths:
        raise FileNotFoundError(f"No .obj frames found in: {folder}")

    frames = [load_mesh_from_path(path) for path in paths]

    if require_correspondence:
        validate_correspondence(frames)

    return prepare_frames_together(frames)

def validate_correspondence(frames) -> None:
    """Q8 assumes the same vertices and triangle connectivity in every frame."""
    first_vertices = np.asarray(frames[0].vertices)
    first_triangles = np.asarray(frames[0].triangles, dtype=np.int64)

    for index, frame in enumerate(frames[1:], start=1):
        if len(frame.vertices) != len(first_vertices):
            raise ValueError(f"Frame {index} has a different number of vertices. Q8 requires point correspondence between frames.")
        if not np.array_equal(np.asarray(frame.triangles, dtype=np.int64), first_triangles):
            raise ValueError(f"Frame {index} has different triangle connectivity. Q8 requires identical connectivity.")


def prepare_frames_together(frames):
    """Apply one common center/scale transform to the whole animation. """
    all_vertices = np.concatenate([np.asarray(frame.vertices, dtype=float) for frame in frames], axis=0,) # (N_frames * N_vertices, 3)
    center = all_vertices.mean(axis=0)
    radius = np.max(np.linalg.norm(all_vertices - center, axis=1))
    if radius <= 0.0: radius = 1.0

    prepared = []
    for frame in frames:
        frame.triangles = np.asarray(frame.triangles, dtype=np.int64)
        frame.vertex_normals = vertex_normals(frame.vertices, frame.triangles)
        frame.vertex_colors = gray_colors(len(frame.vertices))

        result = copy_mesh(frame)
        result.vertices = (np.asarray(frame.vertices, dtype=float) - center) / radius
        result.vertex_normals = vertex_normals(result.vertices, result.triangles)
        result.vertex_colors = gray_colors(len(result.vertices))
        prepared.append(result)

    return prepared


def copy_frames(frames):
    return [copy_mesh(frame) for frame in frames]


def add_noise_to_frames(frames, noise_type: str, level: float, seed: int):
    """Apply the same noise type/level but a different realization per frame."""
    noisy_frames = []

    for frame_index, frame in enumerate(frames):
        vertices = q1_noise.add_noise(frame.vertices, frame.triangles, frame.vertex_normals, noise_type, level, seed + frame_index,)
        noisy = copy_mesh(frame)
        noisy.vertices = np.asarray(vertices, dtype=float)
        noisy.vertex_normals = vertex_normals(noisy.vertices, noisy.triangles)
        noisy_frames.append(noisy)

    return noisy_frames


class DynamicMeshSession:
    """Small state holder mirroring MeshSession, but for a sequence of frames."""

    def __init__(self, folder: str | Path, require_correspondence: bool = True):
        self.require_corresnpondence = require_correspondence
        self.original = load_frames(folder, require_correspondence)
        self.noisy = None
        self.denoised = None
        self.mode = "original"
        self.frame_index = 0

    @property
    def frame_count(self) -> int:
        return len(self.original)

    @property
    def current_frames(self):
        if self.mode == "noisy" and self.noisy is not None:
            return self.noisy
        if self.mode == "denoised" and self.denoised is not None:
            return self.denoised
        return self.original

    @property
    def current(self):
        return self.current_frames[self.frame_index]

    def next_frame(self) -> None:
        self.frame_index = (self.frame_index + 1) % self.frame_count

    def previous_frame(self) -> None:
        self.frame_index = (self.frame_index - 1) % self.frame_count

    def show_original(self) -> None:
        self.mode = "original"

    def show_noisy(self) -> bool:
        if self.noisy is None:
            return False
        self.mode = "noisy"
        return True

    def show_denoised(self) -> bool:
        if self.denoised is None:
            return False
        self.mode = "denoised"
        return True

    def set_noise(self, noise_type: str, level: float, seed: int) -> None:
        self.noisy = add_noise_to_frames(self.original, noise_type, level, seed)
        self.denoised = None
        self.mode = "noisy"

    def set_denoised(self, frames) -> None:
        self.denoised = copy_frames(frames)
        self.mode = "denoised"
