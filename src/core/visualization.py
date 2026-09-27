"""SUPPORT CODE — drawing helpers only."""

import numpy as np
from matplotlib import colormaps as cm

from vvrpywork.constants import Color
from vvrpywork.shapes import LineSet3D


def gray_colors(count: int) -> np.ndarray:
    return np.full((count, 3), 0.65)


def set_axes_visible(plotter, visible: bool) -> bool:
    """Show or hide the VTK axes actor created by VVRPyWork.

    The function works at application level, so the VVRPyWork source code does
    not have to be changed.  It returns False only when no axes actor exists.
    """
    view_properties = plotter.renderer.GetViewProps()
    view_properties.InitTraversal()

    axes_found = False
    for _ in range(view_properties.GetNumberOfItems()):
        actor = view_properties.GetNextProp()
        if actor is not None and actor.IsA("vtkAxesActor"):
            actor.SetVisibility(visible)
            axes_found = True

    if axes_found:
        plotter.render()

    return axes_found


def heatmap_colors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.empty((0, 3), dtype=float)

    minimum = np.min(values)
    maximum = np.max(values)
    if maximum == minimum:
        normalized = np.zeros_like(values)
    else:
        normalized = (values - minimum) / (maximum - minimum)
    return cm["plasma"](normalized)[:, :3]


def normal_lines(mesh, normal_length: float = 0.03, max_lines: int = 1000) -> LineSet3D:
    vertices = np.asarray(mesh.vertices, dtype=float)
    normals = np.asarray(mesh.vertex_normals, dtype=float)

    if max_lines <= 0:
        raise ValueError("max_lines must be positive.")

    step = max(1, int(np.ceil(len(vertices) / max_lines)))
    ids = np.arange(0, len(vertices), step)
    start = vertices[ids]
    end = start + normal_length * normals[ids]

    points = np.vstack((start, end))
    n = len(start)
    lines = np.column_stack((np.arange(n), np.arange(n) + n))
    return LineSet3D(points=points, lines=lines, width=1, color=Color.RED)
