import numpy as np
from vvrpywork.shapes import Mesh3D
from scipy import sparse 
import open3d as o3d
import scipy 
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from questions.q8_dynamic import config

def triangle_edges(triangles: np.ndarray) -> np.ndarray:
    """Return the three undirected vertex-index edges of every triangle."""
    # Get all edges from triangles
    edges = np.concatenate([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]]
    ]) # (n_triangles * 3, 2)

    # Remove duplicate edges 
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return edges


def average_edge_length(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """Return the mean length of the unique mesh edges.

    sigma/amplitude = alpha * average_edge_length.
    """
    # find the unique edges of the mesh
    edges = triangle_edges(triangles)
    # compute the lengths of the unique edges
    edge_lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    # return the mean length
    return np.mean(edge_lengths)

def face_normals_and_areas(vertices: np.ndarray, triangles: np.ndarray,) -> tuple[np.ndarray, np.ndarray]:
    """
    returns unit face normals and triangle areas of the mesh 

    """
    
    # get the three vertices of each triangle
    first = vertices[triangles[:, 0]]
    second = vertices[triangles[:, 1]]
    third = vertices[triangles[:, 2]]

    # cross product is vertical to the triangle plane 
    cross_products = np.cross(second - first, third - first)

    double_areas = np.linalg.norm(cross_products, axis=1)
    areas = 0.5 * double_areas # compute area of each triangle

    normals = np.zeros_like(cross_products, dtype=float)

    # make sure that we don't divide by zero for degenerate triangles
    valid = double_areas > np.finfo(float).eps

    # unit normal = cross_prod/||cross_prod||
    normals[valid] = (cross_products[valid]/ double_areas[valid, None]) 

    return normals, areas


def one_ring_area_weights(vertices: np.ndarray, triangles: np.ndarray,) -> np.ndarray:
    """
    Compute A_i for every vertex.

    A_i is the sum of the areas of all triangular faces incident to vertex i
    """
    _, face_areas = face_normals_and_areas(vertices, triangles,)

    # weights saves the sum of the areas of all triangular faces incident to vertex i
    weights = np.zeros(len(vertices), dtype=float) 

    np.add.at(weights, triangles[:, 0], face_areas,)
    np.add.at(weights, triangles[:, 1], face_areas,)
    np.add.at(weights, triangles[:, 2], face_areas,)

    return weights


def point_to_surface_distances(query_vertices: np.ndarray, surface_vertices: np.ndarray, surface_triangles: np.ndarray,) -> np.ndarray:
    """
    Compute the unsigned distance of every query vertex from the closest point on a triangular surface.

    (point-to-triangle surface distance)
    """
    
    # create 2 tensors for the surface vertices and triangles for the raycasting scene
    surface_vertex_tensor = o3d.core.Tensor(np.asarray(surface_vertices, dtype=np.float32), dtype=o3d.core.Dtype.Float32,)
    surface_triangle_tensor = o3d.core.Tensor(np.asarray(surface_triangles, dtype=np.uint32),dtype=o3d.core.Dtype.UInt32,)

    scene = o3d.t.geometry.RaycastingScene()

    scene.add_triangles(surface_vertex_tensor, surface_triangle_tensor,) 

    # create a tensor for the query vertices 
    query_tensor = o3d.core.Tensor(np.asarray(query_vertices, dtype=np.float32), dtype=o3d.core.Dtype.Float32,)

    # compute the closest points on the surface for every query vertex
    result = scene.compute_closest_points(query_tensor)

    closest_points = np.asarray(result["points"].numpy(), dtype=float,)

    # return the distances of every query vertex from the closest point on the surface
    return np.linalg.norm(query_vertices - closest_points, axis=1,)

def face_geometry(vertices: np.ndarray, triangles: np.ndarray,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return face centroids, unit face normals and triangle areas.

    Shapes:
        centroids: (F, 3)
        normals:   (F, 3)
        areas:     (F,)
    """
    normals, areas = face_normals_and_areas(vertices, triangles)
    centroids = np.mean(vertices[triangles], axis=1)
    return centroids, normals, areas


def surface_area(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """Return total triangle surface area."""
    _, areas = face_normals_and_areas(vertices, triangles)
    return float(np.sum(areas))

def adjacency(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    num_vertices = len(vertices)

    A = np.zeros((num_vertices, num_vertices), dtype=np.uint8)

    i = triangles[:, [0, 1, 2]].flatten()
    j = triangles[:, [1, 2, 0]].flatten()
    A[i, j] = 1
    A[j, i] = 1

    return A

def adjacency_sparse(vertices: np.ndarray, triangles: np.ndarray) -> sparse.csr_array:

    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    num_vertices = len(vertices)

    A = sparse.lil_array((num_vertices, num_vertices), dtype=np.uint8)

    i = triangles[:, [0, 1, 2]].flatten()
    j = triangles[:, [1, 2, 0]].flatten()
    A[i, j] = 1
    A[j, i] = 1

    return A.tocsr()

def degree(A: np.ndarray) -> np.ndarray:

    # the degree of each vertex is the number of adjacent vertices it has
    num_neighbors = A.sum(axis=0) # or axis=1 since A is symmetric
    # it sums the rows of A to get the degree of each vertex (num_of_neighbors=number of 1s in each row)
    D = np.zeros_like(A,dtype=np.uint8)
    np.fill_diagonal(D, num_neighbors)

    return D

def degree_sparse(A: sparse.csr_array) -> sparse.csr_array:
    degrees = np.asarray(A.sum(axis=1)).ravel().astype(float)
    return sparse.diags(degrees, format="csr")

def diagonal_inverse(mat: np.ndarray) -> np.ndarray:

    d = np.diag(mat)
    d_inv = np.diag(1/d)

    return d_inv 


def diagonal_inverse_sparse(mat: sparse.csr_array) -> sparse.csr_array:
    diagonal = np.asarray(mat.diagonal(), dtype=float)
    inverse = np.zeros_like(diagonal)
    nonzero = diagonal > 0.0
    inverse[nonzero] = 1.0 / diagonal[nonzero]
    return sparse.diags(inverse, format="csr")


def random_walk_laplacian(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:

    A = adjacency(vertices, triangles)
    D = degree(A)
    D_inv = diagonal_inverse(D)
    I = np.eye(*A.shape)

    L_RW = I - D_inv @ A
    return L_RW


def random_walk_laplacian_sparse(vertices: np.ndarray, triangles: np.ndarray) -> sparse.csr_array:

    A = adjacency_sparse(vertices,triangles)
    D = degree_sparse(A)
    D_inv = diagonal_inverse_sparse(D)
    I = sparse.eye(A.shape[0], format="csr")

    L_RW = I - D_inv @ A
    return L_RW

def delta_coordinates_sparse(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:

    x = vertices
    L_RW = random_walk_laplacian_sparse(vertices, triangles)
    delta = L_RW @ x

    return delta

def graph_laplacian(vertices: np.ndarray, triangles: np.ndarray,) -> np.ndarray:
    """Symmetric binary Laplacian L = D - C"""

    dtype = np.dtype(config.EIGEN_DTYPE)

    adjacency = adjacency_sparse(vertices, triangles,).astype(dtype)

    degree = degree_sparse(adjacency).astype(dtype)

    return (degree - adjacency).toarray().astype(dtype)


def eigendecomposition(
    laplacian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Full eigenbasis, ordered high-frequency to low-frequency."""

    eigenvalues, eigenvectors = eigh(
        laplacian,
        overwrite_a=True,
        check_finite=False,
        driver=config.EIGH_DRIVER,
    )

    eigenvalues = eigenvalues[::-1].copy()

    eigenvectors = np.ascontiguousarray(
        eigenvectors[:, ::-1],
        dtype=np.dtype(config.EIGEN_DTYPE),
    )

    return eigenvalues, eigenvectors