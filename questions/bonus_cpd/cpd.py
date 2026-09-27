"""A direct NumPy/SciPy implementation of non-rigid CPD."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve


@dataclass
class CPDResult:
    """Values needed by the sequence registration and two-frame viewer."""

    weights: np.ndarray
    weight_history: list[np.ndarray]


def gaussian_kernel(X, Y, beta):
    """Compute K[i,j] = exp(-||X[i]-Y[j]||^2 / (2 beta^2))."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must be point arrays with the same dimension.")

    differences = X[:, None, :] - Y[None, :, :]
    squared_distances = np.sum(differences**2, axis=2)
    return np.exp(-squared_distances / (2.0 * beta**2))


def initial_variance(X, Y):
    """Compute the initial CPD variance between all pairs of points."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    number_of_target_points, dimension = X.shape
    number_of_moving_points = len(Y)

    differences = Y[:, None, :] - X[None, :, :]
    return float(
        np.sum(differences**2)
        / (dimension * number_of_target_points * number_of_moving_points)
    )


def correspondence_probabilities(X, transformed_Y, sigma2, outlier_weight, epsilon=1e-8):
    """Perform the CPD E-step and return P with shape (M, N)."""
    X = np.asarray(X, dtype=float)
    transformed_Y = np.asarray(transformed_Y, dtype=float)

    number_of_target_points, dimension = X.shape
    number_of_moving_points = len(transformed_Y)

    if not 0.0 <= outlier_weight < 1.0:
        raise ValueError("outlier_weight must satisfy 0 <= w < 1.")

    sigma2 = max(float(sigma2), epsilon)
    differences = transformed_Y[:, None, :] - X[None, :, :]
    squared_distances = np.sum(differences**2, axis=2)
    numerator = np.exp(-squared_distances / (2.0 * sigma2))

    if outlier_weight == 0.0:
        outlier_constant = 0.0
    else:
        outlier_constant = (
            (2.0 * np.pi * sigma2) ** (dimension / 2.0)
            * outlier_weight
            / (1.0 - outlier_weight)
            * number_of_moving_points
            / number_of_target_points
        )

    # Every column corresponds to one target point x_n.
    denominator = np.sum(numerator, axis=0, keepdims=True) + outlier_constant
    return numerator / np.maximum(denominator, epsilon)


def registration_variance(X, transformed_Y, probabilities, epsilon=1e-8):
    """Evaluate the trace formula for sigma^2 from the CPD paper."""
    number_of_target_points, dimension = X.shape
    number_of_moving_points = len(transformed_Y)

    if probabilities.shape != (number_of_moving_points, number_of_target_points):
        raise ValueError("The probability matrix has the wrong shape.")

    P1 = np.sum(probabilities, axis=1)
    Pt1 = np.sum(probabilities, axis=0)
    effective_points = float(np.sum(P1))

    if effective_points <= epsilon:
        raise ValueError("CPD found no effective correspondences.")

    PX = probabilities @ X

    # Efficient equivalents of the three trace terms in the paper.
    target_term = np.sum(Pt1 * np.sum(X**2, axis=1))
    cross_term = np.sum(PX * transformed_Y)
    moving_term = np.sum(P1 * np.sum(transformed_Y**2, axis=1))

    sigma2 = (
        target_term - 2.0 * cross_term + moving_term
    ) / (effective_points * dimension)

    return max(float(sigma2), epsilon)


def transform_points(points, control_points, weights, beta):
    """Apply a learned CPD deformation field to arbitrary points."""
    kernel = gaussian_kernel(points, control_points, beta)
    return np.asarray(points, dtype=float) + kernel @ weights


def relative_change(old_value, new_value, epsilon=1e-12):
    """Relative scalar change used by the convergence test."""
    return abs(float(old_value) - float(new_value)) / max(abs(float(old_value)), epsilon)


def nonrigid_cpd(
    target_points,
    moving_points,
    beta,
    regularization,
    outlier_weight,
    max_iterations=50,
    tolerance=1e-5,
    epsilon=1e-8,
):
    """Register moving_points onto target_points with non-rigid CPD."""
    X = np.asarray(target_points, dtype=float)
    Y = np.asarray(moving_points, dtype=float)

    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("target_points and moving_points must have the same dimension.")
    if len(X) == 0 or len(Y) == 0:
        raise ValueError("The CPD point sets cannot be empty.")
    if regularization <= 0.0:
        raise ValueError("regularization must be positive.")

    number_of_moving_points, dimension = Y.shape
    deformation_kernel = gaussian_kernel(Y, Y, beta)
    weights = np.zeros((number_of_moving_points, dimension), dtype=float)
    sigma2 = max(initial_variance(X, Y), epsilon)

    weight_history = [weights.copy()]

    for _ in range(max_iterations):
        transformed_Y = Y + deformation_kernel @ weights

        probabilities = correspondence_probabilities(
            X,
            transformed_Y,
            sigma2,
            outlier_weight,
            epsilon,
        )

        P1 = np.sum(probabilities, axis=1)
        PX = probabilities @ X

        # M-step from the CPD paper:
        # [diag(P1) G + lambda sigma^2 I] W = PX - diag(P1) Y
        A = (
            P1[:, None] * deformation_kernel
            + regularization * sigma2 * np.eye(number_of_moving_points)
        )
        B = PX - P1[:, None] * Y
        weights = solve(A, B)

        transformed_Y = Y + deformation_kernel @ weights
        new_sigma2 = registration_variance(X, transformed_Y, probabilities, epsilon)
        change = relative_change(sigma2, new_sigma2, epsilon)
        sigma2 = new_sigma2

        weight_history.append(weights.copy())

        if change < tolerance:
            break

    return CPDResult(
        weights=weights,
        weight_history=weight_history,
    )
