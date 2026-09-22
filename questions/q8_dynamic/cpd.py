import numpy as np
from scipy.linalg import solve


def gaussian_kernel(X, Y, beta):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    differences = X[:, None, :] - Y[None, :, :]
    squared_distances = np.sum(differences**2, axis=2)

    return np.exp(-squared_distances / (2.0 * beta**2))


def initial_variance(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    N, D = X.shape
    M, _ = Y.shape

    differences = Y[:, None, :] - X[None, :, :] # (M, N, D)

    sigma2 = np.sum(differences**2) / (D * N * M) 

    return float(sigma2)


def correspondence_probabilities(X, TY, sigma2, outlier_weight, epsilon=1e-8,):
    X = np.asarray(X, dtype=float)
    TY = np.asarray(TY, dtype=float)

    N, D = X.shape
    M, _ = TY.shape

    sigma2 = max(float(sigma2), epsilon) # Ensure sigma2 is positive

    differences = TY[:, None, :] - X[None, :, :] # (M, N, D)

    squared_distances = np.sum(differences**2, axis=2)

    numerator = np.exp(-squared_distances / (2.0 * sigma2)) # (M, N)

    # Outlier constant from CPD.

    w = float(outlier_weight)
    if w == 0.0:
        outlier_constant = 0.0
    else:
        outlier_constant = ((2.0 * np.pi * sigma2) ** (D / 2.0) * w / (1.0 - w) * M / N)

    # Shape: (1, N)
    denominator = (np.sum(numerator, axis=0, keepdims=True) + outlier_constant) # (1, N)

    P = numerator / np.maximum(denominator, epsilon)

    return P

def update_sigma2(X, TY, P, epsilon=1e-8,):
    N, D = X.shape
    M, dimension_y = TY.shape

    # P 1_N
    P1 = np.sum(P, axis=1)   # (M,)

    # P^T 1_M
    Pt1 = np.sum(P, axis=0)  # (N,)

    Np = np.sum(P1)

    # P X
    PX = P @ X  # (M, D)

    # tr(X^T diag(Pt1) X)
    term1 = np.sum(Pt1 * np.sum(X**2, axis=1))

    # tr((PX)^T TY)
    term2 = np.sum(PX * TY)

    # tr(TY^T diag(P1) TY)
    term3 = np.sum(P1 * np.sum(TY**2, axis=1))

    sigma2 = (term1 - 2.0 * term2 + term3) / (Np * D)

    return max(float(sigma2), epsilon)

def relative_change(old_value, new_value, epsilon=1e-12):
    denominator = max(abs(float(old_value)), epsilon)

    return (abs(float(old_value) - float(new_value)) / denominator)

def nonrigid_cpd(X, Y, beta, lambda_reg, outlier_weight, max_iterations=50, tolerance=1e-5, epsilon=1e-8,):

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    N, D = X.shape
    M, _ = Y.shape

    W = np.zeros((M, D), dtype=float)
    sigma2 = max(initial_variance(X, Y), epsilon,)

    # Deformation kernel between the moving control points.
    G = gaussian_kernel(Y, Y, beta) # (M, M)

    for iteration in range(max_iterations):
        # current non-rigid deformation.
        TY = Y + G @ W # (M, D)

        # E-step.
        P = correspondence_probabilities(X=X, TY=TY, sigma2=sigma2, outlier_weight=outlier_weight, epsilon=epsilon,) # (M, N)

        # P1[m] = sum_n P[m,n]
        P1 = np.sum(P, axis=1) # (M,)

        # PX[m] = sum_n P[m,n] X[n]
        PX = P @ X # (M, D)

        # M-step:
        #
        # [diag(P1) G + lambda sigma² I] W = PX - diag(P1) Y
        # P1[:, None] * G same with diag(P1) @ G.

        A = (P1[:, None] * G + lambda_reg * sigma2 * np.eye(M))
        B = PX - P1[:, None] * Y

        # solve A W = B.
        # A: (M, M)
        # B: (M, D)
        # W: (M, D)
        W = solve(A, B)

        # new transformed points.
        TY = Y + G @ W

        # new variance.
        new_sigma2 = update_sigma2(X=X, TY=TY, P=P, epsilon=epsilon,)

        change = relative_change(old_value=sigma2, new_value=new_sigma2, epsilon=epsilon,)

        sigma2 = max(new_sigma2, epsilon)

        print(
            f"Iteration {iteration + 1:3d}: "
            f"sigma²={sigma2:.8g}, "
            f"relative change={change:.3e}"
        )

        if change < tolerance:
            break

    # aligned point set: final transformed points.
    TY = Y + G @ W

    P = correspondence_probabilities(X=X, TY=TY, sigma2=sigma2, outlier_weight=outlier_weight, epsilon=epsilon,)

    return TY, W, sigma2, P