import numpy as np
from spectral import solve_and_sort_std_eigv_problem, normalized_Laplacian, unnormalized_Laplacian


def CTD_matrix(W, laplacian_type="normalized"):
    """Calculates the Commute Time Distance (CTD) matrix."""
    d = np.sum(W, axis=1)
    D = np.diag(d)
    Vol = np.sum(d)
    
    if laplacian_type == "normalized":
        L = normalized_Laplacian(W)

    elif laplacian_type == "unnormalized":
        L = unnormalized_Laplacian(W)
        
    # Solve Eigen-problem
    eigvs, eigvecs = solve_and_sort_std_eigv_problem(L)

    if np.sum(eigvs < 1e-10) > 1:
        raise ValueError(
            "Graph is disconnected - CTD requires a single connected component."
        )
        
    # Skip the first eigenvalue/vector 
    # eigvecs are columns, so eigvecs[:, 1:] gives us the relevant vectors
    vals = eigvs[1:]
    Phi = eigvecs[:, 1:]
    
    if laplacian_type == "unnormalized":
        # Standard CTE: sqrt(Vol) * Phi * diag(1/sqrt(Lambda))
        inv_sqrt_vals = 1.0 / np.sqrt(vals + 1e-10)
        CTE = np.sqrt(Vol) * (Phi * inv_sqrt_vals)

    elif laplacian_type == "normalized":
        # Normalized CTE: sqrt(Vol) * D^(-1/2) * Phi * diag(1/sqrt(Lambda))
        d_inv_sqrt = 1.0 / np.sqrt(d + 1e-12)
        inv_sqrt_vals = 1.0 / np.sqrt(vals + 1e-10)
        # Apply degree scaling to the eigenvectors
        CTE = np.sqrt(Vol) * (d_inv_sqrt[:, np.newaxis] * Phi) * inv_sqrt_vals
    
    # Vectorized Squared Euclidean Distance
    sq_norms = np.sum(CTE**2, axis=1)
    C = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * np.dot(CTE, CTE.T)
    
    return C


def asymp_CTD_matrix(W):
    # Extract the diagonal (degrees)
    d = np.sum(W, axis=1) 
    Vol = np.sum(d)
    
    # Inverse degrees
    inv_d = 1.0 / (d + 1e-12) 
    
    # Broadcasting: (N, 1) + (1, N) creates the (N, N) matrix
    C = Vol * (inv_d[:, np.newaxis] + inv_d[np.newaxis, :])
    
    # Distance to self is 0
    np.fill_diagonal(C, 0)
    return C

    