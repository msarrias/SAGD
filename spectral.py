import numpy as np
import scipy.linalg as la
from collections import Counter


def unnormalized_Laplacian(W):
    D = np.diag(np.sum(W, axis=1))
    return D - W


def normalized_Laplacian(W):
    d = np.sum(W, axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d + 1e-12)
    D_inv_sqrt = np.diag(d_inv_sqrt) 
    #L_sym = I - D^(-1/2) W D^(-1/2)
    return np.eye(len(W)) - D_inv_sqrt @ W @ D_inv_sqrt


def solve_and_sort_std_eigv_problem(matrix):
    """Solves the eigenvalue problem and returns sorted (values, vectors)."""
    eigv, eigvc = la.eigh(matrix) # Use eigh for symmetric matrices
    # eigvc[:, i] is the i-th eigenvector. 
    # la.eigh returns them sorted by eigenvalue.
    return eigv, eigvc