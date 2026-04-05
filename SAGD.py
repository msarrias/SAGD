import numpy as np
from distances import CTD_matrix
from stats import normalize, Kruglov_distance

def SAGD(W_i, W_j, norm_type):
    """Compute Shape-Aware Graph Distance."""
    # Process Graph i
    C_Gi = CTD_matrix(W_i)
    triu_i = C_Gi[np.triu_indices(W_i.shape[0], k=1)]
    # Assuming volume-based norm uses total weight sum
    norm_i = normalize(triu_i, norm_type, Vol=np.sum(W_i))

    # Process Graph j
    C_Gj = CTD_matrix(W_j)
    triu_j = C_Gj[np.triu_indices(W_j.shape[0], k=1)]
    norm_j = normalize(triu_j, norm_type, Vol=np.sum(W_j))
    
    return Kruglov_distance(norm_i, norm_j)