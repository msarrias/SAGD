import numpy as np
from distances import CTD_matrix
from stats import normalize, Kruglov_distance

def SAGD(W_i, W_j, laplacian_type="unnormalized", norm_type="norm_wrt_avg_ctd"):
    """Compute Shape-Aware Graph Distance."""
    # Process Graph i
    C_Gi = CTD_matrix(W=W_i, laplacian_type=laplacian_type)
    triu_i = C_Gi[np.triu_indices(W_i.shape[0], k=1)]
    # Assuming volume-based norm uses total weight sum
    norm_i = normalize(list_values=triu_i, norm_type=norm_type, Vol=np.sum(W_i))

    # Process Graph j
    C_Gj = CTD_matrix(W=W_j, laplacian_type=laplacian_type)
    triu_j = C_Gj[np.triu_indices(W_j.shape[0], k=1)]
    norm_j = normalize(list_values=triu_j, norm_type=norm_type, Vol=np.sum(W_j))
    
    return Kruglov_distance(vi=norm_i, vj=norm_j)