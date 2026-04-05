import numpy as np
from collections import Counter


def normalize(list_values, norm_type, Vol=None):
    arr = np.asarray(list_values)
    if norm_type == "scale_and_shift":
        v_min, v_max = arr.min(), arr.max()
        return (arr - v_min) / (v_max - v_min) if v_max != v_min else arr
    if norm_type == "norm_wrt_volume":
        if Vol is None or Vol == 0:
            raise ValueError("Vol must be provided and non-zero for 'norm_wrt_volume'")
        return arr / Vol
    if norm_type == "norm_wrt_avg_ctd":
        if arr_mean == 0:
            raise ValueError("Mean of input values must be non-zero for 'norm_wrt_avg_ctd'")
        return arr / np.mean(arr)
    return arr


def Kruglov_distance(vi, vj):
    """Calculates the distance between two cumulative distributions."""
    u = np.unique(np.concatenate([vi, vj])) # this is sorted
    Ni, Nj = len(vi), len(vj)
    
    vi_counter = Counter(vi)
    vj_counter = Counter(vj)
    
    vi_count = 0
    vj_count = 0
    cdf_distance = 0
    
    # Compute area between CDFs
    for i in range(len(u) - 1):
        vi_count += vi_counter.get(u[i], 0)
        vj_count += vj_counter.get(u[i], 0)
        
        abs_dif = abs((vi_count / Ni) - (vj_count / Nj))
        cdf_distance += (u[i+1] - u[i]) * abs_dif
        
    return cdf_distance

    