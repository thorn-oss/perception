# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
# cython: language_level=3

import numpy as np
import cython
from cython.parallel import prange, parallel
from libc.stdlib cimport abort, malloc, free
from libc.math cimport sqrt

cimport numpy as np
cdef extern from "limits.h":
    int INT_MAX

ctypedef np.uint8_t uint8

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_euclidean_metrics(int[:, :] X_noop, int[:, :] X_tran, uint8[:, :] mask):
    """Compute the positive / negative distance metrics between two sets of vectors
    using euclidean distance. This function obtains the necessary metrics roughly
    10x faster than using scipy.spatial.distance.cdist and numpy functions.
    
    Args:
        X_noop: The vectors for the noop hashes with shape (N, K)
        X_tran: The vectors for the transformed instances with shape (M, K)
        mask: A (M, N) array indicating whether noop n corresponds to transform m
    
    Returns:
        distances: An M by 2 array with the closest false positive and closest
            true positive for each transform.
        indexes: An M by 2 array with the index for the closest false positive
            noop and the closest true positive noop.
    """

    cdef Py_ssize_t n_noop = X_noop.shape[0]
    cdef Py_ssize_t d_noop = X_noop.shape[1]
    cdef Py_ssize_t n_tran = X_tran.shape[0]
    cdef Py_ssize_t d_tran = X_tran.shape[1]
    cdef Py_ssize_t n_mask_tran = mask.shape[0]
    cdef Py_ssize_t n_mask_noop = mask.shape[1]
    cdef Py_ssize_t i_mask_tran
    cdef Py_ssize_t i_mask_noop
    cdef int n_pos

    cdef int current_distance
    cdef int current_closest_fp
    cdef int current_closest_tp
    cdef int[:] x
    cdef int[:] y
    cdef uint8 is_pos
    cdef Py_ssize_t i_noop, i_tran, i_d
    cdef Py_ssize_t i_closest_fp = 0
    cdef Py_ssize_t i_closest_tp = 1
    cdef Py_ssize_t i_closest_fp_idx = 0
    cdef Py_ssize_t i_closest_tp_idx = 1
    cdef int * local_buf
    cdef size_t size = 5
    cdef float NAN
    NAN = float("NaN")

    assert d_noop == d_tran, "Dimensionality of vectors must match."
    assert n_mask_tran == n_tran, "Dimension 0 of mask must correspond to n_transforms."
    assert n_mask_noop == n_noop, "Dimension 1 of mask must correspond to n_noops."
    for i_mask_tran in range(n_mask_tran):
        n_pos = 0
        for i_mask_noop in range(n_mask_noop):
            if mask[i_mask_tran, i_mask_noop] == True:
                n_pos += 1
        assert n_pos > 0, "All transforms must have at least one positive noop."
        assert n_pos < n_mask_noop, "All transforms must have at least one negative noop."
    
    distances = np.zeros((n_tran, 2), dtype=np.float32)
    indexes = np.zeros((n_tran, 2), dtype=np.int32)
    
    cdef np.float32_t[:, :] distances_view = distances
    cdef int[:, :] indexes_view = indexes

    with nogil, parallel():
        local_buf = <int *> malloc(sizeof(int) * size)
        if local_buf is NULL:
            abort()
        for i_tran in prange(n_tran):
            local_buf[1] = INT_MAX  # Smallest false positive distance
            local_buf[2] = INT_MAX  # Smallest true positive distance
            local_buf[3] = 0        # Smallest false positive index
            local_buf[4] = 0        # Smallest true positive index
            for i_noop in range(n_noop):
                local_buf[0] = 0    # Current distance
                is_pos = mask[i_tran, i_noop] == True
                for i_d in range(d_noop):
                    local_buf[0] += (X_noop[i_noop, i_d] - X_tran[i_tran, i_d]) ** 2
                if is_pos and (local_buf[0] < local_buf[2]):
                    local_buf[2] = local_buf[0]
                    local_buf[4] = i_noop
                if not is_pos and (local_buf[0] < local_buf[1]):
                    local_buf[1] = local_buf[0]
                    local_buf[3] = i_noop
            if local_buf[3] < INT_MAX:
                distances_view[i_tran, i_closest_fp] = sqrt(local_buf[1])
            else:
                distances_view[i_tran, i_closest_fp] = NAN
            if local_buf[2] < INT_MAX:
                distances_view[i_tran, i_closest_tp] = sqrt(local_buf[2])
            else:
                distances_view[i_tran, i_closest_tp] = NAN
            indexes_view[i_tran, i_closest_fp_idx] = local_buf[3]
            indexes_view[i_tran, i_closest_tp_idx] = local_buf[4]
        free(local_buf)
    return distances, indexes