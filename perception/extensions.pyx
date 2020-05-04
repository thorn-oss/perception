# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
# cython: language_level=3

import math
import numpy as np
import cython
from cython.parallel import prange, parallel
from libc.stdlib cimport abort, malloc, free

cimport numpy as np
cdef extern from "limits.h":
    int INT_MAX

ctypedef np.uint8_t uint8

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_euclidean_pairwise_duplicates(int[:, :] X, float threshold):
    """Find the pairwise duplicates within an array of vectors. This
    function is faster than using scipy.spatial.distance because
    it computes distances in parallel and avoids computing full
    distances when they're not necessary.
    
    Args:
        X: The vectors with shape (N, D)
    
    Returns:
        is_duplicate: An array of length N!/(2*((N-2)!)) indicating
            whether a duplicate pair exists. The indexing matches that
            of scipy.spatial.pdist. So if N = 4, the array will represent
            comparisons of the vector indexes as follows:
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    """
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]
    cdef Py_ssize_t n_pairs = int(math.factorial(n)/(2*math.factorial(n-2)))
    
    # i_1 is the index of hash1, i_2 is the index of hash2, i_d is the
    # index of the vector dimension we're on, i_i is used to compute
    # the starting index in the flattened vector in the different threads.
    cdef Py_ssize_t i_1, i_2, i_d, i_i
    duplicate_arr = np.zeros(n_pairs, dtype=np.uint8)
    cdef np.uint8_t[:] duplicate = duplicate_arr
    cdef size_t local_buf_size = 2 # distance, flattened array offset
    cdef float threshold2 = threshold ** 2
    with nogil, parallel():
        local_buf = <int *> malloc(sizeof(int) * local_buf_size)
        if local_buf is NULL:
            abort()
        for i_1 in prange(n-1):
            local_buf[1] = 0
            for i_i in range(i_1):
                local_buf[1] += n - i_i - 1
            for i_2 in range(i_1 + 1, n):
                local_buf[0] = 0
                for i_d in range(d):
                    local_buf[0] += (X[i_1, i_d] - X[i_2, i_d]) ** 2
                    if local_buf[0] > threshold2:
                        # If we're already beyond the distance threshold,
                        # we don't need to continue computing squared
                        # distances.
                        break
                if local_buf[0] < threshold2:
                    duplicate[local_buf[1]] = 1
                # Increment the index in the flattened array.
                local_buf[1] += 1
        free(local_buf)
    return duplicate_arr