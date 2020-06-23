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
def compute_euclidean_pairwise_duplicates(int[:, :] X, float threshold, counts: int[:] = None):
    """Find the pairwise duplicates within an array of vectors, where there may be multiple
    vectors for the same file. This function is faster than using scipy.spatial.distance
    because it computes distances in parallel, avoids computing full distances when they're
    not necessary, and skips computing distances for pairs of hashes that are for the
    same file.
    
    Args:
        X: The vectors with shape (N, D). Vectors for the same file need to be
            supplied sequentially so that we can use the counts argument
            to determine which vectors are for the same file.
        counts: For each file, the number of sequential vectors in X. If not
            provided, each vector is assumed to be for a different file (i.e.,
            this is equivalent to `counts = np.ones(N)`).
    
    Returns:
        n_duplicates: An array of length M!/(2*((M-2)!)) indicating
            the number of duplicate hash pairs. The indexing matches that
            of scipy.spatial.pdist. M is the number of files. So if M = 4,
            the array will represent comparisons of the file indexes as follows:
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]. So a possible return would
            be [1, 0, 0, 2, 3] which means that there was one match between
            file 0 and file 1, two matches between file 1 and file 2, and three
            matches between file 1 and file 3.
    """
    if counts is None:
        counts_arr = np.ones(X.shape[0], dtype=np.int32)
        counts = counts_arr
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t m = counts.shape[0]
    cdef Py_ssize_t d = X.shape[1]
    cdef Py_ssize_t n_pairs = int(math.factorial(m)/(2*math.factorial(m-2)))
    
    # i_1 is the index of file1, i_2 is the index of file2, i_d is the
    # index of the vector dimension we're on, i_i is used to compute
    # the starting index in the flattened vector in the different threads.
    # i_1_subhash is the index of the hash on file1, i_2_subhash is
    # the index of the hash on file2.
    cdef Py_ssize_t i_1, i_2, i_d, i_i, i_1_sub, i_2_sub, i_1_offset
    duplicate_arr = np.zeros(n_pairs, dtype=np.double)
    cdef double[:] duplicate = duplicate_arr
    offsets_arr = np.zeros(m, dtype=np.int32)
    cdef np.int32_t[:] offsets = offsets_arr
    for i_1 in range(m):
        for i_i in range(i_1):
            offsets[i_1] += counts[i_i]
    cdef size_t local_buf_size = 4 # distance, flattened array offset, index_offset_1, index_offset_2
    cdef float threshold2 = threshold ** 2
    with nogil, parallel():
        local_buf = <int *> malloc(sizeof(int) * local_buf_size)
        if local_buf is NULL:
            abort()
        # Iterate over all of the files.
        for i_1 in prange(m-1):
            local_buf[1] = 0
            local_buf[2] = offsets[i_1]
            # Compute the index of the output vector
            # where we will count the number of duplicates.
            for i_i in range(i_1):
                local_buf[1] += m - i_i - 1
            # Iterate over all the other files to compare.
            for i_2 in range(i_1 + 1, m):
                local_buf[3] = offsets[i_2]
                # Iterate over all the hashes in file1
                for i_1_sub in range(counts[i_1]):
                    # Iterate over all the hashes in file2
                    for i_2_sub in range(counts[i_2]):
                        local_buf[0] = 0
                        for i_d in range(d):
                            local_buf[0] += (X[local_buf[2] + i_1_sub, i_d] - X[local_buf[3] + i_2_sub, i_d]) ** 2
                            if local_buf[0] > threshold2:
                                # If we're already beyond the distance threshold,
                                # we don't need to continue computing squared
                                # distances.
                                break
                        if local_buf[0] < threshold2:
                            duplicate[local_buf[1]] += 1
                # Increment the index in the flattened array.
                local_buf[1] += 1
        free(local_buf)
    return duplicate_arr