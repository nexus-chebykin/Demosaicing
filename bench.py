from timeit import timeit

import numba
import numpy
from numpy.typing import NDArray
import numba as nb
from numba import prange, float64, float32, void
import numpy as np
from scipy.ndimage import convolve1d


@nb.njit(parallel=True, fastmath=False)
def _cnv_h_numba_parallel(matrix, filter) -> NDArray:
    """Perform horizontal convolution of 2D matrix and 1D filter."""
    m = matrix.shape
    n = filter.shape[0]
    out = np.empty_like(matrix)
    padded_matrix = np.empty((m[0], m[1] + n - 1), dtype=matrix.dtype)
    padded_matrix[:, n // 2:-(n // 2)] = matrix
    padded_matrix[:, :n // 2] = matrix[:, n // 2:0:-1]
    padded_matrix[:, -(n // 2):] = matrix[:, -2:-(n // 2) - 2:-1]
    for i in prange(m[0]):
        for j in prange(m[1]):
            out[i, j] = np.dot(padded_matrix[i, j:j + n], filter)
    return out


@nb.guvectorize([void(float32[:], float32[:], float32[:])], "(n),(m)->(n)", nopython=True, target="parallel")
def _cnv_h_numba_vectorize(vector, filter, out):
    """Perform horizontal convolution of 1D vector and 1D filter."""
    m = vector.shape[0]
    n = filter.shape[0]
    padded_matrix = np.empty((m + n - 1), dtype=vector.dtype)
    padded_matrix[n // 2:-(n // 2)] = vector
    padded_matrix[:n // 2] = vector[n // 2:0:-1]
    padded_matrix[-(n // 2):] = vector[-2:-(n // 2) - 2:-1]
    for j in prange(m):
        out[j] = np.dot(padded_matrix[j:j + n], filter)


@nb.njit()
def _cnv_h_numba(matrix, filter) -> NDArray:
    """Perform horizontal convolution of 2D matrix and 1D filter."""

    m = matrix.shape
    n = filter.shape[0]
    out = np.empty_like(matrix)
    padded_matrix = np.empty((m[0], m[1] + n - 1), dtype=matrix.dtype)
    padded_matrix[:, n // 2:-(n // 2)] = matrix
    padded_matrix[:, :n // 2] = matrix[:, n // 2:0:-1]
    padded_matrix[:, -(n // 2):] = matrix[:, -2:-(n // 2) - 2:-1]
    for i in prange(m[0]):
        for j in prange(m[1]):
            out[i, j] = np.dot(padded_matrix[i, j:j + n], filter)
    return out


@nb.njit(parallel=True, fastmath=False)
def _cnv_h_numba2(matrix, filter) -> NDArray:
    m = matrix.shape
    n = filter.shape[0]
    out = np.empty_like(matrix)
    for i in prange(m[0]):
        arr = np.empty(m[1] + n - 1, dtype=matrix.dtype)
        arr[n//2:-(n//2)] = matrix[i, :]
        arr[:n//2] = matrix[i, n//2:0:-1]
        arr[-(n//2):] = matrix[i, -2:-(n//2)-2:-1]
        result_v = np.empty((n, m[1]), dtype=matrix.dtype)
        for j in range(n):
            result_v[j, :] = arr[j:j + m[1]] * filter[j]
        out[i] = np.sum(result_v, axis=0)
    return out


def _cnv_h_scipy(matrix, filter) -> NDArray:
    return convolve1d(matrix, filter, mode="mirror", axis=1)


# @nb.jit(parallel=True, fastmath=False)
# def _cnv_h_numpy(matrix, filter) -> NDArray:
#     result = np.empty_like(matrix)
#     for i in prange(matrix.shape[0]):
#         result[i, :] = np.convolve(matrix[i, :], filter, mode="same")


@nb.njit(parallel=True, fastmath=False)
def my_where(bool_mask, data, data2):
    result = np.empty_like(data)
    for i in prange(data.shape[0]):
        for j in range(data.shape[1]):
            if bool_mask[i, j]:
                result[i, j] = data[i, j]
            else:
                result[i, j] = data2[i, j]
    return result


def default_where(bool_mask, data, data2):
    return np.where(bool_mask, data, data2)


# generate random float32 data
# bool_mask = np.random.rand(2000, 4000).astype(np.bool)
# print(bool_mask)
data = np.random.rand(100000, 4000).astype(np.float32)
# data2 = np.random.rand(2000, 4000).astype(np.float32)
# my_where(bool_mask, data, data2)
# print(timeit(lambda: my_where(bool_mask, data, data2), number=500))
# print(timeit(lambda: default_where(bool_mask, data, data2), number=500))
filter = np.array([-0.25, 0.5, 0.5, 0.5, -0.25], dtype=np.float32)
# _cnv_h_numba(data, filter)
_cnv_h_numba_parallel(data, filter)
_cnv_h_numba2(data, filter)
# _cnv_h_numba_vectorize(data, filter)
# _cnv_h_numpy(data, filter)
# _cnv_h_numba_stencil(data)
ktests = 10
print(timeit(lambda: _cnv_h_scipy(data, filter), number=ktests))
print(timeit(lambda: _cnv_h_numba_parallel(data, filter), number=ktests))
print(timeit(lambda: _cnv_h_numba2(data, filter), number=ktests))
# print(timeit(lambda: _cnv_h_numba_parallel(data, filter), number=200))
# print(timeit(lambda: _cnv_h_numba_vectorize(data, filter), number=1))
# print(timeit(lambda: _cnv_h_numba_stencil(data), number=5))
# _cnv_h_numba_parallel.parallel_diagnostics(level=4)
