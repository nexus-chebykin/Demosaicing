from __future__ import annotations

import time

import colour
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from colour.hints import ArrayLike, NDArray
from colour.utilities import tstack
from colour_demosaicing.bayer import masks_CFA_Bayer, mosaicing_CFA_Bayer
from scipy.ndimage import convolve, convolve1d

colour.utilities.filter_warnings(*[True] * 4)


@nb.njit(parallel=True, fastmath=False)
def conv_h(matrix, filter) -> NDArray:
    m = matrix.shape
    n = filter.shape[0]
    out = np.empty_like(matrix)
    for i in nb.prange(m[0]):
        arr = np.empty(m[1] + n - 1, dtype=matrix.dtype)
        arr[n//2:-(n//2)] = matrix[i, :]
        arr[:n//2] = matrix[i, n//2:0:-1]
        arr[-(n//2):] = matrix[i, -2:-(n//2)-2:-1]
        result_v = np.empty((n, m[1]), dtype=matrix.dtype)
        for j in range(n):
            result_v[j, :] = arr[j:j + m[1]] * filter[j]
        out[i] = np.sum(result_v, axis=0)
    return out

# def conv_h(x: ArrayLike, y: ArrayLike) -> NDArray:
#     return convolve1d(x, y, mode="mirror", axis=1)
@nb.njit(parallel=True, fastmath=False)
def conv_v(x, y) -> NDArray:
    return conv_h(x.T, y).T
    # return convolve1d(x, y, mode="mirror", axis=0)

@nb.jit(parallel=True, fastmath=False)
def produce_green(CFA, R_mask, G_mask, B_mask, R, G, B, save_vertical_edges):
    green_filter = np.array([-0.25, 0.5, 0.5, 0.5, -0.25], dtype=np.float32)
    classifier_convolution_matrix = np.array(
        [
            [1.0, 0.0, 3.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [3.0, 0.0, 3.0, 0.0, 3.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 3.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    G_H = np.where(~G_mask, conv_h(CFA, green_filter), G)
    G_V = np.where(~G_mask, conv_v(CFA, green_filter), G)

    C_H = np.where(R_mask, R - G_H, 0)
    C_H = np.where(B_mask, B - G_H, C_H)

    C_V = np.where(R_mask, R - G_V, 0)
    C_V = np.where(B_mask, B - G_V, C_V)

    C_H_padded = np.pad(C_H[:, 2:], ((0, 0), (0, 2)), mode="reflect")
    C_V_padded = np.pad(C_V[2:, :], ((0, 2), (0, 0)), mode="reflect")

    D_H = np.abs(C_H - C_H_padded)
    D_V = np.abs(C_V - C_V_padded)

    d_H = convolve(D_H, classifier_convolution_matrix, mode="mirror")
    d_V = convolve(D_V, classifier_convolution_matrix, mode="mirror")

    vertical_edge = d_V < d_H
    # vertical_edge = d_V >= d_H # WRONG!
    horizontal_edge = ~vertical_edge
    if save_vertical_edges:
        plt.imsave('vertical_edge.png', vertical_edge, cmap='gray')
    G = np.where(vertical_edge, G_V, G_H)
    return horizontal_edge, G
@nb.jit(parallel=True, fastmath=False)
def red_in_green(R_mask, G_mask, B_mask, R, G, B, horizontal_edge):

    half_sum = np.array([0.5, 0, 0.5], dtype=np.float32)
    blue_column = np.empty(R.shape[1], dtype=np.bool_)
    for i in range(R.shape[1]):
        blue_column[i] = B_mask[0, i] or B_mask[1, i]
    red_column = np.logical_not(blue_column)
    R = np.where(
        G_mask & blue_column,
        G + conv_h(R - G, half_sum),
        R
    )

    R = np.where(
        G_mask & red_column,
        G + conv_v(R - G, half_sum),
        R
    )

    B = np.where(
        G_mask & red_column,
        G + conv_h(B - G, half_sum),
        B
    )

    B = np.where(
        G_mask & blue_column,
        G + conv_v(B - G, half_sum),
        B
    )
    return R, B, blue_column, red_column
@nb.njit(parallel=True, fastmath=False)
def red_in_blue(R_mask, G_mask, B_mask, R, G, B, horizontal_edge):
    half_sum = np.array([0.5, 0, 0.5], dtype=np.float32)
    B = np.where(
        R_mask,
        np.where(
            horizontal_edge,
            R + conv_h(B - R, half_sum),
            R + conv_v(B - R, half_sum)
        ),
        B

    )
    R = np.where(
        B_mask,
        np.where(
            horizontal_edge,
            B + conv_h(R - B, half_sum),
            B + conv_v(R - B, half_sum)
        ),
        R
    )
    return R, B

@nb.njit(parallel=True, fastmath=False)
def enhance_green_in_red(R_mask, G_mask, B_mask, R, G, B, horizontal_edge):
    low_pass_filter = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
    G_l = np.where(
        horizontal_edge,
        conv_h(G, low_pass_filter),
        conv_v(G, low_pass_filter),
    )
    R_l = np.where(
        horizontal_edge,
        conv_h(R, low_pass_filter),
        conv_v(R, low_pass_filter),
    )
    B_l = np.where(
        horizontal_edge,
        conv_h(B, low_pass_filter),
        conv_v(B, low_pass_filter),
    )

    R_h = R - R_l
    B_h = B - B_l

    G = np.where(
        R_mask,
        G_l + R_h,
        G
    )
    G = np.where(
        B_mask,
        G_l + B_h,
        G
    )
    return G
@nb.njit(parallel=True, fastmath=False)
def enhance_red_in_green(R_mask, G_mask, B_mask, R, G, B, horizontal_edge, blue_column, red_column):
    low_pass_filter = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
    G_l = np.where(
        np.broadcast_to(blue_column, G.shape),
        conv_h(G, low_pass_filter),
        conv_v(G, low_pass_filter),
    )
    R_l = np.where(
        np.broadcast_to(blue_column, R.shape),
        conv_h(R, low_pass_filter),
        conv_v(R, low_pass_filter),
    )
    G_h = G - G_l
    R = np.where(
        G_mask,
        R_l + G_h,
        R
    )

    G_l = np.where(
        np.broadcast_to(red_column, G.shape),
        conv_h(G, low_pass_filter),
        conv_v(G, low_pass_filter),
    )
    B_l = np.where(
        np.broadcast_to(red_column, B.shape),
        conv_h(B, low_pass_filter),
        conv_v(B, low_pass_filter),
    )
    G_h = G - G_l
    B = np.where(
        G_mask,
        B_l + G_h,
        B
    )
    return R, B
@nb.njit(parallel=True, fastmath=False)
def enhance_red_in_blue(R_mask, G_mask, B_mask, R, G, B, horizontal_edge):
    low_pass_filter = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
    R_l = np.where(
        horizontal_edge,
        conv_h(R, low_pass_filter),
        conv_v(R, low_pass_filter),
    )
    B_l = np.where(
        horizontal_edge,
        conv_h(B, low_pass_filter),
        conv_v(B, low_pass_filter),
    )

    R_h = R - R_l
    B_h = B - B_l

    R = np.where(
        B_mask,
        R_l + B_h,
        R
    )
    B = np.where(
        R_mask,
        B_l + R_h,
        B
    )
    return R, B
@nb.jit(parallel=True, fastmath=False)
def do_everything(CFA, R_mask = None, G_mask = None, B_mask = None, should_enhance=True, save_vertical_edges=False, out_file=None):
    if R_mask is None:
        R_mask, G_mask, B_mask = masks_CFA_Bayer(CFA.shape, "BGGR")
    R, G, B = CFA * R_mask, CFA * G_mask, CFA * B_mask
    horizontal_edge, G = produce_green(CFA, R_mask, G_mask, B_mask, R, G, B, save_vertical_edges)
    R, B, blue_column, red_column = red_in_green(R_mask, G_mask, B_mask, R, G, B, horizontal_edge)
    R, B = red_in_blue(R_mask, G_mask, B_mask, R, G, B, horizontal_edge)
    if should_enhance:
        R, G, B = enhance(R_mask, G_mask, B_mask, R, G, B, horizontal_edge, blue_column, red_column)
    RGB = tstack((R, G, B))
    RGB.clip(0, 1, RGB)
    if out_file is not None:
        colour.write_image(RGB, out_file)
    return RGB

@nb.njit(parallel=True)
def enhance(R_mask, G_mask, B_mask, R, G, B, horizontal_edge, blue_column, red_column):
    G = enhance_green_in_red(R_mask, G_mask, B_mask, R, G, B, horizontal_edge)
    R, B = enhance_red_in_green(R_mask, G_mask, B_mask, R, G, B, horizontal_edge, blue_column, red_column)
    R, B = enhance_red_in_blue(R_mask, G_mask, B_mask, R, G, B, horizontal_edge)
    return R, G, B




img = colour.read_image('Lighthouse.png')
img = mosaicing_CFA_Bayer(img, 'BGGR')
do_everything(img.copy())
t = time.perf_counter()
for i in range(60):
    do_everything(img.copy())
print(time.perf_counter() - t)
