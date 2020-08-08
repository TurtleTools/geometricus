import numba as nb
import numpy as np

NUM_MOMENTS = 4


@nb.njit
def nb_mean_axis_0(array: np.ndarray) -> np.ndarray:
    """
    Same as np.mean(array, axis=0) but njitted
    """
    mean_array = np.zeros(array.shape[1])
    for i in range(array.shape[1]):
        mean_array[i] = np.mean(array[:, i])
    return mean_array


@nb.njit(cache=False)
def mu(p, q, r, coords, centroid):
    """
    Central moment
    """
    return np.sum(
        ((coords[:, 0] - centroid[0]) ** p)
        * ((coords[:, 1] - centroid[1]) ** q)
        * ((coords[:, 2] - centroid[2]) ** r)
    )


@nb.njit
def get_second_order_moments(coords: np.ndarray):
    centroid = nb_mean_axis_0(coords)
    mu_200 = mu(2.0, 0.0, 0.0, coords, centroid)
    mu_020 = mu(0.0, 2.0, 0.0, coords, centroid)
    mu_002 = mu(0.0, 0.0, 2.0, coords, centroid)

    j1 = mu_200 + mu_020 + mu_002

    mu_110 = mu(1.0, 1.0, 0.0, coords, centroid)
    mu_101 = mu(1.0, 0.0, 1.0, coords, centroid)
    mu_011 = mu(0.0, 1.0, 1.0, coords, centroid)

    mu_003 = mu(0.0, 0.0, 3.0, coords, centroid)
    mu_012 = mu(0.0, 1.0, 2.0, coords, centroid)
    mu_021 = mu(0.0, 2.0, 1.0, coords, centroid)
    mu_030 = mu(0.0, 3.0, 0.0, coords, centroid)
    mu_102 = mu(1.0, 0.0, 2.0, coords, centroid)
    mu_111 = mu(1.0, 1.0, 1.0, coords, centroid)
    mu_210 = mu(2.0, 1.0, 0.0, coords, centroid)
    mu_201 = mu(2.0, 0.0, 1.0, coords, centroid)
    mu_120 = mu(1.0, 2.0, 0.0, coords, centroid)
    mu_300 = mu(3.0, 0.0, 0.0, coords, centroid)

    j2 = (
        mu_200 * mu_020
        + mu_200 * mu_002
        + mu_020 * mu_002
        - mu_110 ** 2
        - mu_101 ** 2
        - mu_011 ** 2
    )
    j3 = (
        mu_200 * mu_020 * mu_002
        + 2 * mu_110 * mu_101 * mu_011
        - mu_002 * mu_110 ** 2
        - mu_020 * mu_101 ** 2
        - mu_200 * mu_011 ** 2
    )
    j4 = (
        mu_003 ** 2
        + 6 * mu_012 ** 2
        + 6 * mu_021 ** 2
        + mu_030 ** 2
        + 6 * mu_102 ** 2
        + 15 * mu_111 ** 2
        - 3 * mu_102 * mu_120
        + 6 * mu_120 ** 2
        - 3 * mu_021 * mu_201
        + 6 * mu_201 ** 2
        - 3 * mu_003 * (mu_021 + mu_201)
        - 3 * mu_030 * mu_210
        + 6 * mu_210 ** 2
        - 3 * mu_012 * (mu_030 + mu_210)
        - 3 * mu_102 * mu_300
        - 3 * mu_120 * mu_300
        + mu_300 ** 2
    )
    return j1, j2, j3, j4
