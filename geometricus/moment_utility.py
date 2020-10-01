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
    mu_000 = mu(0., 0., 0., coords, centroid)

    mu_200 = mu(2.0, 0.0, 0.0, coords, centroid)
    mu_020 = mu(0.0, 2.0, 0.0, coords, centroid)
    mu_002 = mu(0.0, 0.0, 2.0, coords, centroid)

    j1 = (mu_200 + mu_020 + mu_002) / mu_000

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
    ) / mu_000 ** 2
    j3 = (
        mu_200 * mu_020 * mu_002
        + 2 * mu_110 * mu_101 * mu_011
        - mu_002 * mu_110 ** 2
        - mu_020 * mu_101 ** 2
        - mu_200 * mu_011 ** 2
    ) / mu_000 ** 3
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
    ) / mu_000 ** 2

    j5 = (
        mu_200 ** 3
        + 3 * mu_200 * mu_110 ** 2
        + 3 * mu_200 * mu_101 ** 2
        + 3 * mu_110 ** 2 * mu_020
        + 3 * mu_101 ** 2 * mu_002
        + mu_020 ** 3
        + 3 * mu_020 * mu_011 ** 2
        + 3 * mu_011 ** 2 * mu_002
        + mu_002 ** 3
        + 6 * mu_110 * mu_101 * mu_011
    ) / mu_000 ** 5
    return j1, j2, j3, j5


def alpha(index, coords, centroid):
    mu_200 = mu(2.0, 0.0, 0.0, coords, centroid)
    mu_020 = mu(0.0, 2.0, 0.0, coords, centroid)
    mu_002 = mu(0.0, 0.0, 2.0, coords, centroid)

    if index == 1:
        return mu_002 - mu_020
    elif index == 2:
        return mu_020 - mu_200
    else:
        return mu_200 - mu_002


def beta(index, coords, centroid):
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

    if index == 1:
        return mu_021 - mu_201
    elif index == 2:
        return mu_102 - mu_120
    elif index == 3:
        return mu_210 - mu_012
    elif index == 4:
        return mu_003 - mu_201 - 2 * mu_021
    elif index == 5:
        return mu_030 - mu_012 - 2 * mu_210
    elif index == 6:
        return mu_300 - mu_120 - 2 * mu_102
    elif index == 7:
        return mu_021 - mu_003 + 2 * mu_201
    elif index == 8:
        return mu_102 - mu_300 + 2 * mu_120
    elif index == 9:
        return mu_210 - mu_030 + 2 * mu_012
    elif index == 10:
        return mu_021 + mu_201 - 3 * mu_003
    elif index == 11:
        return mu_012 + mu_210 - 3 * mu_030
    elif index == 12:
        return mu_102 + mu_120 - 3 * mu_300
    elif index == 13:
        return mu_021 + mu_003 + 3 * mu_201
    elif index == 14:
        return mu_102 + mu_300 + 3 * mu_120
    elif index == 15:
        return mu_210 + mu_030 + 3 * mu_012
    elif index == 16:
        return mu_012 + mu_030 + 3 * mu_210
    elif index == 17:
        return mu_201 + mu_003 + 3 * mu_021
    elif index == 18:
        return mu_120 + mu_300 + 3 * mu_102
    else:
        raise IndexError


def gamma(index, coords, centroid):
    mu_022 = mu(0, 2, 2, coords, centroid)
    mu_202 = mu(2, 0, 2, coords, centroid)
    mu_220 = mu(2, 2, 0, coords, centroid)

    mu_400 = mu(4, 0, 0, coords, centroid)
    mu_040 = mu(0, 4, 0, coords, centroid)
    mu_004 = mu(0, 0, 4, coords, centroid)

    mu_112 = mu(1, 1, 2, coords, centroid)
    mu_121 = mu(1, 2, 1, coords, centroid)
    mu_211 = mu(2, 1, 1, coords, centroid)

    mu_130 = mu(1, 3, 0, coords, centroid)
    mu_103 = mu(1, 0, 3, coords, centroid)
    mu_013 = mu(0, 1, 3, coords, centroid)

    mu_310 = mu(3, 1, 0, coords, centroid)
    mu_301 = mu(3, 0, 1, coords, centroid)
    mu_031 = mu(0, 3, 1, coords, centroid)

    if index == 1:
        return mu_022 - mu_400
    elif index == 2:
        return mu_202 - mu_040
    elif index == 3:
        return mu_220 - mu_004
    elif index == 4:
        return mu_112 + mu_130 + mu_310
    elif index == 5:
        return mu_121 + mu_103 + mu_301
    elif index == 6:
        return mu_211 + mu_013 + mu_031
    elif index == 7:
        return mu_022 - mu_220 + mu_004 - mu_400
    elif index == 8:
        return mu_202 - mu_022 + mu_400 - mu_040
    elif index == 9:
        return mu_220 - mu_202 + mu_040 - mu_004
    else:
        raise IndexError

# def ci(coords, centroid):
    # part1 = mu_110 * (mu_022 * (3 * gamma(2, coords, centroid) - 2 * gamma(3, coords, centroid)))
