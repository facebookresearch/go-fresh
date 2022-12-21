import numpy as np


def oracle_distance(x1, x2):
    assert x1.shape[0] in [2, 4] and x2.shape[0] in [2, 4]
    x_dist = abs(x1[0] - x2[0])
    if x_dist > 0.25:
        return 1
    cos_dist = np.abs(np.cos(x1[1]) - np.cos(x2[1]))
    if cos_dist > 0.04:
        return 1
    sin_dist = np.abs(np.sin(x1[1]) - np.sin(x2[1]))
    if sin_dist > 0.3:
        return 1
    return 0
