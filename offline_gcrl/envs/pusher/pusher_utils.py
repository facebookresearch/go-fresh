import numpy as np


def oracle_distance(x1, x2):
    return np.linalg.norm(x1[2:] - x2[2:])
