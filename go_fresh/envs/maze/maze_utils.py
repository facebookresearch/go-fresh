# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

room_centers = {
    1: np.array([5, -10]),
    2: np.array([5, -20]),
    3: np.array([20, -20]),
    4: np.array([20, -10]),
}


def get_room_xy(x, y):
    if x < 12.5:
        if y > -12.5:
            return 1
        return 2
    if y > -12.5:
        return 4
    return 3


def oracle_distance(x1, x2):
    x1_room = get_room_xy(*x1[:2])
    x2_room = get_room_xy(*x2[:2])
    if abs(x1_room - x2_room) < 2:
        return np.linalg.norm(x1[:2] - x2[:2])
    fst_room = x1_room if x1_room < x2_room else x2_room
    fst_x, snd_x = (x1, x2) if x1_room < x2_room else (x2, x1)
    fst_center = room_centers[fst_room]
    fst_next_center = room_centers[fst_room + 1]
    return (
        oracle_distance(fst_x, fst_center)
        + oracle_distance(fst_center, fst_next_center)
        + oracle_distance(fst_next_center, snd_x)
    )
