# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def oracle_distance(x1, x2):
    return np.linalg.norm(x1[2:] - x2[2:])
