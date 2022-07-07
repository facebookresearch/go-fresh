import numpy as np


def shortest_angle(angle):
    if not angle.shape:
        return shortest_angle(angle[None])[0]
    angle = angle % (2 * np.pi)
    angle[angle > np.pi] = 2 * np.pi - angle[angle > np.pi]
    return angle


def oracle_distance(x1, x2):
    assert x1.shape[0] in [9, 18], x2.shape[0] in [9, 18]
    x1, x2 = x1[:9], x2[:9]

    def get_su(_goal):
        dist = np.abs(x1 - _goal)
        dist = dist[..., [0, 2, 3, 4, 6, 7]]
        dist[..., 1] = shortest_angle(dist[..., 1])
        return dist.max(-1)

    return min(get_su(x2), get_su(x2[..., [0, 1, 2, 6, 7, 8, 3, 4, 5]]))


def get_lexa_goals():
    # pose[0] is height
    # pose[1] is x
    # pose[2] is global rotation
    # pose[3:6] - first leg hip, knee, ankle
    # pose[6:9] - second leg hip, knee, ankle
    # Note: seems like walker can't bend legs backwards

    lie_back = [-1.2, 0., -1.57, 0., 0., 0., 0, -0., 0.]
    lie_front = [-1.2, -0, 1.57, 0., 0, 0., 0., 0., 0.]
    legs_up = [-1.24, 0., -1.57, 1.57, 0., 0.0, 1.57, -0., 0.0]

    kneel = [-0.5, 0., 0., 0., -1.57, -0.8, 1.57, -1.57, 0.0]
    side_angle = [-0.3, 0., 0.9, 0., 0., -0.7, 1.87, -1.07, 0.0]
    stand_up = [-0.15, 0., 0.34, 0.74, -1.34, -0., 1.1, -0.66, -0.1]

    lean_back = [-0.27, 0., -0.45, 0.22, -1.5, 0.86, 0.6, -0.8, -0.4]
    boat = [-1.04, 0., -0.8, 1.6, 0., 0.0, 1.6, -0., 0.0]
    bridge = [-1.1, 0., -2.2, -0.3, -1.5, 0., -0.3, -0.8, -0.4]

    head_stand = [-1, 0., -3, 0.6, -1, -0.3, 0.9, -0.5, 0.3]
    one_feet = [-0.2, 0., 0, 0.7, -1.34, 0.5, 1.5, -0.6, 0.1]
    arabesque = [-0.34, 0., 1.57, 1.57, 0, 0., 0, -0., 0.]

    return np.stack([
        lie_back,
        lie_front,
        legs_up,
        kneel,
        side_angle,
        stand_up,
        lean_back,
        boat,
        bridge,
        one_feet,
        head_stand,
        arabesque
    ])
