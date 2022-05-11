import numpy as np


def get_state_from_lexa(env, state):
    new_state = np.pad(state, (0, 9), mode="constant")
    env.unwrapped._env.physics.set_state(new_state)
    env.unwrapped._env.physics.forward()
    return env.get_state()


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
