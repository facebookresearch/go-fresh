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
