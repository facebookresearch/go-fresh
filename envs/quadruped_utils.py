import numpy as np

from scipy.spatial.transform import Rotation

from envs.walker_utils import shortest_angle


def get_state_from_lexa(env, state):
    new_state = np.pad(state, (0, 34), mode="constant")
    env.unwrapped._env.physics.set_state(new_state)
    env.unwrapped._env.physics.forward()
    return env.get_state()


def quat2euler(quat):
    rot = Rotation.from_quat(quat)
    return rot.as_euler('XYZ')


def oracle_distance(x1, x2, goal_idx=None):
    assert x1.shape[0] in [23, 57], x2.shape[0] in [23, 57]
    x1, x2 = x1[:23], x2[:23]

    def get_su(state, goal):
        dist = np.abs(state - goal)
        dist[..., [1, 2, 3]] = shortest_angle(dist[..., [1, 2, 3]])
        if goal_idx in [0, 1, 2, 5, 6, 7, 8, 11]:
            dist = dist[..., [0, 1, 2, 3, 4, 8, 12, 16]]
        return dist.max(-1)

    def rotate(s, times=1):
        # Invariance goes as follows: add 1.57 to azimuth,
        # circle legs 0,1,2,3 -> 1,2,3,0
        s = s.copy()
        for i in range(times):
            s[..., 1] = s[..., 1] + 1.57
            s[..., -16:] = np.roll(s[..., -16:], 12)
        return s

    def normalize(s):
        return np.concatenate((s[..., 2:3], quat2euler(s[..., 3:7]), s[..., 7:]), -1)

    x1_norm = normalize(x1)
    x2_norm = normalize(x2)
    return min(
        get_su(x1_norm, x2_norm),
        get_su(rotate(x1_norm, 1), x2_norm),
        get_su(rotate(x1_norm, 2), x2_norm),
        get_su(rotate(x1_norm, 3), x2_norm)
    )


def get_quadruped_pose(global_rot, global_pos=0.5, legs={}, legs_rot=[0, 0, 0, 0]):
    """
    :param angles: along height, along depth, along left-right
    :param height:
    :param legs:
    :return:
    """
    if not isinstance(global_pos, list):
        global_pos = [0, 0, global_pos]
    pose = np.zeros([23])
    pose[0:3] = global_pos
    pose[3:7] = (Rotation.from_euler('XYZ', global_rot).as_quat())

    pose[[7, 11, 15, 19]] = legs_rot
    for k, v in legs.items():
        for leg in v:
            if k == 'out':
                pose[[8 + leg * 4]] = 0.5  # pitch
                pose[[9 + leg * 4]] = -1.0  # knee
                pose[[10 + leg * 4]] = 0.5  # ankle
            if k == 'inward':
                pose[[8 + leg * 4]] = -0.35  # pitch
                pose[[9 + leg * 4]] = 0.9  # knee
                pose[[10 + leg * 4]] = -0.5  # ankle
            elif k == 'down':
                pose[[8 + leg * 4]] = 1.0  # pitch
                pose[[9 + leg * 4]] = -0.75  # knee
                pose[[10 + leg * 4]] = -0.3  # ankle
            elif k == 'out_up':
                pose[[8 + leg * 4]] = -0.2  # pitch
                pose[[9 + leg * 4]] = -0.8  # knee
                pose[[10 + leg * 4]] = 1.  # ankle
            elif k == 'up':
                pose[[8 + leg * 4]] = -0.35  # pitch
                pose[[9 + leg * 4]] = -0.2  # knee
                pose[[10 + leg * 4]] = 0.6  # ankle

    return pose


def get_lexa_goals():
    # pose[0,1] is x,y
    # pose[2] is height
    # pose[3:7] are vertical rotations in the form of a quaternion (i think?)
    # pose[7:11] are yaw pitch knee ankle for the front left leg
    # pose[11:15] same for the front right leg
    # pose[15:19] same for the back right leg
    # pose[19:23] same for the back left leg

    lie_legs_together = get_quadruped_pose(
        [0, 3.14, 0], 0.2, dict(out_up=[0, 1, 2, 3]), [-0.7, 0.7, -0.7, 0.7]
    )
    lie_rotated = get_quadruped_pose([0.8, 3.14, 0], 0.2, dict(out_up=[0, 1, 2, 3]))
    lie_two_legs_up = get_quadruped_pose(
        [0.8, 3.14, 0], 0.2, dict(out_up=[1, 3], down=[0, 2])
    )

    lie_side = get_quadruped_pose(
        [0., 0, -1.57], 0.3, dict(out=[0, 1, 2, 3]), [-0.7, 0.7, -0.7, 0.7]
    )
    lie_side_back = get_quadruped_pose(
        [0., 0, 1.57], 0.3, dict(out=[0, 1, 2, 3]), [-0.7, 0.7, -0.7, 0.7]
    )
    stand = get_quadruped_pose([1.57, 0, 0], 0.2, dict(up=[0, 1, 2, 3]))
    stand_rotated = get_quadruped_pose([0.8, 0, 0], 0.2, dict(up=[0, 1, 2, 3]))

    stand_leg_up = get_quadruped_pose(
        [1.57, 0, 0.0], 0.7, dict(down=[0, 2, 3], out_up=[1])
    )
    attack = get_quadruped_pose([1.57, 0., -0.4], 0.7, dict(out=[0, 1, 2, 3]))
    balance_front = get_quadruped_pose([1.57, 0.0, 1.57], 0.7, dict(up=[0, 1, 2, 3]))
    balance_back = get_quadruped_pose([1.57, 0.0, -1.57], 0.7, dict(up=[0, 1, 2, 3]))
    balance_diag = get_quadruped_pose(
        [1.57, 0, 0.0], 0.7, dict(down=[0, 2], out_up=[1, 3])
    )

    return np.stack([
        lie_legs_together,
        lie_rotated,
        lie_two_legs_up,
        lie_side,
        lie_side_back,
        stand,
        stand_rotated,
        stand_leg_up,
        attack,
        balance_front,
        balance_back,
        balance_diag
    ])
