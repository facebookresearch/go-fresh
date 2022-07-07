import numpy as np

import envs
import dmc2gym_utils


def oracle_distance(x1, x2):
    assert x1.shape[0] in [2, 4] and x2.shape[0] in [2, 4]
    x_dist = abs(x1[0] - x2[0])
    angle_dist = (
        np.abs(np.cos(x1[1]) - np.cos(x2[1])) + np.abs(np.sin(x1[1]) - np.sin(x2[1]))
    )
    return x_dist + angle_dist


def save_goals(env_cfg, space_info):
    env = envs.make_env(env_cfg, space_info)

    goals = {'state': [], 'vec_obs': [], 'rgb_obs': []}

    x_list = [x for x in np.arange(-1.6, 1.8, step=0.2)]

    for x in x_list:
        theta = 0
        state = np.array([x, theta])
        obs = dmc2gym_utils.get_state_from_qpos(env, state)
        rgb = env.get_image_from_obs(obs)
        goals['state'].append(state)
        goals['vec_obs'].append(obs)
        goals['rgb_obs'].append(rgb)

        theta = np.pi
        state = np.array([x, theta])
        obs = dmc2gym_utils.get_state_from_qpos(env, state)
        rgb = env.get_image_from_obs(obs)
        goals['state'].append(state)
        goals['vec_obs'].append(obs)
        goals['rgb_obs'].append(rgb)

    for k, v in goals.items():
        goals[k] = np.array(v)
        print(k, goals[k].shape)

    env.save_topline_goals(goals)
