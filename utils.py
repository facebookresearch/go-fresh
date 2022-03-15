import torch
import numpy as np

TORCH_DTYPE = {'float32': torch.float32, 'uint8': torch.uint8}

def get_space_info(obs_cfg, action_dim):
    space_info = {'obs_type': obs_cfg.type}
    space_info['action_dim'] = action_dim
    space_info['key'] = {'state': 'state'}
    space_info['type'] = {'state': 'float32'}
    if obs_cfg.type == 'vec':
        space_info['key']['obs'] = 'state'
        space_info['key']['goal'] = 'goal'
        space_info['type']['obs'] = 'float32'
        space_info['shape'] = {'state': (obs_cfg.vec_size,), 'obs':
                (obs_cfg.vec_size,)}
    elif obs_cfg.type == 'rgb':
        space_info['key']['obs'] = 'image'
        space_info['key']['goal'] = 'image_goal'
        space_info['type']['obs'] = 'uint8'
        space_info['shape'] = {'state': (obs_cfg.vec_size,), 'obs': (3,
            obs_cfg.rgb_size, obs_cfg.rgb_size)}
    else:
        raise ValueError(f"invalid obs_type: {obs_type}")
    return space_info

def oracle_reward(state, goal):
    return np.linalg.norm(state[:2] - goal[:2])
