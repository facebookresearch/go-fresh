import torch

TORCH_DTYPE = {'float32': torch.float32, 'uint8': torch.uint8}

def get_space_info(obs_cfg, action_dim):
    space_info = {'obs_type': obs_cfg.type}
    space_info['action_dim'] = action_dim
    space_info['type'] = {'state': 'float32'}
    space_info['shape'] = {'state': (obs_cfg.state_size,)}
    if obs_cfg.type == 'vec':
        space_info['type']['obs'] = 'float32'
        space_info['shape']['obs'] = (obs_cfg.vec_size,)
    elif obs_cfg.type == 'rgb':
        space_info['type']['obs'] = 'uint8'
        space_info['shape']['obs'] = (3, obs_cfg.rgb_size, obs_cfg.rgb_size)
    else:
        raise ValueError(f"invalid obs_type: {obs_type}")
    return space_info
