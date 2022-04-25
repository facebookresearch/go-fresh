from .base_wrapper import NormalizeActions

def make_env(env_cfg, space_info, seed=0, keep_per_goal=True):
    if env_cfg.id.startswith("maze"):
        from .maze_env import make_maze_env
        env = make_maze_env(env_cfg, space_info)
    elif env_cfg.id == 'walker':
        from .walker_env import make_walker_env
        env = make_walker_env(env_cfg, space_info, seed)
    else:
        raise ValueError(f"wrong env name {env_cfg.id}")
    env.seed(seed)
    env = NormalizeActions(env)
    return env
