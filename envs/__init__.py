from .base_wrapper import NormalizeActions

def make_env(env_cfg, seed=0, keep_per_goal=True):
    if env_cfg.id == 'pusher':
        from .pusher_env import make_pusher_env
        env = make_pusher_env(env_cfg)
    elif env_cfg.id.startswith("lexa"):
        from .lexa_env import make_lexa_env
        env = make_lexa_env(env_cfg, keep_per_goal=keep_per_goal)
    elif env_cfg.id.startswith("maze"):
        from .maze_env import make_maze_env
        env = make_maze_env(env_cfg)
    else:
        raise ValueError(f"wrong env name {env_cfg.id}")
    env.seed(seed)
    env = NormalizeActions(env)
    return env
