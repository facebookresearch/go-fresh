from .base_wrapper import NormalizeActions
from .maze import maze_distance
from .dmc2gym import walker_distance, quadruped_distance, cartpole_distance
from .pusher import pusher_distance


def make_env(env_cfg, space_info, seed=0):
    if env_cfg.id.startswith("maze"):
        from .maze.maze_env import make_maze_env

        env = make_maze_env(env_cfg, space_info)
    elif env_cfg.id in ["walker", "quadruped", "cartpole"]:
        from .dmc2gym.dmc2gym_env import make_dmc2gym_env

        env = make_dmc2gym_env(env_cfg, space_info, seed)
    elif env_cfg.id == "pusher":
        from pusher.pusher_env import make_pusher_env

        env = make_pusher_env(env_cfg, space_info)
    else:
        raise ValueError(f"wrong env name {env_cfg.id}")
    env.seed(seed)
    env = NormalizeActions(env)
    return env


def oracle_reward(cfg, x1, x2):
    if cfg.env.id == 'maze_U4rooms':
        oracle_distance = maze_distance
    elif cfg.env.id == 'walker':
        oracle_distance = walker_distance
    elif cfg.env.id == 'quadruped':
        oracle_distance = quadruped_distance
    elif cfg.env.id == 'cartpole':
        oracle_distance = cartpole_distance
    elif cfg.env.id == 'pusher':
        oracle_distance = pusher_distance
    else:
        raise ValueError()

    if cfg.main.reward == "oracle_sparse":
        if oracle_distance(x1, x2) < cfg.env.success_thresh:
            return 0
        return -1
    elif cfg.main.reward == "oracle_dense":
        return -oracle_distance(x1, x2)
    else:
        raise ValueError()
