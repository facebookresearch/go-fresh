from .base_wrapper import NormalizeActions
from .maze_utils import oracle_distance as maze_distance
from .walker_utils import oracle_distance as walker_distance
from .quadruped_utils import oracle_distance as quadruped_distance


def make_env(env_cfg, space_info, seed=0):
    if env_cfg.id.startswith("maze"):
        from .maze_env import make_maze_env

        env = make_maze_env(env_cfg, space_info)
    elif env_cfg.id in ["walker", "quadruped"]:
        from .dmc2gym_env import make_dmc2gym_env

        env = make_dmc2gym_env(env_cfg, space_info, seed)
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
