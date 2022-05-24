import dmc2gym

from gym.wrappers.time_limit import TimeLimit

from envs import quadruped_utils as utils

from envs.base_wrapper import BaseWrapper


class QuadrupedWrapper(BaseWrapper):
    def process_obs(self, obs):
        obs = super().process_obs(obs)
        obs["state"] = self.unwrapped._env.physics.data.qpos
        return obs

    def get_state(self):
        physics = self.unwrapped._env.physics
        obs = self.unwrapped._env._task.get_observation(physics)
        return dmc2gym.wrappers._flatten_obs(obs)

    def get_state_from_obs(self, obs):
        return obs

    def get_image_from_obs(self, obs):
        return self.get_image()

    def get_image(self):
        return self.render(
            mode="rgb_array", height=self.cfg.obs.rgb_size, width=self.cfg.obs.rgb_size
        ).transpose((2, 0, 1))

    def oracle_distance(self, x1, x2):
        goal_idx = None
        if hasattr(self, "goal_idx"):
            goal_idx = self.goal_idx
        return utils.oracle_distance(x1, x2, goal_idx=goal_idx)

    def process_info(self, info, metrics):
        super().process_info(info, metrics)
        del info["discount"]
        del info["internal_state"]
        info_goal = {}
        for g in range(self.num_goals):
            info_goal[f"count-goal{g}"] = int(g == self.goal_idx)
            for k, v in info.items():
                info_goal[f"{k}-goal{g}"] = v if g == self.goal_idx else 0
        info.update(info_goal)
        return info

    def set_info_keys(self):
        super().set_info_keys()
        to_add = []
        for g in range(self.num_goals):
            to_add.append(f"count-goal{g}")
            for k in self.info_keys:
                to_add.append(f"{k}-goal{g}")
        self.info_keys += to_add


def make_quadruped_env(env_cfg, space_info, seed):
    env = dmc2gym.make(
        "quadruped", "walk", frame_skip=env_cfg.action_repeat, seed=seed, camera_id=2
    )
    env = TimeLimit(env, max_episode_steps=env_cfg.max_episode_steps)
    env = QuadrupedWrapper(env, env_cfg, space_info)
    return env
