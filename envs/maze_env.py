import torch
import numpy as np
import mujoco_maze
import matplotlib.pyplot as plt

from matplotlib import collections as mc
from typing import List
from mujoco_maze import maze_env, maze_task
from mujoco_maze.maze_env_utils import MazeCell
from gym.wrappers.time_limit import TimeLimit

import envs.maze_utils as utils

from envs.base_wrapper import BaseWrapper


class MazeWrapper(BaseWrapper):
    def __init__(self, env, cfg, space_info):
        super().__init__(env, cfg, space_info)
        if self.cfg.obs.type == "rgb":
            from mujoco_py import MjRenderContextOffscreen as MjRCO

            self.unwrapped._mj_offscreen_viewer = MjRCO(self.wrapped_env.sim)
            self.unwrapped._maybe_move_camera(self.unwrapped._mj_offscreen_viewer)
        xmin, xmax, ymin, ymax = self.unwrapped._xy_limits()
        self.xlim = (xmin, xmax)
        self.ylim = (ymin, ymax)

    def process_info(self, info, metrics):
        super().process_info(info, metrics)
        del info["position"]
        info_room = {}
        for r in range(1, 5):
            info_room[f"count-room{r}"] = int(r == self.room_goal)
            for k, v in info.items():
                info_room[f"{k}-room{r}"] = v if r == self.room_goal else 0
        info.update(info_room)
        return info

    def set_info_keys(self):
        super().set_info_keys()
        to_add = []
        for r in range(1, 5):
            to_add.append(f"count-room{r}")
            for k in self.info_keys:
                to_add.append(f"{k}-room{r}")
        self.info_keys += to_add

    def oracle_distance(self, x1, x2):
        return utils.oracle_distance(x1, x2)

    def get_image(self):
        return self.unwrapped._render_image().transpose((2, 0, 1))

    def get_image_from_obs(self, obs):
        return self.get_image()

    def get_state(self):
        return self.unwrapped._get_obs()

    def get_state_from_obs(self, obs):
        return obs

    def generate_topline_goals(self, n):
        return self.generate_points(n, random=True)

    def get_room_xy(self, x, y):
        return utils.get_room_xy(x, y)

    def get_room_goal(self):
        goal_idx = self.get_goal_idx()
        goal = self.get_goals()["state"][goal_idx]
        return self.get_room_xy(*goal[:2])

    def set_goal_idx(self, goal_idx):
        super().set_goal_idx(goal_idx)
        self.room_goal = self.get_room_goal()

    def get_xyori(self, x, y, ori=None):
        if ori is None:
            ori = self.generate_random_ori()
        self.wrapped_env.set_xy((x, y))
        self.wrapped_env.set_ori(ori)
        state = self.get_state()
        if self.cfg.obs.type == "vec":
            return state, None
        image = self.get_image()
        return state, image

    def is_valid(self, x, y):
        if self.cfg.id in ["maze_U4rooms", "maze_4rooms"]:
            if x >= 25.6 or x <= -1.6 or y <= -25.6 or y >= 1.6:
                return False
        scaling = self.unwrapped._maze_size_scaling
        x0 = self.unwrapped._init_torso_x
        y0 = self.unwrapped._init_torso_y
        structure = self.unwrapped._maze_structure
        epss = [0.4, -0.4]
        rows = [int((x + eps + x0) / scaling + 0.5) for eps in epss]
        cols = [int((y + eps + y0) / scaling + 0.5) for eps in epss]
        for row in rows:
            for col in cols:
                if structure[col][row].is_block():
                    return False
        return True

    def generate_random_ori(self):
        return np.random.uniform(-np.pi, np.pi)

    def set_random_pos(self):
        x, y = self.generate_random_point()
        ori = self.generate_random_ori()
        self.wrapped_env.set_xy((x, y))
        self.wrapped_env.set_ori(ori)

    def generate_random_point(self):
        valid = False
        while not valid:
            x = np.random.uniform(*self.xlim)
            y = np.random.uniform(*self.ylim)
            valid = self.is_valid(x, y)
        return x, y

    def generate_points(self, n, random):
        points = {"state": []}
        if self.cfg.obs.type == "rgb":
            points["image"] = []
        if random:
            for i in range(n):
                x, y = self.generate_random_point()
                state, image = self.get_xyori(x, y)
                points["state"].append(state)
                if self.cfg.obs.type == "rgb":
                    points["image"].append(image)
        else:
            for x in np.arange(*self.xlim, (self.xlim[1] - self.xlim[0]) / n):
                for y in np.arange(*self.ylim, (self.ylim[1] - self.ylim[0]) / n):
                    if self.is_valid(x, y):
                        state, image = self.get_xyori(x, y)
                        points["state"].append(state)
                        if self.cfg.obs.type == "rgb":
                            points["image"].append(image)
        return {k: np.array(points[k]) for k in points}

    def plot_goals(self, title=None):
        fig = plt.figure()
        title = "goals" if title is None else title
        goals = self.get_goals()
        plt.scatter(goals["state"][:, 0], goals["state"][:, 1])
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.title(title)
        plt.grid()
        return fig

    def plot_buffer(self, buffer_states):
        a = buffer_states.reshape((-1, buffer_states.shape[-1]))
        fig = plt.figure()
        plt.hist2d(a[:, 0], a[:, 1], bins=20, range=(self.xlim, self.ylim))
        return fig

    def plot_stats_goal_idx(self, stats):
        goals = self.get_goals()
        fig = plt.figure()
        sc = plt.scatter(goals["state"][:, 0], goals["state"][:, 1], c=stats)
        plt.colorbar(sc)
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.grid()
        return fig

    def plot_values(
        self, rnet_model, comp_obs, comp_state, obss, states, pos=False, sigmoid=False
    ):
        rnet_values = rnet_model.compute_values(comp_obs, obss)
        if sigmoid:
            rnet_values = torch.sigmoid(torch.from_numpy(rnet_values)).numpy()
        if pos:
            rnet_values = rnet_values > 0
        fig = plt.figure()
        sc = plt.scatter(states[:, 0], states[:, 1], c=rnet_values)
        fig.colorbar(sc)
        plt.scatter(comp_state[0], comp_state[1], c="red", marker="x", s=100)
        return fig

    def plot_embeddings(self, rnet_model, obss, states, show_colors=True, title=None):
        features_red = rnet_model.pca_embeddings(obss)
        colors = states[:, 0] + states[:, 1]
        title = "rnet_features" if title is None else title
        if show_colors:
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].scatter(states[:, 0], states[:, 1], c=colors)
            ax[0].set_title("colors")
            ax[1].scatter(features_red[:, 0], features_red[:, 1], c=colors)
            ax[1].set_title(title)
            ax[1].axis("off")
        else:
            fig = plt.figure()
            plt.scatter(features_red[:, 0], features_red[:, 1], c=colors)
            plt.title(title)
            plt.axis("off")
        return fig

    def plot_graph_dist(self, memory, show_path=True, start=None, end=None):
        fig = plt.figure()
        states = memory.states[: len(memory)]
        if end is None:
            end = np.random.randint(0, len(memory))
        sc = plt.scatter(states[:, 0], states[:, 1], c=-memory.dist[end])

        if show_path:
            if start is None:
                n_trials = 0
                while True:
                    n_trials += 1
                    start = np.random.randint(0, len(memory))
                    if n_trials > 100 or np.linalg.norm(states[start, :2]) < 2:
                        break

            path = memory.retrieve_path(start, end)
            for j in range(len(path) - 1):
                temp = states[[path[j], path[j + 1]]]
                plt.plot(temp[:, 0], temp[:, 1], color="red")
                plt.scatter(temp[:, 0], temp[:, 1], color="red")
        plt.colorbar(sc)
        plt.scatter(states[end, 0], states[end, 1], c="red", marker="x", s=100)
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.grid()
        return fig

    def plot_graph(self, memory):
        fig, ax = plt.subplots()
        states = memory.states[: len(memory)]
        lines = []
        for i in range(len(memory)):
            for j in range(i):
                if memory.adj_matrix[i, j] or memory.adj_matrix[j, i]:
                    lines.append([states[i, :2], states[j, :2]])
        lc = mc.LineCollection(lines, color="gray", alpha=0.5)
        ax.add_collection(lc)
        ax.scatter(states[:, 0], states[:, 1])
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.grid()
        return fig


class NoGoalSquareRoom(maze_task.NoRewardSquareRoom):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = []


class NoGoalBigSquareRoom(NoGoalSquareRoom):
    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B],
            [B, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, E, E, E, R, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, B, B, B, B, B, B, B, B],
        ]


class NoReward4Rooms(maze_task.GoalReward4Rooms):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)

    def reward(self, _obs: np.ndarray) -> float:
        return 0.0


class NoGoal4Rooms(NoReward4Rooms):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = []


class NoGoalU4Rooms(NoGoal4Rooms):
    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B],
            [B, E, E, E, B, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, E, E, E, B, E, E, E, B],
            [B, B, E, B, B, B, E, B, B],
            [B, E, E, E, B, E, E, E, B],
            [B, E, E, E, B, E, E, E, B],
            [B, R, E, E, B, E, E, E, B],
            [B, B, B, B, B, B, B, B, B],
        ]


class MyPointEnv(mujoco_maze.point.PointEnv):
    def set_ori(self, ori):
        qpos = self.sim.data.qpos.copy()
        qpos[2] = ori
        self.set_state(qpos, self.sim.data.qvel)


ZOOM_PARAMS = {
    "square_room": 0.12,
    "big_square_room": 0,
    "4rooms": 0.03,
    "U4rooms": 0.03,
}

ENV_CLS = {
    "square_room": NoGoalSquareRoom,
    "big_square_room": NoGoalBigSquareRoom,
    "4rooms": NoGoal4Rooms,
    "U4rooms": NoGoalU4Rooms,
}

IMG_SIZE = {
    "square_room": 64,
    "big_square_room": 100,
    "4rooms": 128,
    "U4rooms": 128,
}


def make_maze_env(env_cfg, space_info):
    env_name = env_cfg.id[5:]
    assert env_name in ENV_CLS, f"invalid env_id {env_name}"
    zoom = ZOOM_PARAMS[env_name]
    img_size = IMG_SIZE[env_name]
    env_cls = ENV_CLS[env_name]
    env = maze_env.MazeEnv(
        model_cls=MyPointEnv,
        maze_task=env_cls,
        image_shape=(img_size, img_size),
        camera_move_y=1,
        camera_zoom=zoom,
    )
    env = TimeLimit(env, max_episode_steps=env_cfg.max_episode_steps)
    env = MazeWrapper(env, env_cfg, space_info)
    return env
