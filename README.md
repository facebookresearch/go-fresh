# Go-Fresh: Learning Goal-Conditioned Policies Offline with Self-Supervised Reward Shaping

This is the original implementation of the paper

**Learning Goal-Conditioned Policies Offline with Self-Supervised Reward Shaping** [[Project Page]](https://linamezghani.github.io/go-fresh) [[Paper]](https://openreview.net/forum?id=8tmKW-NG2bH)

by [Lina Mezghani](https://linamezghani.github.io/), [Sainbayar Sukhbaatar](https://scholar.google.com/citations?user=ri1sE34AAAAJ&hl=en), [Piotr Bojanowski](https://scholar.google.fr/citations?user=lJ_oh2EAAAAJ&hl=en), [Alessandro Lazaric](https://scholar.google.com/citations?user=6JZ3R6wAAAAJ&hl=en), [Karteek Alahari](https://lear.inrialpes.fr/people/alahari/)

## Prerequisites

### 1. Install [MuJoCo](https://mujoco.org/)

* Download MuJoCo binaries v2.2.0 [here](https://github.com/deepmind/mujoco/releases)
* Unzip the downloaded archive into `~/.mujoco/`
* Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`

### 2. Create conda environment

```
conda env create -f conda_env.yml
conda activate go-fresh
```

### 2. Install [mujoco-maze](https://github.com/kngwyu/mujoco-maze) from source

```
git clone https://github.com/kngwyu/mujoco-maze.git
cd mujoco-maze
pip install --no-deps -e .
```

## Generate Data

## Run the code

### Baselines

To run baselines mentioned in the paper, HER, HER + random uniform action and Actionable Models, run the following command:

```
python -m offline_gcrl.main +exp={env}_baseline replay_buffer.algo={algo}
```

where `env` can be chosen in `[maze, walker, pusher]`, and `algo` in `[HER, HERu, AM]`.
