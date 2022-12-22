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

```shell
conda env create -f conda_env.yml
conda activate go-fresh
```

### 3. Install [mujoco-maze](https://github.com/kngwyu/mujoco-maze) from source

```shell
git clone https://github.com/kngwyu/mujoco-maze.git
cd mujoco-maze
pip install --no-deps -e .
```

## Generate Data

The data used to train our model and baselines can be generated as follows:

### Maze

```shell
python -m offline_gcrl.generate_data --env maze --ep-len 1000
```

### Pusher

```shell
python -m offline_gcrl.generate_data --env pusher --ep-len 200
```

### Walker

Execute the following steps (from [here](https://github.com/denisyarats/exorl#datasets)) to download the dataset of exploration trajectories collected on the `walker` environment with the `proto` algorithm.

```shell
git clone https://github.com/denisyarats/exorl.git
cd exorl/
./download.sh walker proto
cd ..
mv ./datasets/walker/proto/buffer data/walker
```

## Run the code

### Baselines

To run baselines mentioned in the paper, HER, HER + random uniform action and Actionable Models, run the following command:

```shell
python -m offline_gcrl.main +exp=<ENV>_baseline replay_buffer.algo=<ALGO>
```

where `ENV` can be chosen in `maze`, `walker`, `pusher`, and `ALGO` in `HER`, `HERu`, `AM`.

### Our Method

To reproduce our method's results, run

```shell
python -m offline_gcrl.main +exp=<ENV>_ours
```

where `ENV` can be chosen in `maze`, `walker`, `pusher`.

### Details

* To visualize training info with [Weights&Biases](https://wandb.ai/), simply set parameter `wandb.enable=True`.

* The seed can be chosen by setting the `main.seed` parameter. All experiments presented in the paper were ran with the following random seeds: [123, 234, 345].

## Citation

If you use this repo in your research, please consider citing the paper as follows:
```bibtex
@inproceedings{mezghani2022learning,
  title={Learning Goal-Conditioned Policies Offline with Self-Supervised Reward Shaping},
  author={Mezghani, Lina and Sukhbaatar, Sainbayar and Bojanowski, Piotr and Lazaric, Alessandro and Alahari, Karteek},
  booktitle={CoRL-Conference on Robot Learning},
  year={2022}
}
```

## License

`go-fresh` is [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode) licensed, as found in the `LICENSE` file.
