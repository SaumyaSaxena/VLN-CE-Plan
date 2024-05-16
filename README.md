# Vision-and-Language Navigation in Continuous Environments (VLN-CE)

[Project Website](https://jacobkrantz.github.io/vlnce/) — [VLN-CE Challenge](https://eval.ai/web/challenges/challenge-page/719) — [RxR-Habitat Challenge](https://ai.google.com/research/rxr/habitat)

Official implementations:

- *Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments* ([paper](https://arxiv.org/abs/2004.02857))
- *Waypoint Models for Instruction-guided Navigation in Continuous Environments* ([paper](https://arxiv.org/abs/2110.02207), [README](/vlnce_baselines/config/r2r_waypoint/README.md))

Vision and Language Navigation in Continuous Environments (VLN-CE) is an instruction-guided navigation task with crowdsourced instructions, realistic environments, and unconstrained agent navigation. This repo is a launching point for interacting with the VLN-CE task and provides both baseline agents and training methods. Both the Room-to-Room (**R2R**) and the Room-Across-Room (**RxR**) datasets are supported. VLN-CE is implemented using the Habitat platform.

<p align="center">
  <img width="775" height="360" src="./data/res/VLN_comparison.gif" alt="VLN-CE comparison to VLN">
</p>

## Setup

### Create conda environment

This project is developed with Python 3.6. If you are using [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://anaconda.org/), you can create an environment:

```bash
conda create -n vlnce python=3.9.18
conda activate vlnce
conda install -n vlnce -c conda-forge cmake=3.16.3
# Check cmake version 
cmake --version
```

### Install torch
```
pip install torch torchvision torchaudio
```

### Install Habitat-sim

VLN-CE uses Habitat-Sim [built from source](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7#installation)
```bash
git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
python setup.py install --with-cuda --headless --cmake
```
If GL errors occur try installing: sudo apt-get install libegl1-mesa-dev libgles2-mesa

### Install Habitat-Lab

Install [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7):

```bash
git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
# installs both habitat and habitat_baselines
# Comment out torch and tensorflow from the requirements files
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all
```

### Install VLN_CE_Plan

```bash
git clone git@github.com:SaumyaSaxena/VLN-CE-Plan.git
cd VLN-CE-Plan
# Comment out torch and tensorflow from the requirements files
pip install -r requirements.txt
```

### Install Generative model APIs

```bash
pip install -q -U google-generativeai
pip install openai
```

## Data

Note: It will be useful to symlink the 'VLN-CE-Plan/data' folder to a location that has atleast 300GB of free storage.

### Scenes: Matterport3D

Matterport3D (MP3D) scene reconstructions are used. The official Matterport3D download script (`download_mp.py`) can be accessed by following the instructions on their [project webpage](https://niessner.github.io/Matterport/). This can take around 30min to download and needs 21GB of storage space.

The scene data can then be downloaded:

```bash
# requires running with python 2.7
python download_mp.py --task habitat -o data/scene_datasets/mp3d/

# Or without the download script
wget "http://kaldir.vc.in.tum.de/matterport/v1/tasks/mp3d_habitat.zip"
```

Extract such that it has the form `VLN-CE-Plan/data/scene_datasets/mp3d/{scene}/{scene}.glb`. There should be 90 scenes.


### Encoder Weights

Baseline models encode depth observations using a ResNet pre-trained on PointGoal navigation. Those weights can be downloaded from [here](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7/habitat_baselines/rl/ddppo) (672M). Extract the contents to `VLN-CE-Plan/data/ddppo-models/{model}.pth`.

### Episodes: Room-Across-Room (RxR)

Download: [RxR_VLNCE_v0.zip](https://storage.googleapis.com/rxr-habitat/RxR_VLNCE_v0.zip)

About the [Room-Across-Room dataset](https://ai.google.com/research/rxr/) (RxR):

- multilingual instructions (English, Hindi, Telugu)
- an order of magnitude larger than existing datasets
- varied paths to break a shortest-path-to-goal assumption

RxR was ported to continuous environments originally for the [RxR-Habitat Challenge](https://ai.google.com/research/rxr/habitat). The dataset has `train`, `val_seen`, `val_unseen`, and `test_challenge` splits with both Guide and Follower trajectories ported. The starter code expects files in this structure:

```graphql
data/datasets
├─ RxR_VLNCE_v0
|   ├─ train
|   |    ├─ train_guide.json.gz
|   |    ├─ train_guide_gt.json.gz
|   |    ├─ train_follower.json.gz
|   |    ├─ train_follower_gt.json.gz
|   ├─ val_seen
|   |    ├─ val_seen_guide.json.gz
|   |    ├─ val_seen_guide_gt.json.gz
|   |    ├─ val_seen_follower.json.gz
|   |    ├─ val_seen_follower_gt.json.gz
|   ├─ val_unseen
|   |    ├─ val_unseen_guide.json.gz
|   |    ├─ val_unseen_guide_gt.json.gz
|   |    ├─ val_unseen_follower.json.gz
|   |    ├─ val_unseen_follower_gt.json.gz
|   ├─ test_challenge
|   |    ├─ test_challenge_guide.json.gz
|   ├─ text_features
|   |    ├─ ...
```

### BERT text features

The baseline models for RxR-Habitat use precomputed BERT instruction features which can be downloaded from [here](https://github.com/google-research-datasets/RxR#downloading-bert-text-features) and saved to `data/datasets/RxR_VLNCE_v0/text_features/rxr_{split}/{instruction_id}_{language}_text_features.npz`. This requires 159GB of storage space. Will need to [install 'gsutil'](https://cloud.google.com/storage/docs/gsutil_install#linux) for downloading.

OR 

Generate and save the BERT features using the `scripts/save_bert_features_subtasks.py` script. For instructions on how to save BERT features for the datasets, refer to `scripts/README.md`.

## Running training/eval/inference

### Train/eval using full instructions

```bash
python run.py --exp-config vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml --run-type train/eval/inference
```

### Train/eval using only high-level instructions

```bash
python run.py --exp-config vlnce_baselines/config/rxr_baselines/rxr_cma_en_hl_instr.yaml --run-type train/eval
```

For this to work you should have the following files:

GT data file: `data/datasets/RxR_VLNCE_v0/{split}/{split}_{role}_gt.json.gz`

High level instruction file: `data/datasets/RxR_VLNCE_v0/{split}/{split}_{role}_high_level_instr.json.gz`

BERT features saved at: `data/datasets/RxR_VLNCE_v0/text_features_highlevel_instr/rxr_{split}/{id:06}_{lang}_text_features.npz`

### Train/eval subtasks
```bash
python run.py --exp-config vlnce_baselines/config/rxr_baselines/rxr_cma_en_subtasks.yaml --run-type train/eval
```

For this to work you should have the following files:

GT data file: `data/datasets/RxR_VLNCE_v0/{split}/{split}_{role}_subtasks_rollouts_merged_gt.json.gz`

High level instruction file: `data/datasets/RxR_VLNCE_v0/{split}/{split}_{role}_subtasks_rollouts_merged_hli.json.gz`

BERT features saved at: `data/datasets/RxR_VLNCE_v0/text_features_subtasks_merged/rxr_{split}/{id:06}_{lang}_text_features.npz`

### Evalution: Use learned subtask policies to follow high level instructions using context learning and planning
```bash
python run.py --exp-config vlnce_baselines/config/rxr_baselines/rxr_cma_en_subtasks_context.yaml --run-type eval
```
