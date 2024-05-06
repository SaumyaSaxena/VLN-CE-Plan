import sys, os
cwd = os.getcwd()
sys.path.append(f'{cwd}/..')
import gzip,json
import numpy as np
import time
from tqdm import tqdm
from PIL import Image

from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
from habitat_baselines.common.environments import get_env_class
from vlnce_baselines.config.default import get_config
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import extract_instruction_tokens
from habitat_extensions.utils import observations_to_image

def load_data(data_type='train', role='guide'):
    data_location = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}.json.gz'
    data = {}
    with gzip.open(data_location,"rt",) as f:
        data.update(json.load(f))
    return data

def load_gt_data(data_type='train', role='guide', subtasks=False):
    if subtasks:
        gt_data_file = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_subtasks_rollouts_merged_gt.json.gz'
    else:
        gt_data_file = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_gt.json.gz'
    gt_data = {}
    with gzip.open(gt_data_file,"rt",) as f:
        gt_data.update(json.load(f))
    return gt_data


def save_trajectory(save_dir, images):
    for i, im in enumerate(images):
        tmp_img = Image.fromarray(im.astype(np.uint8))
        tmp_img.save(f'{save_dir}/img_t{i}.png')

if __name__== "__main__":
    save_video = True
    subtasks = False
    save_dataset = False

    data_type = 'train'
    role = 'guide'

    if save_dataset:
        gt_data_new = {}
    os.chdir(f'{cwd}/..') # Need to do this cos the config mergers use current working dir

    gt_data = load_gt_data(data_type=data_type, role=role, subtasks=subtasks)
    data = load_data(data_type=data_type, role=role)

    dataset_size = len(data['episodes'])

    all_ep_idxs = [i for i in range(dataset_size) if ('en' in data['episodes'][i]['instruction']['language'])]
    all_ep_ids = [data['episodes'][i]['episode_id'] for i in range(dataset_size) if ('en' in data['episodes'][i]['instruction']['language'])]
    
    # Load config
    config_path = 'vlnce_baselines/config/rxr_baselines/rxr_cma_en_subtasks.yaml'if subtasks else 'vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml'
    config = get_config(config_path, [])

    split = data_type
    config.defrost()
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.DATASET.ROLES = ["guide"]
    config.TASK_CONFIG.TASK.NDTW.SPLIT = split
    config.TASK_CONFIG.TASK.SDTW.SPLIT = split
    config.IL.RECOLLECT_TRAINER.gt_path = (config.TASK_CONFIG.TASK.NDTW.GT_PATH)
    config.IL.RECOLLECT_TRAINER.gt_file = (config.TASK_CONFIG.TASK.NDTW.GT_PATH)
    config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
    config.use_pbar = not is_slurm_batch_job()
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (-1)
    config.TASK_CONFIG.MEASUREMENTS = []

    if save_video:
        video_dir = f"/home/saumyas/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_images"
        os.makedirs(video_dir, exist_ok=True)
    config.freeze()

    envs = construct_envs(
        config,
        get_env_class(config.ENV_NAME),
        auto_reset_done=False,
        episodes_allowed=all_ep_ids,
    )

    # Initialize envs
    env_step = [0 for _ in range(envs.num_envs)]
    rgb_frames = [[] for _ in range(envs.num_envs)]
    dones = [False for _ in range(envs.num_envs)]
    at = [0 for _ in range(envs.num_envs)]
    observations = envs.reset()
    infos = [envs.call_at(i, "get_info", {"observations": {}}) for i in range(envs.num_envs)]

    observations = extract_instruction_tokens(
        observations,
        config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
    )
    current_episodes = envs.current_episodes()
    stats_episodes = {}
    total_num_episodes = sum(envs.number_of_episodes)
    print(f"TOTAL EPISODES: {total_num_episodes}")
    
    while len(stats_episodes) < total_num_episodes:
        current_episodes = envs.current_episodes()
        for i in range(envs.num_envs):
            if save_video:
                frame = observations_to_image(observations[i], infos[i], append_depth=False, append_map=False)
                rgb_frames[i].append(frame)
            
            if dones[i]:
                if save_video:
                    save_dir = f"{video_dir}/{current_episodes[i].episode_id}"
                    # print(f"SAVING EPISODE ID: {save_dir}")
                    if os.path.exists(save_dir):
                        continue
                    else:
                        # print("Saving episode id", current_episodes[i].episode_id)
                        os.makedirs(save_dir, exist_ok=False)
                        save_trajectory(save_dir, rgb_frames[i])


                stats_episodes[current_episodes[i].episode_id] = infos[i]
                if len(stats_episodes) %20 == 0:
                    print(f'Done:{len(stats_episodes)}/{total_num_episodes}')
                env_step[i] = 0
                rgb_frames[i] = []

                observations[i] = envs.reset_at(i)[0]

                current_episodes = envs.current_episodes()

                infos[i] = envs.call_at(i, "get_info", {"observations": {}})
                at[i] = gt_data[current_episodes[i].episode_id]['actions'][env_step[i]]
                if save_video:
                    frame = observations_to_image(observations[i], infos[i], append_depth=False, append_map=False)
                    rgb_frames[i].append(frame)
            else:
                # at[i] = trajectories[current_episodes[i].episode_id][env_step[i]][1]
                at[i] = gt_data[current_episodes[i].episode_id]['actions'][env_step[i]]
            
            env_step[i] += 1

        outputs = envs.step(at)
        observations, _, dones, infos = [list(x) for x in zip(*outputs)]
        
        observations = extract_instruction_tokens(
            observations,
            config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
        )