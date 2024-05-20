import sys, os
cwd = os.getcwd()
sys.path.append(f'{cwd}/..')
import gzip,json
import numpy as np
import time
from tqdm import tqdm

from habitat.utils.visualizations.utils import append_text_to_image

from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
from habitat_baselines.common.environments import get_env_class
from vlnce_baselines.config.default import get_config
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_auto_reset_false
from vlnce_baselines.common.utils import extract_instruction_tokens
from habitat_extensions.utils import generate_video, observations_to_image, text_to_append

import matplotlib.pyplot as plt

def load_gt_data(data_type='train', role='guide', subtasks=False):
    if subtasks:
        gt_data_file = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_subtasks_rollouts_merged_gt.json.gz'
    else:
        gt_data_file = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_gt.json.gz'
    gt_data = {}
    with gzip.open(gt_data_file,"rt",) as f:
        gt_data.update(json.load(f))
    return gt_data

def load_pose_trace(instruction_id, data_type='train', role='guide'):
    pose_traces_path = f'{cwd}/../data/datasets/RxR_VLNCE_v0/pose_traces/rxr_{data_type}/'
    pose_traces_file_name = '{id:06}_{role}_pose_trace.npz'

    pose_traces_file_name = pose_traces_file_name.replace('{role}', role)

    pose_traces_file_name = pose_traces_path + pose_traces_file_name.format(id=int(instruction_id))
    pose_traces = np.load(pose_traces_file_name)
    return pose_traces

if __name__== "__main__":
    save_video = True
    subtasks = True

    data_type = 'train'
    role = 'guide'

    os.chdir(f'{cwd}/..') # Need to do this cos the config mergers use current working dir

    gt_data = load_gt_data(data_type=data_type, role=role, subtasks=subtasks)
    
    inp_file = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_mapping_original_to_subtask.json.gz'
    mapping_original_to_subtask = {}
    with gzip.open(inp_file,"rt",) as f:
        mapping_original_to_subtask.update(json.load(f))
    
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
    
    # config.SIMULATOR_GPU_IDS: [0] # Evals run on these GPUs
    # config.NUM_ENVIRONMENTS: 50 # Number of envs per GPU
    
    orig_ep_idx = 11288 # '11288', '11292', '11293', '11300', '11301', '11302', '11303', '11313', '11314', '11315
    all_ep_idxs = mapping_original_to_subtask.keys()
    episodes_allowed = mapping_original_to_subtask[str(orig_ep_idx)]
    
    if save_video:
        # config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
        video_dir = f'{cwd}/media/{data_type}/subtasks/{orig_ep_idx}' if subtasks else f'{cwd}/media/{data_type}/full_trajs'
        os.makedirs(video_dir, exist_ok=True)
    config.freeze()

    envs = construct_envs(
            config,
            get_env_class(config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=episodes_allowed
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
    # pbar = tqdm(desc='episodes', total=total_num_episodes)
    while len(stats_episodes) < total_num_episodes:
        current_episodes = envs.current_episodes()
        _ep_ids = [current_episodes[i].episode_id for i in range(envs.num_envs)]
        
        for i in range(envs.num_envs):
            if save_video:
                frame = observations_to_image(observations[i], infos[i], append_depth=False)
                _text = text_to_append(current_episodes[i].instruction)
                frame = append_text_to_image(frame, _text)
                rgb_frames[i].append(frame)
            
            if dones[i]:
                if save_video:
                    print("Saving episode id", current_episodes[i].episode_id)
                    if subtasks:
                        metrics={"spl": infos[i]["spl_rxr"],
                                 "orig_ep_idx": current_episodes[i].original_episode_idx}
                    else:
                        metrics={"spl": infos[i]["spl_rxr"]}
                    generate_video(
                        video_option=['disk'],
                        video_dir=video_dir,
                        images=rgb_frames[i],
                        episode_id=current_episodes[i].episode_id,
                        checkpoint_idx=0,
                        metrics=metrics,
                        tb_writer=None,
                    )
                
                stats_episodes[current_episodes[i].episode_id] = infos[i]

                # pbar.update(len(stats_episodes))
                env_step[i] = 0
                rgb_frames[i]= []

                observations[i] = envs.reset_at(i)[0]
                current_episodes = envs.current_episodes()
                infos[i] = envs.call_at(i, "get_info", {"observations": {}})
                at[i] = gt_data[current_episodes[i].episode_id]['actions'][env_step[i]]

                if save_video:
                    frame = observations_to_image(observations[i], infos[i], append_depth=False)
                    _text = text_to_append(current_episodes[i].instruction)
                    frame = append_text_to_image(frame, _text)
                    rgb_frames[i].append(frame)
            else:
                at[i] = gt_data[current_episodes[i].episode_id]['actions'][env_step[i]]
            
            env_step[i] += 1

        outputs = envs.step(at)
        observations, _, dones, infos = [list(x) for x in zip(*outputs)]
        
        observations = extract_instruction_tokens(
            observations,
            config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
        )
    
    print(f"All episodes rolled out for epsiode {orig_ep_idx}")