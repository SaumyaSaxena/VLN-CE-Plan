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

def plot_env_obs(agent_state, world_time, actions, instruction_id, episode_id, gt_data, data_type, role):
    actions = np.array(actions)[:,1]
    agent_state = np.array(agent_state)
    pose_traces = load_pose_trace(instruction_id, data_type=data_type, role=role)
    
    gt_locations = np.array(gt_data[episode_id]['locations'])
    gt_actions = gt_data[episode_id]['actions']

    fig, axs = plt.subplots(5, 3, figsize=(7, 10), constrained_layout=True)
    # fig.suptitle('Pose traces', fontsize=12)

    axs[0,0].plot(pose_traces['time'], pose_traces['extrinsic_matrix'][:,0,3])
    axs[0,0].set_xlabel('t'); axs[0,0].set_ylabel('pose_traces x')
    axs[0,1].plot(gt_locations[:,0])
    axs[0,1].set_xlabel('steps'); axs[0,1].set_ylabel('GT locations x')
    axs[0,2].plot(agent_state[:,0])
    axs[0,2].set_xlabel('steps'); axs[0,2].set_ylabel('Observations x')

    axs[1,0].plot(pose_traces['time'], pose_traces['extrinsic_matrix'][:,1,3])
    axs[1,0].set_xlabel('t'); axs[1,0].set_ylabel('pose_traces y')
    axs[1,1].plot(gt_locations[:,1])
    axs[1,1].set_xlabel('steps'); axs[1,1].set_ylabel('GT locations y')
    axs[1,2].plot(agent_state[:,1])
    axs[1,2].set_xlabel('steps'); axs[1,2].set_ylabel('Observations y')

    axs[2,0].plot(pose_traces['time'], pose_traces['extrinsic_matrix'][:,2,3])
    axs[2,0].set_xlabel('t'); axs[2,0].set_ylabel('pose_traces z')
    axs[2,1].plot(gt_locations[:,2])
    axs[2,1].set_xlabel('steps'); axs[2,1].set_ylabel('GT locations z')
    axs[2,2].plot(agent_state[:,2])
    axs[2,2].set_xlabel('steps'); axs[2,2].set_ylabel('Observations z')

    axs[3,0].plot(pose_traces['extrinsic_matrix'][:,0,3], pose_traces['extrinsic_matrix'][:,2,3])
    axs[3,0].set_xlabel('pose_traces x'); axs[3,0].set_ylabel('pose_traces z')
    axs[3,1].plot(gt_locations[:,0], gt_locations[:,2])
    axs[3,1].set_xlabel('GT locations x'); axs[3,1].set_ylabel('GT locations z')
    axs[3,2].plot(agent_state[:,0], agent_state[:,2])
    axs[3,2].set_xlabel('Observations x'); axs[3,2].set_ylabel('Observations z')

    axs[4,0].scatter(range(actions.shape[0]), actions, s=1)
    axs[4,0].set_xlabel('Steps'); axs[4,0].set_ylabel('actions applied')
    axs[4,0].scatter(range(len(gt_actions)), gt_actions, s=1)
    axs[4,0].set_xlabel('steps'); axs[4,0].set_ylabel('actions gt_data')
    axs[4,1].plot(pose_traces['time'])
    axs[4,1].set_xlabel('steps'); axs[4,1].set_ylabel('Data time')
    axs[4,2].plot(world_time)
    axs[4,2].set_xlabel('steps'); axs[4,2].set_ylabel('Simulation world time')
    plt.savefig(f"/home/sax1rng/Projects/VLN-CE-Plan/tests/media/rollout_envs/obs_traj_ep_id_{episode_id}.jpg")
    

if __name__== "__main__":
    save_video = True
    subtasks = True
    save_dataset = False

    data_type = 'train'
    role = 'guide'

    if save_dataset:
        gt_data_new = {}
    os.chdir(f'{cwd}/..') # Need to do this cos the config mergers use current working dir

    gt_data = load_gt_data(data_type=data_type, role=role, subtasks=subtasks)

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

    if save_video:
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
        video_dir = f'{cwd}/media/subtasks_rollout_merged' if subtasks else f'{cwd}/media'
    config.freeze()
 
    envs = construct_envs(
        config,
        get_env_class(config.ENV_NAME),
        auto_reset_done=False,
    )

    # Initialize envs
    env_step = [0 for _ in range(envs.num_envs)]
    position = [[] for _ in range(envs.num_envs)]
    rotation = [[] for _ in range(envs.num_envs)]
    actions = [[] for _ in range(envs.num_envs)]
    world_time = [[] for _ in range(envs.num_envs)]
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
        for i in range(envs.num_envs):

            if save_video:
                frame = observations_to_image(observations[i], infos[i])
                _text = text_to_append(current_episodes[i].instruction)
                frame = append_text_to_image(frame, _text)
                rgb_frames[i].append(frame)
            
            position[i].append(infos[i]['agent_position'])
            rotation[i].append([infos[i]['agent_rotation'].x, infos[i]['agent_rotation'].y, infos[i]['agent_rotation'].z, infos[i]['agent_rotation'].w])
            world_time[i].append(infos[i]['world_time'])
            
            if dones[i]:

                # plot_env_obs(
                #     position[i], 
                #     world_time[i],
                #     trajectories[current_episodes[i].episode_id],
                #     current_episodes[i].instruction.instruction_id,
                #     current_episodes[i].episode_id,
                #     gt_data,
                #     data_type,
                #     role
                #     )

                if save_video:
                    print("Saving episode id", current_episodes[i].episode_id)
                    generate_video(
                        video_option=['disk'],
                        video_dir=video_dir,
                        images=rgb_frames[i],
                        episode_id=current_episodes[i].episode_id,
                        checkpoint_idx=0,
                        metrics={"spl": infos[i]["spl_rxr"]},
                        tb_writer=None,
                    )

                if save_dataset:
                    gt_data_new[current_episodes[i].episode_id] = {'actions': [], 'locations': [], 'forward_steps': 0}
                    gt_data_new[current_episodes[i].episode_id]['actions'] = actions[i]
                    gt_data_new[current_episodes[i].episode_id]['locations'] = position[i]
                    gt_data_new[current_episodes[i].episode_id]['rotations'] = rotation[i]
                    gt_data_new[current_episodes[i].episode_id]['forward_steps'] = int(np.sum(np.array(actions[i])==1))
                
                stats_episodes[current_episodes[i].episode_id] = infos[i]
                if len(stats_episodes) %10 == 0:
                    print(f'Done:{len(stats_episodes)}/{total_num_episodes}')
                # pbar.update(len(stats_episodes))
                env_step[i] = 0
                position[i], rotation[i], world_time[i], rgb_frames[i], actions[i] = [], [], [], [], []

                observations[i] = envs.reset_at(i)[0]

                current_episodes = envs.current_episodes()

                infos[i] = envs.call_at(i, "get_info", {"observations": {}})
                position[i].append(infos[i]['agent_position'])
                rotation[i].append([infos[i]['agent_rotation'].x, infos[i]['agent_rotation'].y, infos[i]['agent_rotation'].z, infos[i]['agent_rotation'].w])
                world_time[i].append(infos[i]['world_time'])
                # at[i] = trajectories[current_episodes[i].episode_id][env_step[i]][1]
                at[i] = gt_data[current_episodes[i].episode_id]['actions'][env_step[i]]
                actions[i].append(at[i])
                if save_video:
                    frame = observations_to_image(observations[i], infos[i])
                    _text = text_to_append(current_episodes[i].instruction)
                    frame = append_text_to_image(frame, _text)
                    rgb_frames[i].append(frame)
            else:
                # at[i] = trajectories[current_episodes[i].episode_id][env_step[i]][1]
                at[i] = gt_data[current_episodes[i].episode_id]['actions'][env_step[i]]
                actions[i].append(at[i])
            
            env_step[i] += 1

        outputs = envs.step(at)
        observations, _, dones, infos = [list(x) for x in zip(*outputs)]
        
        observations = extract_instruction_tokens(
            observations,
            config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
        )
    
    if save_dataset:
        print(f'Saving new GT {data_type} file')
        output_file = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_rollout_gt3.json.gz'
        with gzip.open(output_file, "wt") as f:
            f.write(json.dumps(gt_data_new))
            print("Output file saved at ", output_file)