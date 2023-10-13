import sys, os
sys.path.append('/home/sax1rng/Projects/VLN-CE-Plan')
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
# from scripts.llama2_convert_high_level_instr import load_data, load_pose_trace, load_gt_data, get_llama2_model

def load_data(data_type='train', role='guide'):
    data_location = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}.json.gz'
    data = {}
    with gzip.open(data_location,"rt",) as f:
        data.update(json.load(f))
    return data

def load_gt_data(data_type='train', role='guide'):
    gt_data_file = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_gt.json.gz'
    gt_data = {}
    with gzip.open(gt_data_file,"rt",) as f:
        gt_data.update(json.load(f))
    return gt_data

def load_pose_trace(instruction_id, data_type='train', role='guide'):
    pose_traces_path = f'/fs/scratch/rng_cr_aas3_gpu_user_c_lf/sp02_025_rrt/datasets/rxr-data/pose_traces/rxr_{data_type}/'
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
    os.chdir('/home/sax1rng/Projects/VLN-CE-Plan') # Need to do this cos the config mergers use current working dir

    # data = load_data(data_type=data_type, role=role)
    # gt_data = load_gt_data(data_type=data_type, role=role)

    # gt_data_file = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_subtasks_rollouts_small_gt.json.gz'
    # gt_data_rollouts = {}
    # with gzip.open(gt_data_file,"rt",) as f:
    #     gt_data_rollouts.update(json.load(f))

    # Load config
    config_path = 'vlnce_baselines/config/rxr_baselines/rxr_cma_en_subtasks.yaml'if subtasks else 'vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml'
    config = get_config(config_path, [])

    # From training script
    split = config.TASK_CONFIG.DATASET.SPLIT
    config.defrost()
    config.TASK_CONFIG.TASK.NDTW.SPLIT = split
    config.TASK_CONFIG.TASK.SDTW.SPLIT = split
    config.IL.RECOLLECT_TRAINER.gt_path = (config.TASK_CONFIG.TASK.NDTW.GT_PATH)
    config.use_pbar = not is_slurm_batch_job()
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (-1)
    config.TASK_CONFIG.MEASUREMENTS = []
    if save_video:
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
        video_dir = '/home/sax1rng/Projects/VLN-CE-Plan/tests/media/subtasks_rollout' if subtasks else '/home/sax1rng/Projects/VLN-CE-Plan/tests/media'
    config.freeze()

    with gzip.open(config.IL.RECOLLECT_TRAINER.trajectories_file, "rt") as f:
        trajectories = json.load(f)
    
    episodes_allowed = list(trajectories.keys())
    # import ipdb; ipdb.set_trace()
    envs = construct_envs(
        config,
        get_env_class(config.ENV_NAME),
        episodes_allowed=episodes_allowed,
        auto_reset_done=False,
    )
    # import ipdb; ipdb.set_trace()
    print(f'Number of episodes allowed: {len(episodes_allowed)}/{len(list(trajectories.keys()))}' )
    # envs = construct_envs_auto_reset_false(
    #     config, get_env_class(config.ENV_NAME)
    # ) # requires manually resetting the env using reset_at(i) method

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
    
    # Rollout
    # start = time.time()
    num_episodes_done = 0
    done_all_episodes = False
    # pbar = tqdm(desc='episodes', total=len(episodes_allowed))
    while not done_all_episodes:
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
            # if dones[i] or (env_step[i] > len(trajectories[current_episodes[i].episode_id])-1):
                # print('DONE: ', current_episodes[i].episode_id)

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

                env_step[i] = 0
                position[i], rotation[i], world_time[i], rgb_frames[i], actions[i] = [], [], [], [], []

                # end = time.time()
                # # print(f"time_elapsed for episode id {current_episodes[i].episode_id} is {end - start}")
                # print(f"Total time is {(end - start)*len(list(trajectories.keys()))/60/60}")
                # start = time.time()
                # import ipdb; ipdb.set_trace()
                observations[i] = envs.reset_at(i)[0]
                
                num_episodes_done += 1
                # pbar.update(num_episodes_done)
                if num_episodes_done % 10 == 0:
                    print('num_episodes_done:', num_episodes_done)
                if num_episodes_done > (len(episodes_allowed)-1):
                    print('KILLL MEEEE:', num_episodes_done)
                    done_all_episodes = True
                current_episodes = envs.current_episodes()
                # print('after reset: ', current_episodes[i].episode_id)
                infos[i] = envs.call_at(i, "get_info", {"observations": {}})
                position[i].append(infos[i]['agent_position'])
                rotation[i].append([infos[i]['agent_rotation'].x, infos[i]['agent_rotation'].y, infos[i]['agent_rotation'].z, infos[i]['agent_rotation'].w])
                world_time[i].append(infos[i]['world_time'])
                at[i] = trajectories[current_episodes[i].episode_id][env_step[i]][1]
                actions[i].append(at[i])
                if save_video:
                    frame = observations_to_image(observations[i], infos[i])
                    _text = text_to_append(current_episodes[i].instruction)
                    frame = append_text_to_image(frame, _text)
                    rgb_frames[i].append(frame)
            else:
                at[i] = trajectories[current_episodes[i].episode_id][env_step[i]][1]
                actions[i].append(at[i])
            
            env_step[i] += 1

        outputs = envs.step(at)
        observations, _, dones, infos = [list(x) for x in zip(*outputs)]
        
        observations = extract_instruction_tokens(
            observations,
            config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
        )
        if done_all_episodes:
            break
    
    print(f'Saving new GT {data_type} file')
    if save_dataset:
        output_file = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_rollout_gt_small.json.gz'
        with gzip.open(output_file, "wt") as f:
            f.write(json.dumps(gt_data_new))
            print("Output file saved at ", output_file)