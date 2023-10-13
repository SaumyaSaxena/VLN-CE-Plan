import sys, os
sys.path.append('/home/sax1rng/Projects/VLN-CE-Plan')

from transformers import AutoTokenizer
import transformers
import torch, gzip, json
import numpy as np
import time
import matplotlib.pyplot as plt

from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
from habitat_baselines.common.environments import get_env_class
from vlnce_baselines.config.default import get_config
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_auto_reset_false
from vlnce_baselines.common.utils import extract_instruction_tokens
from habitat_extensions.utils import generate_video, observations_to_image, text_to_append

from habitat.core.utils import try_cv2_import
cv2 = try_cv2_import()

def get_llama2_model():
    model = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return tokenizer, pipeline

def plot_pose_trace_discretized(tstamp_posetrace_idx, pose_trace, episode_id):
    
    fig, axs = plt.subplots(2, 2, figsize=(5, 5), constrained_layout=True)
    fig.suptitle('Pose traces', fontsize=12)
    colors = []
    for i in range(len(tstamp_posetrace_idx)-1):
        color = np.random.uniform(low=0.0, high=1.0, size=(3,))
        colors.append(color)

        axs[0,0].plot(pose_trace['time'][tstamp_posetrace_idx[i]:tstamp_posetrace_idx[i+1]+1], 
            pose_trace['extrinsic_matrix'][tstamp_posetrace_idx[i]:tstamp_posetrace_idx[i+1]+1,0, 3],color=color)
        axs[0,0].set_xlabel('t'); axs[0,0].set_ylabel('x')

        axs[0,1].plot(pose_trace['time'][tstamp_posetrace_idx[i]:tstamp_posetrace_idx[i+1]+1], 
            pose_trace['extrinsic_matrix'][tstamp_posetrace_idx[i]:tstamp_posetrace_idx[i+1]+1,1, 3],color=color)
        axs[0,1].set_xlabel('t'); axs[0,1].set_ylabel('y')

        axs[1,0].plot(pose_trace['time'][tstamp_posetrace_idx[i]:tstamp_posetrace_idx[i+1]+1], 
            pose_trace['extrinsic_matrix'][tstamp_posetrace_idx[i]:tstamp_posetrace_idx[i+1]+1,2, 3],color=color)
        axs[1,0].set_xlabel('t'); axs[1,0].set_ylabel('z')

        axs[1,1].plot(pose_trace['extrinsic_matrix'][tstamp_posetrace_idx[i]:tstamp_posetrace_idx[i+1]+1,0, 3], 
            pose_trace['extrinsic_matrix'][tstamp_posetrace_idx[i]:tstamp_posetrace_idx[i+1]+1,2, 3],color=color)
        axs[1,1].set_xlabel('x'); axs[1,1].set_ylabel('y')
    plt.savefig(f"/home/sax1rng/Projects/VLN-CE-Plan/tests/media/pose_trace_stepwise_{episode_id}.jpg")
    plt.close()
    return colors

def plot_gtdata_discretized(tstamp_gtdata_idx, gt_data, episode_id, colors):
    tstamp_gt_actions_idx, tstamp_gt_locations_idx = tstamp_gtdata_idx

    gt_locations = np.array(gt_data[episode_id]['locations'])
    loc_steps = np.arange(gt_locations.shape[0])
    gt_actions = np.array(gt_data[episode_id]['actions'])
    action_steps = np.arange(gt_actions.shape[0])

    fig, axs = plt.subplots(2, 2, figsize=(5, 5), constrained_layout=True)
    fig.suptitle('GT data', fontsize=12)

    for i in range(len(tstamp_gt_actions_idx)-1):
        axs[0,0].plot(loc_steps[tstamp_gt_locations_idx[i]:tstamp_gt_locations_idx[i+1]+1], gt_locations[tstamp_gt_locations_idx[i]:tstamp_gt_locations_idx[i+1]+1,0],color=colors[i])
        axs[0,0].set_xlabel('fwd steps'); axs[0,0].set_ylabel('x')

        # axs[0,1].plot(gt_locations[tstamp_gt_locations_idx[i]:tstamp_gt_locations_idx[i+1]+1,1],color=colors[i])
        # axs[0,1].set_xlabel('fwd steps'); axs[0,1].set_ylabel('y')

        axs[0,1].plot(action_steps[tstamp_gt_actions_idx[i]:tstamp_gt_actions_idx[i+1]+1], gt_actions[tstamp_gt_actions_idx[i]:tstamp_gt_actions_idx[i+1]+1],color=colors[i])
        axs[0,1].set_xlabel('fwd steps'); axs[0,1].set_ylabel('actions')

        axs[1,0].plot(loc_steps[tstamp_gt_locations_idx[i]:tstamp_gt_locations_idx[i+1]+1], gt_locations[tstamp_gt_locations_idx[i]:tstamp_gt_locations_idx[i+1]+1,2],color=colors[i])
        axs[1,0].set_xlabel('fwd steps'); axs[1,0].set_ylabel('z')

        axs[1,1].plot(gt_locations[tstamp_gt_locations_idx[i]:tstamp_gt_locations_idx[i+1]+1,0],
            gt_locations[tstamp_gt_locations_idx[i]:tstamp_gt_locations_idx[i+1]+1,2], color=colors[i])
        axs[1,1].set_xlabel('x'); axs[1,1].set_ylabel('z')

    plt.savefig(f"/home/sax1rng/Projects/VLN-CE-Plan/tests/media/gt_data_stepwise_{episode_id}.jpg")
    plt.close()
    return colors

def load_pose_trace(instruction_id):
    pose_traces_path = '/fs/scratch/rng_cr_aas3_gpu_user_c_lf/sp02_025_rrt/datasets/rxr-data/pose_traces/rxr_train/'
    pose_traces_file_name = "{id:06}_guide_pose_trace.npz"

    pose_traces_file_name = pose_traces_path + pose_traces_file_name.format(id=int(instruction_id))
    pose_traces = np.load(pose_traces_file_name)
    return pose_traces

def load_gt_data(data_type='train'):
    gt_data_file = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_guide_gt.json.gz'
    gt_data = {}
    with gzip.open(gt_data_file,"rt",) as f:
        gt_data.update(json.load(f))
    return gt_data

def setup_envs(save_video=False):
    # Load config
    config_path = 'vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml'
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
        video_dir = '/home/sax1rng/Projects/VLN-CE-Plan/tests'
    config.freeze()

    with gzip.open(config.IL.RECOLLECT_TRAINER.trajectories_file, "rt") as f:
        trajectories = json.load(f)

    envs = construct_envs(
        config,
        get_env_class(config.ENV_NAME),
        episodes_allowed=list(trajectories.keys()),
    )

    observations = envs.reset()
    observations = extract_instruction_tokens(
        observations,
        config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
    )
    return envs, trajectories, config

def get_llama_sequence(instruction, llama2_tokenizer, llama2_pipeline):
    found_good_sequence = False
    trail=0
    while not found_good_sequence:
        trail += 1
        sequences = llama2_pipeline(
            f'Split/divide/chunk the following instructions to numbered step by step instructions (e.g. Step 1: ). Use all the words as in instruction. Add no new words. Do not remove words from the given instruction: "{instruction}"',
            do_sample=True,
            top_k=20,
            num_return_sequences=1,
            eos_token_id=llama2_tokenizer.eos_token_id,
            max_length=1000)
        if 'step 2: ' in (sequences[0]['generated_text']).lower():
            found_good_sequence = True
    return sequences[0]['generated_text']

def clean_stepwise_instruction(stepwise_instruction):
    stepwise_instruction = stepwise_instruction.replace('.','')
    stepwise_instruction = stepwise_instruction.replace(',','')
    stepwise_instruction = stepwise_instruction.lower()

    stepwise_instruction_steps = [i for i in stepwise_instruction.split('\n') if i.startswith('step')]
    assert len(stepwise_instruction_steps) > 0, 'Llama constructed inadmissable stepwise instructions.'
    stepwise_instruction_steps_c = [i.split(': ')[-1] for i in stepwise_instruction_steps]
    stepwise_instruction_steps_words = [i.split(' ') for i in stepwise_instruction_steps_c]
    return stepwise_instruction_steps_words, stepwise_instruction_steps

def clean_timed_instruction(timed_instruction):
    timed_instruction_clean = []
    prev_end_time = 0.0
    for word_dict in timed_instruction:
        word_dict_new = word_dict.copy()
        word_dict_new['word'] = word_dict_new['word'].lower()
        word_dict_new['word'] = word_dict_new['word'].replace('.','')
        word_dict_new['word'] = word_dict_new['word'].replace(',','')

        if ('end_time' not in word_dict.keys()) or ('start_time' not in word_dict.keys()):
            word_dict_new['start_time'] = prev_end_time
            word_dict_new['end_time'] = prev_end_time
        
        words = word_dict_new['word'].split(' ')
        try:
            times = np.linspace(word_dict_new['start_time'], word_dict_new['end_time'], len(words)+1)
        except:
            import ipdb; ipdb.set_trace()
        for i, word in enumerate(words):
            word_dict_new2 = word_dict_new.copy()
            word_dict_new2['word'] = word
            word_dict_new2['start_time'] = times[i]
            word_dict_new2['end_time'] = times[i+1]
            timed_instruction_clean.append(word_dict_new2)
        
        prev_end_time = timed_instruction_clean[-1]['end_time']
    
    words_in_gt_instruction = [i['word'] for i in timed_instruction_clean]
    return timed_instruction_clean, words_in_gt_instruction

def get_end_time_stamps_for_stepwise_instructions(stepwise_instruction, timed_instruction, pose_trace):
    stepwise_instruction_steps_words, stepwise_instruction_steps = clean_stepwise_instruction(stepwise_instruction)
    total_words=sum([len(i) for i in stepwise_instruction_steps_words])

    timed_instruction_clean, words_in_gt_instruction = clean_timed_instruction(timed_instruction)
    
    window = 3
    current_idx_gt_words = 0
    tstamp_posetrace_idx = [0]
    for instr in stepwise_instruction_steps_words: # Loop through stepwise instructions
        try:
            # for idx in range(current_idx_gt_words, current_idx_gt_words+5):
            #     found = False if words_in_gt_instruction[idx].find(instr[0]) == -1 else True
            #     if found:
            #         first_word_idx = idx
            #         break
            # if not found:
            #     first_word_idx = current_idx_gt_words
            try:
                first_word_idx = current_idx_gt_words+words_in_gt_instruction[current_idx_gt_words:current_idx_gt_words+5].index(instr[0]) # search for first word in timed instruction
            except:
                first_word_idx = current_idx_gt_words
                    
            search_window_start = min( len(words_in_gt_instruction)-1, max(0,first_word_idx+len(instr)-window-1) )
            search_window_end = min(len(words_in_gt_instruction), first_word_idx+len(instr)+window-1)
            try:
                last_word_idx = search_window_start + words_in_gt_instruction[search_window_start:search_window_end].index(instr[-1]) # search for last word in timed instruction
            except:
                last_word_idx = search_window_start
            
            ts = timed_instruction_clean[last_word_idx]['end_time']

            # found_end_time = False
            # end_time_idx = last_word_idx
            # while not found_end_time:
            #     if 'end_time' in timed_instruction_clean[end_time_idx].keys():
            #         ts = timed_instruction_clean[end_time_idx]['end_time']
            #         found_end_time = True
            #     else:
            #         end_time_idx -= 1
            
            tstamp_posetrace_idx.append(np.argmin(abs(pose_trace['time'] - ts)))
            current_idx_gt_words = last_word_idx + 1
            print('instr successful',instr)
        except:
            import ipdb; ipdb.set_trace()
        
    return tstamp_posetrace_idx, stepwise_instruction_steps

def get_gt_end_steps_for_stepwise_instructions(tstamp_posetrace_idx, pose_trace, gt_data, episode_id):
    gt_locations = np.array(gt_data[episode_id]['locations'])
    gt_pos = gt_locations[:,[0,2]] # Considering xz coordinates
    gt_actions = np.array(gt_data[episode_id]['actions'])
    action_fwd_idx = np.arange(len(gt_actions))[gt_actions == 1]

    assert len(action_fwd_idx) == (gt_locations.shape[0]-1), "Number of fwd steps not equal to actions==1"

    tstamp_gt_actions_idxs = [0]
    tstamp_gt_locations_idxs = [0]
    for i in range(len(tstamp_posetrace_idx)-1):
        poset = np.array(
            [[pose_trace['extrinsic_matrix'][tstamp_posetrace_idx[i+1],0, 3],
            pose_trace['extrinsic_matrix'][tstamp_posetrace_idx[i+1],2, 3]]]
        )
        gtpos_idx = np.argmin(np.linalg.norm(gt_pos-poset, axis=1))
        if gtpos_idx == 0:
            tstamp_gt_actions_idxs.append(action_fwd_idx[gtpos_idx]-1)
        else:
            tstamp_gt_actions_idxs.append(action_fwd_idx[gtpos_idx-1])
        tstamp_gt_locations_idxs.append(gtpos_idx)

    return tstamp_gt_actions_idxs, tstamp_gt_locations_idxs

def rollout_episodes(envs, trajectories, config, tstamp_gtdata_idxs, tstamp_posetrace_idxs_step_instr):
    save_video = True
    save_final_frame = True
    env_step = [0 for _ in range(envs.num_envs)]
    _env_infos = [[] for _ in range(envs.num_envs)]
    _world_time = [[] for _ in range(envs.num_envs)]
    rgb_frames = [[] for _ in range(envs.num_envs)]

    stepwise_instruction_steps = [tstamp_posetrace_idxs_step_instr[i][1] for i in range(envs.num_envs)]
    tstamp_gt_actions_idx = [tstamp_gtdata_idxs[i][0] for i in range(envs.num_envs)]
    
    instr_step = 0
    
    while True:
        current_episodes = envs.current_episodes()
        prev_eps = current_episodes # resets on done

        # get the next action for each env
        actions = [
            trajectories[ep.episode_id][env_step[i]][1]
            for i, ep in enumerate(current_episodes)
        ]

        outputs = envs.step(actions)
        observations, _, dones, infos = [list(x) for x in zip(*outputs)]
        
        observations = extract_instruction_tokens(
            observations,
            config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
        )
        current_episodes = envs.current_episodes()

        for i in range(envs.num_envs):
            episode_id = prev_eps[i].episode_id
            try:
                if env_step[i] >= tstamp_gt_actions_idx[i][instr_step+1]:
                    instr_step = min(instr_step+1, len(stepwise_instruction_steps[i])-1)
            except:
                import ipdb; ipdb.set_trace()
            
            env_step[i] += 1
            
            if save_video:
                frame = observations_to_image(observations[i], infos[i])
                # _text = text_to_append(prev_eps[i].instruction)
                _text = stepwise_instruction_steps[i][instr_step]
                frame = append_text_to_image(frame, _text)
                rgb_frames[i].append(frame)
            
            _env_infos[i].append(infos[i]['agent_state'])
            _world_time[i].append(infos[i]['world_time'])
            
            if dones[i]:
                print('---------DONE: ', current_episodes[i].episode_id)
                
                if save_video:
                    generate_video(
                        video_option=['disk'],
                        video_dir='/home/sax1rng/Projects/VLN-CE-Plan/tests/media',
                        images=rgb_frames[i],
                        episode_id=prev_eps[i].episode_id,
                        checkpoint_idx=0,
                        metrics={"spl": infos[i]["spl"]},
                        tb_writer=None,
                        fps=2,
                    )
                # if save_final_frame:
                #     frame = observations_to_image(observations[i], infos[i])
                #     _text = text_to_append(prev_eps[i].instruction)
                #     frame = append_text_to_image(frame, _text)
                #     cv2.imwrite(f'/home/sax1rng/Projects/VLN-CE-Plan/tests/media/top_down_map_episode_id_{episode_id}', frame)
                return _env_infos[i]

if __name__== "__main__":
    save_video = True
    os.chdir('/home/sax1rng/Projects/VLN-CE-Plan')

    gt_data = load_gt_data()

    llama2_tokenizer, llama2_pipeline = get_llama2_model()

    envs, trajectories, config = setup_envs(save_video=True)

    while True:
        
        current_episodes = envs.current_episodes()
        instructions = [current_episodes[i].instruction.instruction_text for i in range(envs.num_envs)]

        llama_sequences = [get_llama_sequence(instructions[i], llama2_tokenizer, llama2_pipeline) for i in range(envs.num_envs)]
        
        print(llama_sequences[0])
        pose_traces = [load_pose_trace(current_episodes[i].instruction.instruction_id) for i in range(envs.num_envs)]

        tstamp_posetrace_idxs_step_instr = [
            get_end_time_stamps_for_stepwise_instructions(
                llama_sequences[i], 
                current_episodes[i].instruction.timed_instruction, 
                pose_traces[i]) for i in range(envs.num_envs)]
        
        colors = [plot_pose_trace_discretized(tstamp_posetrace_idxs_step_instr[i][0], pose_traces[i], current_episodes[i].episode_id) for i in range(envs.num_envs)]

        tstamp_gtdata_idxs = [
            get_gt_end_steps_for_stepwise_instructions(
                tstamp_posetrace_idxs_step_instr[i][0], 
                pose_traces[i], 
                gt_data,
                current_episodes[i].episode_id) for i in range(envs.num_envs)]
        
        _ = [plot_gtdata_discretized(tstamp_gtdata_idxs[i], gt_data, current_episodes[i].episode_id, colors[i]) for i in range(envs.num_envs)]
        start = time.time()
        if save_video:
            infos = rollout_episodes(envs, trajectories, config, tstamp_gtdata_idxs, tstamp_posetrace_idxs_step_instr)
        else:
            observations = [envs.reset_at(i)[0] for i in range(envs.num_envs)]
        end = time.time()
        
        print(f"time_elapsed for episode id {episode_id} is {end - start}")

