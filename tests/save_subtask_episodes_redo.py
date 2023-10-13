import sys, os
sys.path.append('/home/sax1rng/Projects/VLN-CE-Plan')
import gzip, json
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def load_data(data_type='train', role='guide', subtasks=False):
    postfix_subtask = '_subtasks' if subtasks else ''
    data_location = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}{postfix_subtask}.json.gz'
    data = {}
    with gzip.open(data_location,"rt",) as f:
        data.update(json.load(f))
    return data

def load_gt_data(data_type='train', role='guide', subtasks=False):
    postfix_subtask = '_subtasks' if subtasks else ''
    gt_data_file = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}{postfix_subtask}_gt.json.gz'
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

def clean_stepwise_instruction(stepwise_instruction, few_future_isntructions):
    stepwise_instruction = stepwise_instruction.replace('.','')
    stepwise_instruction = stepwise_instruction.replace(',','')
    stepwise_instruction = stepwise_instruction.lower()

    stepwise_instruction_steps = [i for i in stepwise_instruction.split('\n') if (i.replace(' ', '')).startswith('step')]
    if len(stepwise_instruction_steps) == 0:
        print('Llama constructed inadmissable stepwise instructions.')
        import ipdb; ipdb.set_trace()
    # assert len(stepwise_instruction_steps) > 0, 'Llama constructed inadmissable stepwise instructions.'
    stepwise_instruction_steps_c = [i.split(': ')[-1] for i in stepwise_instruction_steps if len((i.split(': ')[-1]).replace(' ', ''))>0]
    stepwise_instruction_steps_words = [i.split(' ') for i in stepwise_instruction_steps_c]
    return stepwise_instruction_steps_words, stepwise_instruction_steps_c

def clean_timed_instruction(timed_instruction):
    timed_instruction_clean = []
    prev_end_time = 0.0
    for word_dict in timed_instruction:
        word_dict_new = word_dict.copy()
        word_dict_new['word'] = word_dict_new['word'].lower()
        word_dict_new['word'] = word_dict_new['word'].replace('.','')
        word_dict_new['word'] = word_dict_new['word'].replace(',','')

        if ('end_time' not in word_dict.keys()) or ('start_time' not in word_dict.keys()) or (word_dict_new['start_time'] is None) or (word_dict_new['end_time'] is None) or \
            (math.isnan(word_dict_new['start_time'])) or (math.isnan(word_dict_new['end_time'])):
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

def get_end_time_stamps_for_stepwise_instructions(stepwise_instruction, timed_instruction, pose_trace, few_future_isntructions):
    stepwise_instruction_steps_words, stepwise_instruction_steps = clean_stepwise_instruction(stepwise_instruction, few_future_isntructions)
    total_words=sum([len(i) for i in stepwise_instruction_steps_words])

    timed_instruction_clean, words_in_gt_instruction = clean_timed_instruction(timed_instruction)
    
    window = 3
    current_idx_gt_words = 0
    tstamp_posetrace_idx = [0]
    stepwise_timed_instruction = []
    for instr in stepwise_instruction_steps_words: # Loop through stepwise instructions
        try:
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
            
            tstamp_posetrace_idx.append(np.argmin(abs(pose_trace['time'] - ts)))
            current_idx_gt_words = last_word_idx + 1
            # print('instr successful',instr)
            stepwise_timed_instruction.append(timed_instruction_clean[first_word_idx:last_word_idx+1])
        except:
            import ipdb; ipdb.set_trace()
        
    return tstamp_posetrace_idx, stepwise_instruction_steps, stepwise_timed_instruction

def get_gt_end_steps_for_stepwise_instructions(tstamp_posetrace_idx, pose_trace, gt_data, episode_id):
    gt_locations = np.array(gt_data[episode_id]['locations'])
    gt_pos = gt_locations[:,[0,2]] # Considering xz coordinates
    gt_actions = np.array(gt_data[episode_id]['actions'])
    action_fwd_idx = np.arange(len(gt_actions))[gt_actions == 1]

    if len(action_fwd_idx) == 0:
        tstamp_gt_locations_idxs = [0]*len(tstamp_posetrace_idx)
        tstamp_gt_actions_idxs = [0]*len(tstamp_posetrace_idx)
        return tstamp_gt_actions_idxs, tstamp_gt_locations_idxs

    assert len(action_fwd_idx) == (gt_locations.shape[0]-1), "Number of fwd steps not equal to actions==1"

    tstamp_gt_actions_idxs = [0]
    tstamp_gt_locations_idxs = [0]
    for i in range(len(tstamp_posetrace_idx)-1):
        poset = np.array(
            [[pose_trace['extrinsic_matrix'][tstamp_posetrace_idx[i+1],0, 3],
            pose_trace['extrinsic_matrix'][tstamp_posetrace_idx[i+1],2, 3]]]
        )
        gtpos_idx = tstamp_gt_locations_idxs[-1] + np.argmin(
            np.linalg.norm(gt_pos[tstamp_gt_locations_idxs[-1]:]-poset, axis=1))

        if gtpos_idx == 0:
            tstamp_gt_actions_idxs.append(0)
        else:
            tstamp_gt_actions_idxs.append(action_fwd_idx[gtpos_idx-1])
        tstamp_gt_locations_idxs.append(gtpos_idx)

    return tstamp_gt_actions_idxs, tstamp_gt_locations_idxs

if __name__== "__main__":

    data_type = 'train'
    role = 'guide'

    data = load_data(data_type=data_type, role=role)
    gt_data = load_gt_data(data_type=data_type, role=role)

    data_subtasks = load_data(data_type=data_type, role=role, subtasks=True)
    gt_data_subtasks = load_gt_data(data_type=data_type, role=role, subtasks=True)

    data_new = {'episodes': []}
    gt_data_new = {}

    dataset_size = len(data['episodes'])
    en_idx = [i for i in range(dataset_size) if ('en' in data['episodes'][i]['instruction']['language'])]

    subtask_dataset_size = len(data_subtasks['episodes'])

    subtask_dir = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/subtask_instructions/'
    subtask_files = os.listdir(subtask_dir)

    episode_idx_new = 0
    for subtask_file in tqdm(subtask_files):
        llama_sequences = {}
        with gzip.open(subtask_dir+subtask_file,"rt",) as f:
            llama_sequences.update(json.load(f))

        ep_idx_sub = list(llama_sequences.keys())

        for ep_idx in ep_idx_sub:
            instruction_id = data['episodes'][int(ep_idx)]['instruction']['instruction_id']
            pose_trace = load_pose_trace(instruction_id, data_type=data_type, role=role)
            end_idx = min(episode_idx_new+20,subtask_dataset_size)
            few_future_isntructions = [data_subtasks['episodes'][d]['instruction']['instruction_text'] for d in range(episode_idx_new,end_idx)]
            tstamp_posetrace_idx, stepwise_instruction_steps, stepwise_timed_instruction = get_end_time_stamps_for_stepwise_instructions(
                llama_sequences[ep_idx], 
                data['episodes'][int(ep_idx)]['instruction']['timed_instruction'], 
                pose_trace,
                few_future_isntructions)
            
            tstamp_gt_actions_idxs, tstamp_gt_locations_idxs = get_gt_end_steps_for_stepwise_instructions(
                tstamp_posetrace_idx, 
                pose_trace, 
                gt_data,
                data['episodes'][int(ep_idx)]['episode_id'])
            # print("Full instruction: \n", data['episodes'][int(ep_idx)]['instruction']['instruction_text'])

            for step_idx, sub_instr in enumerate(stepwise_instruction_steps):
                sub_instr_saved = data_subtasks['episodes'][episode_idx_new]['instruction']['instruction_text']
                # print("Saved sub instruction: \n", data_subtasks['episodes'][int(episode_idx_new)]['instruction']['instruction_text'])
                # print("New sub instruction: \n", sub_instr)
                if sub_instr_saved == sub_instr:
                    gt_data_new[str(episode_idx_new)] = {'actions': [], 'locations': [], 'forward_steps': 0}
                    gt_data_new[str(episode_idx_new)]['actions'] = gt_data[data['episodes'][int(ep_idx)]['episode_id']]['actions'][
                        tstamp_gt_actions_idxs[step_idx]+1:tstamp_gt_actions_idxs[step_idx+1]+1]
                    gt_data_new[str(episode_idx_new)]['locations'] = gt_data[data['episodes'][int(ep_idx)]['episode_id']]['locations'][
                        tstamp_gt_locations_idxs[step_idx]:tstamp_gt_locations_idxs[step_idx+1]+1]
                    gt_data_new[str(episode_idx_new)]['forward_steps'] = int(np.sum(np.array(gt_data_new[str(episode_idx_new)]['actions'])==1))
                else:
                    import ipdb; ipdb.set_trace()
                episode_idx_new += 1
        
    print(f'Saving new GT {data_type} file')
    output_file = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_subtasks_gt2.json.gz'
    with gzip.open(output_file, "wt") as f:
        f.write(json.dumps(gt_data_new))
        print("Output file saved at ", output_file)