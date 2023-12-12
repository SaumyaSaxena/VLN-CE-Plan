import sys, os
cwd = os.getcwd()
import numpy as np
import gzip, json

def load_data(data_type='train', role='guide', subtasks=False):
    if subtasks:
        data_location = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_subtasks_rollouts.json.gz'
    else:
        data_location = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}.json.gz'
    data = {}
    with gzip.open(data_location,"rt",) as f:
        data.update(json.load(f))
    return data

def load_gt_data(data_type='train', role='guide', subtasks=False):
    if subtasks:
        gt_data_file = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_subtasks_rollouts_gt.json.gz'
    else:
        gt_data_file = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_gt.json.gz'
    gt_data = {}
    with gzip.open(gt_data_file,"rt",) as f:
        gt_data.update(json.load(f))
    return gt_data

def merge_episodes(gt_data, data, start_i, end_i, episode_idx_new):
    episode_new = {}
    episode_new['episode_id'] = str(episode_idx_new)
    episode_new['trajectory_id'] = str(episode_idx_new)
    episode_new['scene_id'] = data['episodes'][start_i]['scene_id']
    episode_new['info'] = data['episodes'][start_i]['info']
    episode_new['reference_path'] = data['episodes'][start_i]['reference_path']
    episode_new['start_position'] = gt_data[str(start_i)]['locations'][0]
    episode_new['start_rotation'] = gt_data[str(start_i)]['rotations'][0]
    episode_new['goals'] = [
        {'position': gt_data[str(end_i)]['locations'][-1],
        'radius': 3.0}]

    instruction = {}
    instruction['instruction_id'] = str(episode_idx_new)
    instruction['language'] = data['episodes'][start_i]['instruction']['language']
    instruction['annotator_id'] = data['episodes'][start_i]['instruction']['annotator_id']
    instruction['edit_distance'] = data['episodes'][start_i]['instruction']['edit_distance']
    instruction['instruction_text'] = ' '.join(
            [data['episodes'][i]['instruction']['instruction_text'] 
            for i in range(start_i, end_i+1)]
        )
    instruction['timed_instruction'] = sum(
        [data['episodes'][i]['instruction']['timed_instruction'] 
            for i in range(start_i, end_i+1)], []
    )
    episode_new['instruction'] = instruction

    gt_data_new = {'actions': [], 'locations': [], 'rotations': [], 'forward_steps': 0}
    gt_data_new['actions'] = sum(
        [gt_data[str(i)]['actions'][:-1]
            for i in range(start_i, end_i)], []
        )
    gt_data_new['actions'].extend(gt_data[str(end_i)]['actions'])
    
    gt_data_new['locations'] = sum(
        [gt_data[str(i)]['locations'][:-2]
            for i in range(start_i, end_i)], []
        )
    gt_data_new['locations'].extend(gt_data[str(end_i)]['locations'])

    gt_data_new['rotations'] = sum(
        [gt_data[str(i)]['rotations'][:-2]
            for i in range(start_i, end_i)], []
        )
    gt_data_new['rotations'].extend(gt_data[str(end_i)]['rotations']) 

    gt_data_new['forward_steps'] = int(np.sum(np.array(gt_data_new['actions'])==1))
    gt_data_new['original_episode_idx'] = gt_data[str(start_i)]['original_episode_idx']

    return gt_data_new, episode_new

def merge_with_next(gt_data, data, episode_idx_old, episode_idx_new):
    i = episode_idx_old + 1
    if i == len(list(gt_data.keys())):
        data_episode = data['episodes'][episode_idx_old].copy()
        data_episode['episode_id'] = str(episode_idx_new)
        data_episode['trajectory_id'] = str(episode_idx_new)
        return i, gt_data[str(episode_idx_old)], data_episode
   
    while True:
        if i == len(list(gt_data.keys())):
            i = i - 1
            break
        elif not (gt_data[str(i)]['original_episode_idx'] == gt_data[str(episode_idx_old)]['original_episode_idx']):
            i = i - 1
            break
        else:
            if gt_data[str(i)]['forward_steps'] > 0:
                break
            else:
                i += 1

    assert gt_data[str(i)]['original_episode_idx'] == gt_data[str(episode_idx_old)]['original_episode_idx'], "Merging data not from same episode"
    assert i < len(list(gt_data.keys())), "length too long"

    gt_episode, data_episode = merge_episodes(gt_data, data, episode_idx_old, i, episode_idx_new)
    
    return i+1, gt_episode, data_episode

if __name__== "__main__":

    data_type = 'val_seen'
    role = 'guide'
    subtasks = True

    data = load_data(data_type=data_type, role=role, subtasks=subtasks)
    gt_data = load_gt_data(data_type=data_type, role=role, subtasks=subtasks)
    
    print("Loaded data")
    num_episodes = len(list(gt_data.keys()))

    data_new = {'episodes': []}
    gt_data_new = {}
    episode_idx_new = 0
    episode_idx_old = 0

    while episode_idx_old < num_episodes:
        if episode_idx_old % 100 == 0:
            print(f'done {episode_idx_old}/{num_episodes}')
        # if len(gt_data[str(episode_idx_old)]['actions']) == 1:

        if gt_data[str(episode_idx_old)]['forward_steps'] == 0:
            episode_idx_old, gt_episode, data_episode = merge_with_next(gt_data, data, episode_idx_old, episode_idx_new)
        else:
            gt_episode = gt_data[str(episode_idx_old)].copy()
            data_episode = data['episodes'][episode_idx_old].copy()
            data_episode['episode_id'] = str(episode_idx_new)
            data_episode['trajectory_id'] = str(episode_idx_new)
            episode_idx_old += 1
        data_new['episodes'].append(data_episode)
        gt_data_new[str(episode_idx_new)] = gt_episode
        episode_idx_new += 1
    
    print(f'Saving new {data_type} file')
    output_file = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_subtasks_rollouts_merged.json.gz'
    with gzip.open(output_file, "wt") as f:
        f.write(json.dumps(data_new))
        print("Output file saved at ", output_file)

    print(f'Saving new GT {data_type} file')
    output_file = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_subtasks_rollouts_merged_gt.json.gz'
    with gzip.open(output_file, "wt") as f:
        f.write(json.dumps(gt_data_new))
        print("Output file saved at ", output_file)


