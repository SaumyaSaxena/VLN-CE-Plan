import json, gzip, os
from tqdm import tqdm

cwd = os.getcwd()

def load_data(data_type='train', role='guide', subtasks=False):
    if subtasks:
        data_location = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_subtasks_rollouts_merged_hli.json.gz'
    else:
        data_location = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_high_level_instr.json.gz'
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

if __name__== "__main__":

    data_type = 'train'
    role = 'guide'

    data = load_data(data_type=data_type, role=role, subtasks=False)
    gt_data = load_gt_data(data_type=data_type, role=role, subtasks=False)

    data_subtasks = load_data(data_type=data_type, role=role, subtasks=True)
    gt_data_subtasks = load_gt_data(data_type=data_type, role=role, subtasks=True)

    if True:
        # Adding original index and high level instructions to subtask data
        new_data_subtasks = data_subtasks.copy()
        for ep_idx in tqdm(gt_data_subtasks.keys()):
            original_episode_idx = gt_data_subtasks[ep_idx]['original_episode_idx']
            new_data_subtasks['episodes'][int(ep_idx)]['original_episode_idx'] = original_episode_idx
            new_data_subtasks['episodes'][int(ep_idx)]['instruction']['high_level_instruction'] = data['episodes'][original_episode_idx]['instruction']['high_level_instruction']
            import ipdb; ipdb.set_trace()
        # print(f'Saving new {data_type} file')
        # output_file = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_subtasks_rollouts_merged_hli.json.gz'
        # with gzip.open(output_file, "wt") as f:
        #     f.write(json.dumps(new_data_subtasks))
        #     print("Output file saved at ", output_file)
    
    if True:
        # Save a file that maps from full task indices to subtask indices
        dataset_size = len(data['episodes'])
        _keys = [str(i) for i in range(dataset_size) if ('en' in data['episodes'][i]['instruction']['language'])]
        mapping_original_to_subtask = {k: [] for k in _keys}
        for ep_idx in tqdm(gt_data_subtasks.keys()):
            original_episode_idx = gt_data_subtasks[ep_idx]['original_episode_idx']
            mapping_original_to_subtask[str(original_episode_idx)].append(ep_idx)

        print(f'Saving new {data_type} file')
        output_file = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_mapping_original_to_subtask.json.gz'
        with gzip.open(output_file, "wt") as f:
            f.write(json.dumps(mapping_original_to_subtask))
            print("Output file saved at ", output_file)