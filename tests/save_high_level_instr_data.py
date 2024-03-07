import json, gzip, os
from tqdm import tqdm

cwd = os.getcwd()

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

if __name__== "__main__":

    data_type = 'val_unseen'
    role = 'guide'
    subtasks = False

    data = load_data(data_type=data_type, role=role, subtasks=subtasks)
    dataset_size = len(data['episodes'])

    gt_data = load_gt_data(data_type=data_type, role=role, subtasks=subtasks)

    all_ep_idxs = [i for i in range(dataset_size) if ('en' in data['episodes'][i]['instruction']['language'])]

    new_data = data.copy()

    hl_instruction_files = os.listdir(f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/high_level_instr/')

    for hl_file_name in tqdm(hl_instruction_files):
        hl_file_loc = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/high_level_instr/' + hl_file_name
        data_hl = {}
        with gzip.open(hl_file_loc,"rt",) as f:
            data_hl.update(json.load(f))
        
        ep_idx = list(data_hl.keys())
        for e in ep_idx:
            new_data['episodes'][int(e)]['instruction']['high_level_instruction'] = data_hl[e]
            # print("High level instruction: ", data_hl[e])
            # print("Full instruction: ", new_data['episodes'][int(e)]['instruction']['instruction_text'])
            # import ipdb; ipdb.set_trace()

    
    print('Finished copying data. Checking data')
    for i in tqdm(all_ep_idxs):
        if 'high_level_instruction' in new_data['episodes'][i]['instruction'].keys():
            continue
        else:
            print('Key not found')
            import ipdb; ipdb.set_trace()

    print(f'Saving new {data_type} file')
    output_file = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_high_level_instr.json.gz'
    with gzip.open(output_file, "wt") as f:
        f.write(json.dumps(new_data))
        print("Output file saved at ", output_file)