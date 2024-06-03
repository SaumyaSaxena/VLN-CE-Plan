import sys, os
cwd = os.getcwd()
import gzip, json

def load_data(data_type='train', role='guide'):
    data_location = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_subtasks_rollouts_merged_hli.json.gz'
    data = {}
    with gzip.open(data_location,"rt",) as f:
        data.update(json.load(f))
    return data

def load_gt_data(data_type='train', role='guide'):
    gt_data_file = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_subtasks_rollouts_merged_gt.json.gz'
    gt_data = {}
    with gzip.open(gt_data_file,"rt",) as f:
        gt_data.update(json.load(f))
    return gt_data

if __name__== "__main__":
    data_type='train'
    role='guide'

    data = load_data(data_type=data_type, role=role)
    dataset_size = len(data['episodes'])
    gt_data = load_gt_data(data_type=data_type, role=role)

    all_ep_idxs = [i for i in range(dataset_size) if ('en' in data['episodes'][i]['instruction']['language'])]
    all_ep_ids = [data['episodes'][i]['episode_id'] for i in range(dataset_size) if ('en' in data['episodes'][i]['instruction']['language'])]

    import ipdb; ipdb.set_trace()