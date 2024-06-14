import sys, os
cwd = os.getcwd()
import gzip, json
import matplotlib.pyplot as plt
import numpy as np

def load_data(data_type='train', role='guide', subtasks=False):
    if subtasks:
        data_location = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_subtasks_rollouts_merged_hli.json.gz'
    else:
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

if __name__== "__main__":
    data_type='train'
    role='guide'

    data = load_data(data_type=data_type, role=role, subtasks=False)
    dataset_size = len(data['episodes'])
    all_ep_idxs = [i for i in range(dataset_size) if ('en' in data['episodes'][i]['instruction']['language'])]
    dataset_size_en = len(all_ep_idxs)
    gt_data = load_gt_data(data_type=data_type, role=role, subtasks=False)

    data_subtasks = load_data(data_type=data_type, role=role, subtasks=True)
    dataset_size_subtasks = len(data_subtasks['episodes'])
    gt_data_subtasks = load_gt_data(data_type=data_type, role=role, subtasks=True)

    fwd_steps = [v['forward_steps'] for v in gt_data_subtasks.values()]

    # Create a histogram with customizations

    fig, axs = plt.subplots(3, figsize=(5, 5), constrained_layout=True)

    n_samples, bins, patches = axs[0].hist(fwd_steps, bins=268, alpha=0.5, color='blue', edgecolor='black')
    axs[0].set_xlabel('Forward steps')
    axs[0].set_ylabel('Frequency')

    num_0_steps = np.sum(np.array(fwd_steps) == 0)
    perc_steps = n_samples/dataset_size_subtasks*100
    perc_steps_cumsum = np.cumsum(perc_steps)

    axs[1].plot(bins[1:], perc_steps)
    axs[1].set_xlabel('Forward steps')
    axs[1].set_ylabel('Perc')

    axs[2].plot(bins[1:], perc_steps_cumsum)
    axs[2].set_xlabel('Forward steps')
    axs[2].set_ylabel('Cumsum perc')
    
    # Add titles and labels
    # plt.xlim(0, 50)
    
    plt.savefig(f"/home/saumyas/Projects/VLN-CE-Plan/tests/media/subtasks_forward_steps_hist.jpg")
    plt.close()

    import ipdb; ipdb.set_trace()