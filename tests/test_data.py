import gzip, json, os
from tqdm import trange
from tqdm import tqdm

if __name__== "__main__":
    
    data_type = 'val_unseen'
    data_location = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_guide.json.gz'
    data = {}
    with gzip.open(data_location,"rt",) as f:
        data.update(json.load(f))
    all_ep_idxs = [i for i in range(len(data['episodes'])) if ('en' in data['episodes'][i]['instruction']['language'])]
    new_data = data.copy()

    # Check BERT features 
    print('Checking saved BERT features')
    save_dir = f'/fs/scratch/rng_cr_aas3_gpu_user_c_lf/sp02_025_rrt/datasets/rxr-data/text_features_highlevel_instr/rxr_{data_type}/'
    save_file_name = "{id:06}_en_text_features.npz"
    for i in tqdm(all_ep_idxs):
        instruction_id = data['episodes'][i]['instruction']['instruction_id']
        file_name = save_dir + save_file_name.format(id=int(instruction_id))
        if os.path.exists(file_name):
            continue
        else:
            print('File not found')
            import ipdb; ipdb.set_trace()
    print('-------------------------All files found! BERT features test successful----------------------')

    for i in trange(37):
        data_loc = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/high_level_instr/{data_type}_guide_high_level_instr_{i}.json.gz'
        data_hl = {}
        with gzip.open(data_loc,"rt",) as f:
            data_hl.update(json.load(f))
        ep_idx_sub = list(data_hl.keys())
        for ep_id in tqdm(ep_idx_sub):
            new_data['episodes'][int(ep_id)]['instruction']['high_level_instruction'] = data_hl[ep_id]

    print('Finished copying data. Checking data')
    for i in tqdm(all_ep_idxs):
        if 'high_level_instruction' in new_data['episodes'][i]['instruction'].keys():
            continue
        else:
            print('Key not found')
            import ipdb; ipdb.set_trace()
    
    print(f'Saving new {data_type} file')
    output_file = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_guide_high_level_instr.json.gz'
    with gzip.open(output_file, "wt") as f:
        f.write(json.dumps(new_data))
        print("Output file saved at ", output_file)