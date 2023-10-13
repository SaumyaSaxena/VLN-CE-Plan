from transformers import AutoTokenizer
import transformers
import torch, gzip, json
import numpy as np
import matplotlib.pyplot as plt

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

def load_data(data_type='train'):
    data_location = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_guide.json.gz'
    data = {}
    with gzip.open(data_location,"rt",) as f:
        data.update(json.load(f))
    return data

def load_gt_data(data_type='train'):
    gt_data_file = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_guide_gt.json.gz'
    gt_data = {}
    with gzip.open(gt_data_file,"rt",) as f:
        gt_data.update(json.load(f))
    return gt_data

def load_pose_trace(instruction_id):
    pose_traces_path = '/fs/scratch/rng_cr_aas3_gpu_user_c_lf/sp02_025_rrt/datasets/rxr-data/pose_traces/rxr_train/'
    pose_traces_file_name = "{id:06}_guide_pose_trace.npz"

    pose_traces_file_name = pose_traces_path + pose_traces_file_name.format(id=int(instruction_id))
    pose_traces = np.load(pose_traces_file_name)
    return pose_traces

def plot_pose_trace_discretized(tstamp_posetrace_idx, pose_trace):
    
    fig, axs = plt.subplots(2, 2, figsize=(5, 5), constrained_layout=True)
    fig.suptitle('Pose traces', fontsize=12)

    for i in range(len(tstamp_posetrace_idx)-1):
        color = np.random.uniform(low=0.0, high=1.0, size=(3,))

        axs[0,0].plot(pose_traces['time'][tstamp_posetrace_idx[i]:tstamp_posetrace_idx[i+1]], 
            pose_traces['extrinsic_matrix'][tstamp_posetrace_idx[i]:tstamp_posetrace_idx[i+1],0, 3],color=color)
        axs[0,0].set_xlabel('t'); axs[0,0].set_ylabel('x')

        axs[0,1].plot(pose_traces['time'][tstamp_posetrace_idx[i]:tstamp_posetrace_idx[i+1]], 
            pose_traces['extrinsic_matrix'][tstamp_posetrace_idx[i]:tstamp_posetrace_idx[i+1],1, 3],color=color)
        axs[0,1].set_xlabel('t'); axs[0,1].set_ylabel('y')

        axs[1,0].plot(pose_traces['time'][tstamp_posetrace_idx[i]:tstamp_posetrace_idx[i+1]], 
            pose_traces['extrinsic_matrix'][tstamp_posetrace_idx[i]:tstamp_posetrace_idx[i+1],2, 3],color=color)
        axs[1,0].set_xlabel('t'); axs[1,0].set_ylabel('z')

        axs[1,1].plot(pose_traces['extrinsic_matrix'][tstamp_posetrace_idx[i]:tstamp_posetrace_idx[i+1],0, 3], 
            pose_traces['extrinsic_matrix'][tstamp_posetrace_idx[i]:tstamp_posetrace_idx[i+1],1, 3],color=color)
        axs[1,1].set_xlabel('x'); axs[1,1].set_ylabel('y')
    plt.savefig("trajs_discretized.jpg")

def get_end_time_stamps_from_instructions(stepwise_instruction, instruction_dict, pose_trace):

    stepwise_instruction = stepwise_instruction.replace('.','')
    stepwise_instruction = stepwise_instruction.replace(',','')
    stepwise_instruction = stepwise_instruction.lower()

    stepwise_instruction_steps = [i for i in stepwise_instruction.split('\n') if i.startswith('step')]
    stepwise_instruction_steps_c = [i.split(': ')[-1] for i in stepwise_instruction_steps]
    stepwise_instruction_steps_words = [i.split(' ') for i in stepwise_instruction_steps_c]
    total_words=sum([len(i) for i in stepwise_instruction_steps_words])
    
    words_in_gt_instruction = [i['word'].lower() for i in instruction_dict['timed_instruction']]
    words_in_gt_instruction = [i.replace('.','') for i in words_in_gt_instruction]
    words_in_gt_instruction = [i.replace(',','') for i in words_in_gt_instruction]
    if len(words_in_gt_instruction) == total_words:
        final_indx = np.cumsum([len(i) for i in stepwise_instruction_steps_words]) - 1

        tstamp_posetrace_idx = [0]
        for i in range(len(final_indx)):
            if 'end_time' in instruction_dict['timed_instruction'][final_indx[i]].keys():
                ts = instruction_dict['timed_instruction'][final_indx[i]]['end_time']
            else:
                ts = instruction_dict['timed_instruction'][final_indx[i]-1]['end_time']
            tstamp_posetrace_idx.append(np.argmin(abs(pose_traces['time'] - ts)))

        plot_pose_trace_discretized(tstamp_posetrace_idx, pose_trace)
        import ipdb; ipdb.set_trace()
    else:
        window = 2
        current_idx_gt_words = 0
        tstamp_posetrace_idx = [0]
        for instr in stepwise_instruction_steps_words:
            try:
                first_word_idx = words_in_gt_instruction[current_idx_gt_words:].index(instr[0]) # search for first word in timed instruction
                search_window_start = max(0,current_idx_gt_words+len(instr)-window-1)
                search_window_end = min(len(words_in_gt_instruction), current_idx_gt_words+len(instr)+window-1)
                last_word_idx = search_window_start + words_in_gt_instruction[search_window_start:search_window_end].index(instr[-1])

                if 'end_time' in instruction_dict['timed_instruction'][last_word_idx].keys():
                    ts = instruction_dict['timed_instruction'][last_word_idx]['end_time']
                else:
                    ts = instruction_dict['timed_instruction'][last_word_idx-1]['end_time']
                tstamp_posetrace_idx.append(np.argmin(abs(pose_traces['time'] - ts)))
                current_idx_gt_words = last_word_idx + 1
                print('instr successful',instr)
            except:
                import ipdb; ipdb.set_trace()
        
        plot_pose_trace_discretized(tstamp_posetrace_idx, pose_trace)
        import ipdb; ipdb.set_trace()
    
    tstamp_pose_trace = pose_traces['time']
    return tstamp_pose_trace


if __name__== "__main__":

    llama2_tokenizer, llama2_pipeline = get_llama2_model()

    data = load_data()
    gt_data = load_gt_data()
    dataset_size = len(data['episodes'])
    en_idx = [i for i in range(dataset_size) if ('en' in data['episodes'][i]['instruction']['language'])]

    while True:
        episode_id = np.random.choice(en_idx)
        instruction_text = data['episodes'][episode_id]['instruction']['instruction_text']
        instruction_id = data['episodes'][episode_id]['instruction']['instruction_id']

        print(f"Original instruction: {instruction_text}")
        sequences = llama2_pipeline(
            f'Split/divide/chunk the following instructions to numbered step by step instructions (e.g. Step 1: ). Use all the words as in instruction. Add no new words. Do not remove words from the given instruction: "{instruction_text}"',
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=llama2_tokenizer.eos_token_id,
            max_length=1000)
        
        print(sequences[0]['generated_text'])
        import ipdb; ipdb.set_trace()
        # for seq in sequences:
        #     print(f"Result: {seq['generated_text']}")
        pose_traces = load_pose_trace(instruction_id)
        
        time_stamps = get_end_time_stamps_from_instructions(sequences[0]['generated_text'], data['episodes'][episode_id]['instruction'], pose_traces)

# f'Convert the following instructions to numbered step by step instructions (e.g. Step 1: ) using same text and all the text as in the instruction: "{instruction_text}"',