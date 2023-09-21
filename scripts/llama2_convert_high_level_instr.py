from transformers import AutoTokenizer
import transformers
import torch, gzip, json
import numpy as np

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

def get_end_time_stamps_from_instructions(stepwise_instruction, instruction_dict):
    print(stepwise_instruction.split('\n'))
    import ipdb; ipdb.set_trace()
    return time_stamps


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
            f'Convert the following instructions to numbered step by step instructions (e.g. Step 1: ) using same text as in the instruction: "{instruction_text}"',
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=llama2_tokenizer.eos_token_id,
            max_length=500)
        
        print(sequences[0]['generated_text'])
        # for seq in sequences:
        #     print(f"Result: {seq['generated_text']}")
        pose_trace = load_pose_trace(instruction_id)

        time_stamps = get_end_time_stamps_from_instructions(sequences[0]['generated_text'], data['episodes'][episode_id]['instruction'])
        
        import ipdb; ipdb.set_trace()

