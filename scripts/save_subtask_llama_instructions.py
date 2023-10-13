import sys, os
sys.path.append('/home/sax1rng/Projects/VLN-CE-Plan')

from transformers import AutoTokenizer
import transformers
import torch, gzip, json
import time
from tqdm import tqdm

def load_data(data_type='train', role='guide'):
    data_location = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}.json.gz'
    data = {}
    with gzip.open(data_location,"rt",) as f:
        data.update(json.load(f))
    return data

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

def create_subtask_prompt(instruction):
    instruction = instruction.replace('\n','')
    prompt = f'Split/divide/chunk the following instructions to numbered step by step instructions (e.g. Step 1: ). Use all the words as in instruction. Add no new words. Do not remove words from the given instruction: "{instruction}"'
    return prompt

def get_llama_sequence(instruction, llama2_tokenizer, llama2_pipeline):
    found_good_sequence = False
    trial=0
    while not found_good_sequence:
        trial += 1
        sequences = llama2_pipeline(
            create_subtask_prompt(instruction),
            do_sample=True,
            top_k=20,
            num_return_sequences=1,
            eos_token_id=llama2_tokenizer.eos_token_id,
            max_length=1000)

        if trial > 20:
            import ipdb; ipdb.set_trace()
            sequences[0]['generated_text'] = instruction.replace('. ', ' \nstep 2: ')
        if 'step 2: ' in (sequences[0]['generated_text']).lower():
            found_good_sequence = True

    return sequences[0]['generated_text']

if __name__== "__main__":
    save_results = True
    data_type = 'val_unseen'
    role = 'follower'

    llama2_tokenizer, llama2_pipeline = get_llama2_model()

    data = load_data(data_type=data_type, role=role)
    dataset_size = len(data['episodes'])
    en_idx = [i for i in range(dataset_size) if ('en' in data['episodes'][i]['instruction']['language'])]
    n_episodes = len(en_idx)

    batch_size = 100
    approx_total_batches = n_episodes/batch_size
    n_batch=0
    done = False
    while not done:
        start = time.time()
        output_file = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/subtask_instructions/{data_type}_{role}_subtask_instructions_{n_batch}.json.gz'
        
        if os.path.exists(output_file):
            print(f"Skipped batch {n_batch} since path exists")
            n_batch += 1
            continue

        print(f'batch:{n_batch}/{n_episodes/batch_size}')
        batch_end = min(n_episodes,(n_batch+1)*batch_size)
        
        batch_idx = en_idx[n_batch*batch_size:batch_end]

        llama_sequences = {}
        for i in tqdm(batch_idx):
            llama_sequences[i] = get_llama_sequence(data['episodes'][i]['instruction']['instruction_text'], llama2_tokenizer, llama2_pipeline)
        
        end = time.time()
        time_elapsed = end - start
        print(f'Time for batch {n_batch}: {time_elapsed}')
        print(f'Approx total time: {time_elapsed*approx_total_batches/3600} hours')

        if save_results:
            print("Saving results")
            with gzip.open(output_file, "wt") as f:
                f.write(json.dumps(llama_sequences))
            print("Output file saved at ", output_file)

        if batch_end == n_episodes:
            done=True
            print("DONE SAVING ALL SUBTASK INSTRUCTIONS!!")
            
        n_batch += 1