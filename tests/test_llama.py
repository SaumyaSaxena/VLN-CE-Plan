from transformers import AutoTokenizer
import transformers
import torch, gzip, json
import numpy as np

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

data_type = 'train'
data_location = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_guide.json.gz'
data = {}
with gzip.open(data_location,"rt",) as f:
    data.update(json.load(f))

dataset_size = len(data['episodes'])
en_idx = [i for i in range(dataset_size) if ('en' in data['episodes'][i]['instruction']['language'])]


while True:
    episode_id = np.random.choice(en_idx)
    instruction_text = data['episodes'][episode_id]['instruction']['instruction_text']
    instruction_id = data['episodes'][episode_id]['instruction']['instruction_id']

    print(f"Original instruction: {instruction}")
    sequences = pipeline(
        f'Convert the following instructions to numbered step by step instructions (e.g. Step 1: ) using same text as in the instruction: "{instruction_text}"',
        do_sample=True,
        top_k=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=500)
    
    print(sequences[0]['generated_text'])
    # for seq in sequences:
    #     print(f"Result: {seq['generated_text']}")
    import ipdb; ipdb.set_trace()

