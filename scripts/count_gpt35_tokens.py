import openai
import json
import tiktoken
import gzip
import time
from tqdm import trange

def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
    See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

if __name__== "__main__":

    data_type = 'val_unseen'
    data_locations = [f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_guide.json.gz']
    data = {}
    for data_file in data_locations:
        with gzip.open(data_file,"rt",) as f:
            data.update(json.load(f))
    
    dataset_size = len(data['episodes'])
    
    # data has len 60300, episode_id corresponds to keys in gt_data
    n_tokens = 0
    n_episodes = 0
    for i in trange(dataset_size):
        if 'en' in data['episodes'][i]['instruction']['language']:
            # start = time.time()
            instruction_text = data['episodes'][i]['instruction']['instruction_text']
            messages=[
                {"role": "system", "content": "You are trying to reach a certain goal given some instructions."},
                {"role": "user", "content": f"Instruction: '{instruction_text}' Give output in the following format: 'goto <goal> which is near <nearest landmark>'"}
            ]
            n_tokens += num_tokens_from_messages(messages, model="gpt-3.5-turbo")
            n_episodes += 1
            # end = time.time()
            # time_elapsed = end - start
    
    print(f"{data_type} dataset size:", len(data['episodes']))
    print(f"Number of tokens in {data_type}:", n_tokens)
    print(f"Number of english episodes in {data_type}:", n_episodes)