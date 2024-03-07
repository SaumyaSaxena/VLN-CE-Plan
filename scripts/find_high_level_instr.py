import openai
import json, gzip, time, os
from tqdm import tqdm
# from multiprocessing import Process
# import multiprocessing

openai.api_key = os.environ['OPENAI_API_KEY']

def gpt_response(messages):
    done = False
    while not done:
        try:
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
            done=True
        except:
            time.sleep(0.1)
    return response

def update_dict_with_gpt_response(messages,return_dict,procnum):
    return_dict[procnum] = gpt_response(messages)

def load_data(data_type='train'):
    data_location = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_guide.json.gz'
    data = {}
    with gzip.open(data_location,"rt",) as f:
        data.update(json.load(f))
    return data

def create_messages_from_instruction(instruction_text):
    messages=[
            {"role": "system", "content": "You are trying to reach a certain goal given some instructions."},
            {"role": "user", "content": f"Instruction: '{instruction_text}' Give output in the following format: 'goto <goal object> which is near <nearest landmark> in <room>'"}
        ]
    return messages

if __name__== "__main__":
    # manager = multiprocessing.Manager()
    # return_dict = manager.dict()

    test_saved_file=False
    data_type = 'val_unseen'
    if test_saved_file:
        saved_file = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/high_level_instr/{data_type}_guide_high_level_instr_0.json.gz'
        saved_data = {}
        with gzip.open(saved_file,"rt",) as f:
            saved_data.update(json.load(f))
        import ipdb; ipdb.set_trace()

    
    data = load_data(data_type)
    dataset_size = len(data['episodes'])
    # new_data = data.copy()

    en_idx = [i for i in range(dataset_size) if ('en' in data['episodes'][i]['instruction']['language'])]
    messages_en = [create_messages_from_instruction(data['episodes'][i]['instruction']['instruction_text']) for i in en_idx]
    n_episodes = len(en_idx)
    
    # batch_size = 100
    # n_batch = 0
    # done = False
    # while not done:
    #     print(f'batch:{n_batch}/{n_episodes/batch_size}')
    #     batch_end = min(n_episodes,(n_batch+1)*batch_size)
    #     if batch_end == n_episodes:
    #         done=True
    #     batch_idx = en_idx[n_batch*batch_size:batch_end]
        
    #     # instantiating process with arguments
    #     print("Starting processes")

    #     procs = []
    #     for i, idx in enumerate(batch_idx):
    #         mess = create_messages_from_instruction(data['episodes'][idx]['instruction']['instruction_text'])
    #         proc = Process(target=update_dict_with_gpt_response, args=(mess,return_dict,idx))
    #         procs.append(proc)
    #         proc.start()

    #     # complete the processes
    #     for proc in procs:
    #         proc.join()

    #     print("Updating the new data dict")
    #     for i, idx in enumerate(batch_idx):
    #         new_data['episodes'][idx]['instruction']['high_level_instruction'] = return_dict[idx]['choices'][0]['message']['content']
        
    #     n_batch +=1

    # Without multiprocessing
    batch_size = 100 # save every 100 calls
    done = False
    n_batch=0
    while not done:
        print(f'batch:{n_batch}/{n_episodes/batch_size}')
        batch_end = min(n_episodes,(n_batch+1)*batch_size)

        batch_idx = en_idx[n_batch*batch_size:batch_end]

        output_file = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/high_level_instr/{data_type}_guide_high_level_instr_{n_batch}.json.gz'
        if os.path.exists(output_file):
            print(f"Skipped batch {n_batch} since path exists")
            n_batch += 1
            continue

        new_data = {}
        start = time.time()
        for i in tqdm(batch_idx):
            high_level_instr = gpt_response(create_messages_from_instruction(data['episodes'][i]['instruction']['instruction_text']))
            # new_data['episodes'][i]['instruction']['high_level_instruction'] = high_level_instr['choices'][0]['message']['content']
            new_data[i] = high_level_instr['choices'][0]['message']['content']
        end = time.time()
        print(f"time_elapsed for batch {n_batch} is {end - start}")
        save_results=True
        if save_results:
            print("Saving results")
            with gzip.open(output_file, "wt") as f:
                f.write(json.dumps(new_data))
            print("Output file saved at ", output_file)
            
        if batch_end == n_episodes:
            done=True
            print("DONE SAVING ALL HIGH LEVEL INSTRUCTIONS!!")
        n_batch += 1