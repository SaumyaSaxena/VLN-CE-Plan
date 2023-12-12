import requests
from PIL import Image
import openai
from transformers import AutoProcessor, Blip2ForConditionalGeneration, AutoTokenizer
import torch
import transformers
import imageio
import time
import gzip, json, os

openai.api_key = "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"

def gpt_response(messages):
    done = False
    while not done:
        # try:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        done=True
        # except:
            # time.sleep(0.1)
    return response

def create_messages_from_instruction(instruction_text):
    messages=[
            {"role": "system", "content": "You are given sequential observations that a robot makes while navigating a house. Summarize these observations:"},
            {"role": "user", "content": f"Observations: '{instruction_text}'"}
        ]
    return messages

def create_instruction_from_summary(summary, high_level_instruction):
    messages=[
            {"role": "system", "content": "You are a robot navigating a house you have never been in before. You get a high level instruction of where to go and a summary of what you have seen in the house so far. Concisely, output the next step you should take to complete the high level instruction:"},
            {"role": "user", "content": f"Given a high level instruction: '{high_level_instruction}'. Summary of observations so far: '{summary}' "}
        ]
    return messages

def get_blip2_model():
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # by default `from_pretrained` loads the weights in float32
    # we load in float16 instead to save memory
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model

if __name__== "__main__":
    data_type = 'train'
    role = 'guide'
    subtasks = True

    data_file = f'/home/saumyas/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_subtasks_rollouts_merged.json.gz'
    data = {}
    with gzip.open(data_file,"rt",) as f:
        data.update(json.load(f))
    
    blip2_processor, blip2_model = get_blip2_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    episodes = range(0,6)
    high_level_instruction = 'Go to the sink in the bathroom'

    video_dir = '/home/saumyas/Projects/VLN-CE-Plan/tests/media/subtasks_rollout_merged/'
    vlm_prompt = "Question: You are in an indoor environment. What do you see? Give only high level details. Answer:"
    summary_prompt = 'A person made the following observations one after the other in a sequence while navigating a house. Summarize what the person has seen so far: \n '

    videos = os.listdir(video_dir)
    previous_episode_video_description = [" ' "]
    for ep in episodes:
        video_prefix = f'episode={str(ep)}'
        video_name = [v for v in videos if video_prefix in v]
        video_path = video_dir + video_name[0]

        vid = imageio.get_reader(video_path, 'ffmpeg')
        
        video_description = []
        video_description.append(previous_episode_video_description[0])
        video_down_sample = 3

        # per time step image description of episode
        for i, im in enumerate(vid):
            if i % video_down_sample == 0:
                image = Image.fromarray(im).convert('RGB')
                inputs = blip2_processor(image, text=vlm_prompt, return_tensors="pt").to(device, torch.float16)
                generated_ids = blip2_model.generate(**inputs, max_new_tokens=40)
                generated_text = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                # print(generated_text)
                video_description.append(generated_text)
        
        # Summarize episode
        previous_episode_video_description = video_description.copy()
        video_description.append(" ' ")
        
        video_description_input = summary_prompt + " ".join(video_description)
        summary = gpt_response(create_messages_from_instruction(" ".join(video_description_input)))
        summary = summary['choices'][0]['message']['content']
        print("Summary:", summary)

        # LLM planner next step
        next_instruction = gpt_response(create_instruction_from_summary(summary, high_level_instruction))
        next_instruction = next_instruction['choices'][0]['message']['content']

        print("Next LLM instruction:", next_instruction)
        print("GT instruction", data['episodes'][ep]['instruction']['instruction_text'])
        import ipdb; ipdb.set_trace()

