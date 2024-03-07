import gzip, json, os, sys, requests
import numpy as np
from PIL import Image
import imageio
from pathlib import Path

from transformers import AutoProcessor, Blip2ForConditionalGeneration, AutoTokenizer
import torch
import openai
openai.api_key = os.environ['OPENAI_API_KEY']

import google.generativeai as genai
GOOGLE_API_KEY=os.environ['GOOGLE_API_KEY']
genai.configure(api_key=GOOGLE_API_KEY)

cwd = os.getcwd()
sys.path.append(f'{cwd}/..')

next_instruction_prompt = "You are a robot navigating a house. You get a high level instruction of where to go and a summary of what you have seen in the house so far. The image shows where you are right now. To which visible landmark should you move next to get nearer to high level goal."
summary_prompt = 'A person made the following consecutive observations while navigating a house. Can you describe the layout of the house seen so far based on the observations very concisely: \n '
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
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": f"Observations: '{instruction_text}'"}
        ]
    return messages

def create_instruction_from_summary(summary, high_level_instruction):
    messages=[
            {"role": "system", "content": "You are a robot navigating a house you have never been in before. You get a high level instruction of where to go and a summary of what you have seen in the house so far. Concisely, output based on where you right now where you should move next to get nearer to high level goal:"},
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

    ## DATA
    data_type = 'train'
    role = 'guide'
    subtasks = True
    data_file = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_subtasks_rollouts_merged_hli.json.gz'
    data = {}
    with gzip.open(data_file,"rt",) as f:
        data.update(json.load(f))
    
    ## Pretrained models
    vl_model_type = 'gemini-pro' # 'gemini-pro', 'llava'
    if vl_model_type == 'blip2':
        blip2_processor, blip2_model = get_blip2_model()
    elif vl_model_type == 'gemini-pro':
        gemini_model = genai.GenerativeModel('gemini-pro-vision')
        ## For some reason throws error without doing below
        url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' 
        img = Image.open(requests.get(url, stream=True).raw).convert('RGB')   
        response = gemini_model.generate_content(["Write a short, engaging blog post based on this picture.", img], stream=True)
        response.resolve()
    else:
        raise NotImplementedError(f"VL model type {vl_model_type} not available.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    video_dir = f'{cwd}/../scripts/media/{data_type}/subtasks/'
    orig_ep_idxs = os.listdir(video_dir)

    vlm_prompt = "Question: You are in an indoor environment. What do you see? Give only high level details. Answer:"

    for i in range(len(orig_ep_idxs)):
        orig_ep_idx = orig_ep_idxs[i]
        print("CONTEXT GENERATION FOR ORIGINAL EPISODE INDEX:", orig_ep_idx)
        videos = os.listdir(video_dir+orig_ep_idx)
        
        previous_episode_video_description = [" ' "]
        ep_idxs = [int((v.split('episode=')[1]).split('-ckpt')[0]) for v in videos if '.mp4' in v]
        ep_idxs = np.sort(ep_idxs)

        output_txt_name = video_dir+orig_ep_idx+'/summaries_'+f'{vl_model_type}'+'.txt'
        if Path(output_txt_name).exists():
            print(f"Summary of {orig_ep_idx} exists.")
            continue

        output_txt = open(output_txt_name, 'w')
        
        high_level_instruction = data['episodes'][int(ep_idxs[0])]['instruction']['high_level_instruction']
        output_txt.write("High level instruction:\n"+high_level_instruction+"\n\n\n")
        for ep_idx in ep_idxs:
            video_name = [v for v in videos if str(ep_idx) in v]
            video_path = video_dir + orig_ep_idx + '/' + video_name[0]
            vid = imageio.get_reader(video_path, 'ffmpeg')
            
            video_description = []
            video_description.append(previous_episode_video_description[0])
            video_down_sample = 3

            # per time step image description of episode
            for i, im in enumerate(vid):
                if i % video_down_sample == 0:
                    image = Image.fromarray(im).convert('RGB')
                    if vl_model_type == 'blip2':
                        inputs = blip2_processor(image, text=vlm_prompt, return_tensors="pt").to(device, torch.float16)
                        generated_ids = blip2_model.generate(**inputs, max_new_tokens=40)
                        generated_text = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                        video_description.append(generated_text)
                    elif vl_model_type == 'gemini-pro':
                        generated_text = gemini_model.generate_content([vlm_prompt, image])
                        generated_text.resolve()
                        video_description.append(generated_text.text)
                    
            # Summarize episode
            previous_episode_video_description = video_description.copy()
            video_description.append(" ' ")
            
            video_description_input = " ".join(video_description)
            summary = gpt_response(create_messages_from_instruction(" ".join(video_description_input)))
            summary = summary['choices'][0]['message']['content']
            
            # LLM planner next step
            # next_instruction = gpt_response(create_instruction_from_summary(summary, high_level_instruction))
            # next_instruction = next_instruction['choices'][0]['message']['content']
            next_instruction_prompt = f"You are a robot navigating a house. You get a high level instruction ({high_level_instruction}) of where to go and a summary ({summary}) of what you have seen in the house so far. The image shows where you are right now. To which visible landmark should you move next to get nearer to high level goal."
            if vl_model_type == 'blip2':
                inputs = blip2_processor(image, text=next_instruction_prompt, return_tensors="pt").to(device, torch.float16)
                generated_ids = blip2_model.generate(**inputs, max_new_tokens=40)
                next_instruction = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            elif vl_model_type == 'gemini-pro':
                next_instruction = gemini_model.generate_content([next_instruction_prompt, image])
                next_instruction.resolve()
                next_instruction = next_instruction.text
            output_txt.write("Summary:\n"+summary+"\n")
            output_txt.write("Next instruction:\n"+next_instruction+"\n")
            output_txt.write("GT current instruction:\n"+data['episodes'][int(ep_idx)]['instruction']['instruction_text']+"\n\n\n")

        output_txt.close()