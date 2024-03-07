import requests
from PIL import Image
import openai
from transformers import AutoProcessor, Blip2ForConditionalGeneration, AutoTokenizer
import torch
import transformers
import imageio
import time

openai.api_key = os.environ['OPENAI_API_KEY']

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

if __name__== "__main__":
    text_summary_model = 'gpt'
    blip2_processor, blip2_model = get_blip2_model()

    if 'llama' in text_summary_model:
        llama2_tokenizer, llama2_pipeline = get_llama2_model()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_path = '/home/saumyas/Projects/VLN-CE-Plan/tests/media/full_tasks/episode=1-ckpt=0-spl=0.97.mp4'
    vid = imageio.get_reader(video_path, 'ffmpeg')
    prompt = "Question: You are in an indoor environment. What do you see? Give only high level details. Answer:"
    video_description = [" ' "]
    video_down_sample = 3
    for i, im in enumerate(vid):
        if i % video_down_sample == 0:
            image = Image.fromarray(im).convert('RGB')
            inputs = blip2_processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
            generated_ids = blip2_model.generate(**inputs, max_new_tokens=40)
            generated_text = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            print(generated_text)
            video_description.append(generated_text)

    video_description.append(" ' ")
    summary_prompt = 'A person made the following observations one after the other in a sequence while navigating a house. Summarize what the person has seen so far: \n '
    video_description = summary_prompt + " ".join(video_description)
    
    if 'llama' in text_summary_model:
        sequences = llama2_pipeline(video_description,
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=llama2_tokenizer.eos_token_id,
            max_length=1000,
        )
        summary = sequences[0]['generated_text']
    elif 'gpt' in text_summary_model:
        summary = gpt_response(create_messages_from_instruction(" ".join(video_description)))
        summary = summary['choices'][0]['message']['content']

    high_level_instruction = "Go to the bathtub in the bathroom"
    next_instruction = gpt_response(create_instruction_from_summary(summary, high_level_instruction))
    next_instruction = next_instruction['choices'][0]['message']['content']

    print("Summary:", summary)
    print("Next instruction:", next_instruction)