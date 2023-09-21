import requests
from PIL import Image

from transformers import AutoProcessor, Blip2ForConditionalGeneration, AutoTokenizer
import torch
import transformers
import imageio

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


def get_image():
    url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' 
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')   
    return image

def get_video():
    video_path = '/home/sax1rng/Projects/VLN-CE-Plan/outputs/rxr_cma_en/pretrained/evals/videos_ckpt0/episode=4639-ckpt=0-spl=1.00.mp4'
    vid = imageio.get_reader(video_path,  'ffmpeg')
    
    for i, im in enumerate(vid):
        import ipdb; ipdb.set_trace()
        img_pil = Image.fromarray(im).convert('RGB')
        print('Mean of frame %i is %1.1f' % (i, im.mean()))


if __name__== "__main__":
    # get_video()
    blip2_processor, blip2_model = get_blip2_model()
    llama2_tokenizer, llama2_pipeline = get_llama2_model()
    image = get_image()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_path = '/home/sax1rng/Projects/VLN-CE-Plan/outputs/rxr_cma_en/pretrained/evals/videos_ckpt0/episode=4639-ckpt=0-spl=1.00.mp4'
    vid = imageio.get_reader(video_path, 'ffmpeg')
    prompt = "Question: You are in an indoor environment. What do you see? Answer:"
    video_description = [" ' "]
    for i, im in enumerate(vid):
        image = Image.fromarray(im).convert('RGB')
        inputs = blip2_processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
        generated_ids = blip2_model.generate(**inputs, max_new_tokens=40)
        generated_text = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(generated_text)
        video_description.append(generated_text)

    video_description.append(" ' ")
    summary_prompt = 'A person made the following observations one after the other in a sequence while navigating a house. Summarize what the person has seen so far: \n '
    video_description = summary_prompt + " ".join(video_description)
    
    sequences = llama2_pipeline(video_description,
        do_sample=True,
        top_k=1,
        num_return_sequences=1,
        eos_token_id=llama2_tokenizer.eos_token_id,
        max_length=1000,
    )

    print("------SUMMARY------")
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")