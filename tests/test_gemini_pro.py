import os, requests
from PIL import Image 

import google.generativeai as genai
GOOGLE_API_KEY=os.environ['GOOGLE_API_KEY']
genai.configure(api_key=GOOGLE_API_KEY)


for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

# model = genai.GenerativeModel('gemini-pro')
# response = model.generate_content("What is the meaning of life?")

model = genai.GenerativeModel('gemini-pro-vision')

# url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' 
# img = Image.open(requests.get(url, stream=True).raw).convert('RGB')
# img = Image.open(r"https://t0.gstatic.com/licensed-image?q=tbn:ANd9GcQ_Kevbk21QBRy-PgB4kQpS79brbmmEG7m3VOTShAn4PecDU5H5UxrJxE3Dw1JiaG17V88QIol19-3TM2wCHw") 

img = Image.open("/home/saumyas/Projects/VLN-CE-Plan/tests/media/water_on_toaster.png")
response = model.generate_content(["Is this safe?", img])
response.resolve()
print(response.text)

# video_path = "/home/saumyas/Projects/VLN-CE-Plan/scripts/media/train/subtasks/576/episode=10759-ckpt=0-spl=1.00-orig_ep_idx=576.00.mp4"
# import imageio
# vid = imageio.get_reader(video_path, 'ffmpeg')
# vlm_prompt = "Question: You are in an indoor environment. What do you see? Give only high level details. Answer:"
# for i, im in enumerate(vid):
#   image = Image.fromarray(im).convert('RGB')
#   generated_text = model.generate_content([vlm_prompt, image])
#   generated_text.resolve()
#   print(generated_text.text)
#   import ipdb; ipdb.set_trace()