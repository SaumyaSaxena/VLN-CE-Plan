from PIL import Image
import requests
# from transformers import pipeline

# pipe = pipeline("text-generation", model="liuhaotian/llava-v1.6-vicuna-7b")

# Load model directly
from transformers import AutoProcessor, AutoModelForCausalLM

processor = AutoProcessor.from_pretrained("liuhaotian/llava-v1.6-34b")
model = AutoModelForCausalLM.from_pretrained("liuhaotian/llava-v1.6-34b")


prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs, max_length=30)
generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(generated_text)