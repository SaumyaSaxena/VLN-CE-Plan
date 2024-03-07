from typing import Any, Dict, List

from gym import spaces

import tensorflow as tf
import tensorflow_hub as hub

def extract_instruction_tokens(
    observations: List[Dict],
    instruction_sensor_uuid: str,
    tokens_uuid: str = "tokens",
) -> Dict[str, Any]:
    """Extracts instruction tokens from an instruction sensor if the tokens
    exist and are in a dict structure.
    """
    if (
        instruction_sensor_uuid not in observations[0]
        or instruction_sensor_uuid == "pointgoal_with_gps_compass"
    ):
        return observations
    for i in range(len(observations)):
        if (
            isinstance(observations[i][instruction_sensor_uuid], dict)
            and tokens_uuid in observations[i][instruction_sensor_uuid]
        ):
            observations[i][instruction_sensor_uuid] = observations[i][
                instruction_sensor_uuid
            ]["tokens"]
        else:
            break
    return observations


def single_frame_box_shape(box: spaces.Box) -> spaces.Box:
    """removes the frame stack dimension of a Box space shape if it exists."""
    if len(box.shape) < 4:
        return box

    return spaces.Box(
        low=box.low.min(),
        high=box.high.max(),
        shape=box.shape[1:],
        dtype=box.high.dtype,
    )

def get_gemini_vision_model():
    from PIL import Image
    import os, requests
    import google.generativeai as genai
    GOOGLE_API_KEY=os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)

    gemini_model = genai.GenerativeModel('gemini-pro-vision')
    ## For some reason throws error without doing below
    url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' 
    img = Image.open(requests.get(url, stream=True).raw).convert('RGB')   
    response = gemini_model.generate_content(["Write a short, engaging blog post based on this picture.", img], stream=True)
    response.resolve()

    return gemini_model

def get_gemini_text_model():
    from PIL import Image
    import os, requests
    import google.generativeai as genai
    GOOGLE_API_KEY=os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)

    return genai.GenerativeModel('gemini-pro')

def get_blip2_model():
    from transformers import AutoProcessor, Blip2ForConditionalGeneration
    import torch
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # by default `from_pretrained` loads the weights in float32
    # we load in float16 instead to save memory
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    return processor, model

def make_bert_preprocess_model(seq_length=128):
    input_segments = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.load(f'{cwd}/../data/bert_models/bert_multi_cased_preprocess_3/')

    tokenizer = hub.KerasLayer(preprocessor.tokenize, name='tokenizer')
    tokenized_input = [tokenizer(input_segments)]

    packer = hub.KerasLayer(preprocessor.bert_pack_inputs,
                        arguments=dict(seq_length=seq_length),
                        name='packer')
    model_inputs = packer(tokenized_input)
    return tf.keras.Model(input_segments, model_inputs)

def load_model():
    preprocess_model = make_bert_preprocess_model()
    encoder = hub.KerasLayer(f'{cwd}/../data/bert_models/bert_multi_cased_L-12_H-768_A-12_4/',trainable=True)
    return preprocess_model, encoder