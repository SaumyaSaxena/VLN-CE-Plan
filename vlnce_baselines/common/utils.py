from typing import Any, Dict, List, Mapping, Tuple, Optional

from gym import spaces

import numpy as np

import torch
import tensorflow as tf
import tensorflow_hub as hub

from heapq import heappush, heappop

from timm.scheduler.scheduler_factory import create_scheduler

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

    safety_settings = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]

    gemini_model = genai.GenerativeModel('gemini-pro-vision', safety_settings=safety_settings)
    ## For some reason throws error without doing below
    
    # url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' 
    # img = Image.open(requests.get(url, stream=True).raw).convert('RGB')   
    # response = gemini_model.generate_content(["Write a short, engaging blog post based on this picture.", img], stream=True)
    # response.resolve()

    return gemini_model

def get_gemini_text_model():
    from PIL import Image
    import os, requests
    import google.generativeai as genai
    GOOGLE_API_KEY=os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)

    return genai.GenerativeModel('gemini-pro')

def get_blip2_model(device):
    from transformers import AutoProcessor, Blip2ForConditionalGeneration
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # by default `from_pretrained` loads the weights in float32
    # we load in float16 instead to save memory
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
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

def create_optimizer_with_params(config, params):
    optimizer = None
    if config['name'] == 'Adam':
        adam_params = config['Adam']
        optimizer = torch.optim.Adam(params, lr=adam_params['lr'], eps=adam_params['eps'])
    elif config['name'] == 'AdamW':
        adamw_params = config[config['name']]
        optimizer = torch.optim.AdamW(params, lr=adamw_params['lr'], 
                                      eps=adamw_params['eps'],
                                      weight_decay=adamw_params['weight_decay'])
    else:
        raise ValueError(f"Invalid optimizer: {config['name']}")
    return optimizer

def create_schedulder_with_params(config: Mapping[str, Any], optimizer, batches) -> Tuple[Optional[Any], Mapping]:
    scheduler = None
    extra_dict = {}

    # Get scheduler params config
    scheduler_config = config[config['name']]
    if scheduler_config['use_timm']: 
        # Use timm scheduler
        
        if 'epoch' in config['update_per']:
            # Manually set the epochs correctly
            total_epochs = scheduler_config['epochs']
            cooldown_epochs = scheduler_config.get('cooldown_epochs', 0)
            scheduler_config['epochs'] = total_epochs - cooldown_epochs
            extra_dict['t_in_epochs'] = True
        else: # update lr per step and cycle
            total_epochs = batches
            cooldown_steps = int(batches*config['cooldown_ratio'])
            scheduler_config['epochs'] = batches - cooldown_steps
            scheduler_config['warmup_epochs'] = int(batches*config['warmup_ratio'])
            scheduler_config['cooldown_epochs'] = cooldown_steps
            extra_dict['t_in_epochs'] = False

        scheduler, num_epochs = create_scheduler(scheduler_config, optimizer)

        assert num_epochs == total_epochs, (
            f"timm scheduler epochs {num_epochs} and total epochs {total_epochs} do not match.")

        extra_dict['num_epochs'] = num_epochs
        extra_dict['timm_scheduler'] = True

    elif config['name'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_config['t_max'], 
            eta_min=scheduler_config['eta_min'], 
            last_epoch=scheduler_config['last_epoch'])
        extra_dict['t_in_epochs'] = False
        extra_dict['timm_scheduler'] = False

    else:
        raise ValueError(f"Invalid optim schedulder: {config['name']}")
    
    return scheduler, extra_dict

def int2bits(x, n, out_dtype=None):
    """Convert an integer x in (...) into bits in (..., n)."""

    if isinstance(x, torch.Tensor):
        x = torch.bitwise_right_shift(x.unsqueeze(-1), torch.arange(n))
        x = x % 2
        if out_dtype and out_dtype != x.dtype:
            x = x.type(out_dtype)
    else:
        x = x.astype(np.uint8)
        x = np.unpackbits(x[..., np.newaxis], axis=-1, count=n-8, bitorder='little').astype(int) # this way it behaves like torch implementation
    return x

def bits2int(x, out_dtype=torch.int32):
    """Converts bits x in (..., n) into an integer in (...)."""
    if isinstance(x, torch.Tensor):
        device = x.device
        x = x.type(out_dtype)
        x = torch.sum(x * (2 ** torch.arange(x.shape[-1]).to(device)), -1)
    else:
        x = np.packbits(x, axis=-1, bitorder='little')
    return x


class TopKLogger:
    def __init__(self, k: int):
        self.max_to_keep = k
        self.checkpoint_queue = []
    
    def push(self, ckpt: str, success: float):
        # NOTE: We have a min heap
        if len(self.checkpoint_queue) < self.max_to_keep:
            heappush(self.checkpoint_queue, (success, ckpt))
            return True
        else:
            curr_min_success, _ = self.checkpoint_queue[0]
            if curr_min_success < success:
                heappop(self.checkpoint_queue)
                heappush(self.checkpoint_queue, (success, ckpt))
                return True
            else:
                return False
            
if __name__ == "__main__":
    # aa = torch.tensor([0, 1, 2, 3, 4, 5])
    aa = np.array([0, 1, 2, 3, 4, 5])
    bits = int2bits(aa, 4)
    ints = bits2int(bits, torch.int32)
    import ipdb; ipdb.set_trace()
