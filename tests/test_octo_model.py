import sys, os
sys.path.append('/home/saumyas/Projects/VLN-CE-Plan')

import huggingface_hub
import tensorflow as tf
import json
import numpy as np
import msgpack
import enum 
from typing import Dict, Any 

import torch

import requests
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf

from vlnce_baselines.models.octo_policy import OctoPolicy

_dict_to_tuple = lambda dct: tuple(dct[str(i)] for i in range(len(dct)))

def _unchunk(data: Dict[str, Any]):
  """Convert canonical dictionary of chunked arrays back into array."""
  assert '__msgpack_chunked_array__' in data
  shape = _dict_to_tuple(data['shape'])
  flatarr = np.concatenate(_dict_to_tuple(data['chunks']))
  return flatarr.reshape(shape)

def _unchunk_array_leaves_in_place(d):
  """Convert chunked array leaves back into array leaves, in place."""
  if isinstance(d, dict):
    if '__msgpack_chunked_array__' in d:
      return _unchunk(d)
    else:
      for k, v in d.items():
        if isinstance(v, dict) and '__msgpack_chunked_array__' in v:
          d[k] = _unchunk(v)
        elif isinstance(v, dict):
          _unchunk_array_leaves_in_place(v)
  return d

class _MsgpackExtType(enum.IntEnum):
  """Messagepack custom type ids."""

  ndarray = 1
  native_complex = 2
  npscalar = 3

def _ndarray_from_bytes(data: bytes) -> np.ndarray:
  """Load ndarray from simple msgpack encoding."""
  shape, dtype_name, buffer = msgpack.unpackb(data, raw=True)
  return np.frombuffer(
    buffer, dtype=_dtype_from_name(dtype_name), count=-1, offset=0
  ).reshape(shape, order='C')

def _dtype_from_name(name: str):
  """Handle JAX bfloat16 dtype correctly."""
  if name == b'bfloat16':
    return jax.numpy.bfloat16
  else:
    return np.dtype(name)

def _msgpack_ext_unpack(code, data):
  """Messagepack decoders for custom types."""
  if code == _MsgpackExtType.ndarray:
    return _ndarray_from_bytes(data)
  elif code == _MsgpackExtType.native_complex:
    complex_tuple = msgpack.unpackb(data)
    return complex(complex_tuple[0], complex_tuple[1])
  elif code == _MsgpackExtType.npscalar:
    ar = _ndarray_from_bytes(data)
    return ar[()]  # unpack ndarray to scalar
  return msgpack.ExtType(code, data)

def load_pretrained(checkpoint_path):
    checkpoint_path = huggingface_hub.snapshot_download(checkpoint_path.removeprefix("hf://"))

    with tf.io.gfile.GFile(
            tf.io.gfile.join(checkpoint_path, "config.json"), "r"
        ) as f:
            config_octo = json.load(f)
    # load example batch
    with tf.io.gfile.GFile(
        tf.io.gfile.join(checkpoint_path, "example_batch.msgpack"), "rb"
    ) as f:
        example_batch = msgpack.unpackb(
            f.read(), ext_hook=_msgpack_ext_unpack, raw=False
        )

    # load dataset statistics
    with tf.io.gfile.GFile(
        tf.io.gfile.join(checkpoint_path, "dataset_statistics.json"), "r"
    ) as f:
        dataset_statistics = json.load(f)

    # import yaml
    # with open('octo_config.yaml', 'w') as outfile:
    #   yaml.dump(config_octo, outfile, default_flow_style=False)
    # with open('/home/saumyas/Projects/VLN-CE-Plan/tests/octo_config2.yml', 'r') as file:
    #   config = yaml.safe_load(file)
    config = OmegaConf.load('/home/saumyas/Projects/VLN-CE-Plan/vlnce_baselines/config/rxr_baselines/octo_config.yaml')
    model = OctoPolicy.from_config(config, example_batch, dataset_statistics)
    
    IMAGE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
    img = np.array(Image.open(requests.get(IMAGE_URL, stream=True).raw).resize((256, 256)))
    # plt.imshow(img)

    # create obs & task dict, run inference
    # add batch + time horizon 2
    img = img[np.newaxis,...]
    img = np.stack([img]*2, axis=1) # horizon=2
    observation = {"image_primary": torch.Tensor(img), "pad_mask": torch.Tensor([[True, True]])}
    task = model.create_tasks(texts=["pick up the fork"])
    
    action = model.sample_actions(observation, task)
    return action

if __name__ == "__main__":
    action = load_pretrained("hf://rail-berkeley/octo-small")
    import ipdb; ipdb.set_trace()
