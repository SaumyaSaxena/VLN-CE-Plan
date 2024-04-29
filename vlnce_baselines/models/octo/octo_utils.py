from typing import Dict, Any 
import msgpack
import enum 
import numpy as np
import json

import tensorflow as tf
import huggingface_hub

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

def get_octo_data(checkpoint_path):
	checkpoint_path = huggingface_hub.snapshot_download(checkpoint_path.removeprefix("hf://"))
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

	return example_batch, dataset_statistics