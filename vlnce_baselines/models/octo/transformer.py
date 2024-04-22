# adapted from https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
from typing import Callable, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vlnce_baselines.models.octo.base import TokenGroup

class AddPositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs."""

    def __init__(self, window_size, dim):
        super().__init__()
        self.embed = nn.Embedding(window_size, dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight)

    def forward(self, inputs):
        """Applies the AddPositionEmbs module.

        Args:
          inputs: Inputs to the layer.

        Returns:
          Output tensor with shape `(bs, timesteps, in_dim)`.
        """
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, (
            "Number of dimensions should be 3," " but it is: %d" % inputs.ndim
        )
        return inputs + self.embed


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    def __init__(self,
        mlp_dim: int,
        out_dim: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.dense_layer1 = nn.Linear(
            in_features=out_dim,
            out_features=mlp_dim
        )
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense_layer2 = nn.Linear(
            in_features=mlp_dim,
            out_features=out_dim
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.dense_layer1.weight)
        nn.init.normal_(self.dense_layer1.bias)

        nn.init.xavier_uniform_(self.dense_layer2.weight)
        nn.init.normal_(self.dense_layer2.bias)
    

    def forward(self, inputs):
        """Applies Transformer MlpBlock module."""
        x = self.dense_layer1(inputs)
        x = F.gelu(x)
        x = self.dropout1(x)
        
        output = self.dense_layer2(x)
        output = self.dropout2(output)
        return output


class MAPHead(nn.Module):
    """Multihead Attention Pooling.

    From https://github.com/google-research/big_vision/blob/main/big_vision/models/vit.py
    """
    def __init__(self,
        mlp_dim: Optional[int] = None,  # Defaults to 4x input dim
        num_heads: int = 8,
        num_readouts: int = 1,
    ):
        super().__init__()
        self.num_readouts = num_readouts
        self.probe = nn.Embedding(1, num_readouts, dim) # TODO(saumya)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=mlp_dim,
            num_heads=num_heads
        )
        self.layer_norm = nn.LayerNorm(mlp_dim)
        self.mlp_block = MlpBlock(mlp_dim=mlp_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.probe.weight)


    def forward(self, x: Union[torch.Tensor, TokenGroup], train=True):
        if isinstance(x, TokenGroup):
            x, mask = x.tokens, x.mask
        else:
            mask = None

        *batch_dims, l, d = x.shape
        x = x.reshape(-1, l, d)
        batch_size = x.shape[0]

        self.probe = torch.tile(self.probe, [batch_size, 1, 1])

        if mask is not None:
            mask = mask.reshape(-1, l)
            mask = torch.broadcast_to(
                mask[:, None, None, :], (batch_size, 1, self.num_readouts, l)
            )

        out = self.multihead_attention(self.probe, x, x, mask=mask)

        # TODO: dropout on head?
        y = self.layer_norm(out)

        out = out + self.mlp_block(y)
        out = out.reshape(*batch_dims, self.num_readouts, d)
        return out


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
    """
    def __init__(self,
        token_embedding_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.mlp_dim = mlp_dim

        self.layer_norm1 = nn.LayerNorm(token_embedding_size)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=token_embedding_size,
            num_heads=num_heads,
            dropout=attention_dropout_rate,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.layer_norm2 = nn.LayerNorm(token_embedding_size)
        self.mlp_block = MlpBlock(mlp_dim=mlp_dim, out_dim=token_embedding_size, dropout_rate=dropout_rate)
        

    def forward(self, inputs, attention_mask):
        """Applies Encoder1DBlock module.

        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.

        Returns:
          output after transformer encoder block.
        """

        # Attention block.
        assert inputs.ndim == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"
        x = self.layer_norm1(inputs)
        x, attn_weights = self.multihead_attention(x, x, x, attn_mask=attention_mask)
        x = self.dropout(x)
        x = x + inputs

        # MLP block.
        y = self.layer_norm2(x)
        y = self.mlp_block(y)

        return x + y


class Transformer(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
      num_layers: number of layers
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate in self attention.
    """
    def __init__(
        self,
        num_layers: int,
        mlp_dim: int,
        num_attention_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        add_position_embedding: bool = False,
        window_size: int = 2,
        token_embedding_size: int = 384,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.add_position_embedding = add_position_embedding

        self.position_emb = AddPositionEmbs(window_size=window_size, dim=mlp_dim)
        self.pos_emb_dropout = nn.Dropout(dropout_rate)

        self.encoder_blocks = []
        for _ in range(self.num_layers):
            self.encoder_blocks.append(
                Encoder1DBlock(
                    token_embedding_size=token_embedding_size,
                    mlp_dim=mlp_dim,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    num_heads=num_attention_heads,
                )
            )
        
        self.layer_norm = nn.LayerNorm(token_embedding_size)

    def forward(self, x, attention_mask, *, train):
        """Applies Transformer model on the inputs.

        Args:
          x: Inputs to the layer.
          train: Set to `True` when training.

        Returns:
          output of a transformer encoder.
        """
        assert x.ndim == 3  # (batch, len, emb)
        if self.add_position_embedding:
            x = self.position_emb(x)
            x = self.pos_emb_dropout(x)

        # Input Encoder
        for lyr in range(self.num_layers):
            x = self.encoder_blocks[lyr](x, attention_mask)
        
        encoded = self.layer_norm(x)
        return encoded


def common_transformer_sizes(transformer_size: str) -> (int, dict):
    """
    Args:
        transformer_size (str): The size of the transformer. One of "dummy", "vanilla", "vit_s", "vit_b", "vit_l", "vit_h"

    Returns:
            token_embedding_size (int): The size of the token embeddings
            transformer_kwargs (dict): The kwargs to pass to the transformer

    """
    assert transformer_size in ["dummy", "vanilla", "vit_s", "vit_b", "vit_l", "vit_h"]
    default_params = {
        "attention_dropout_rate": 0.0,
        "add_position_embedding": False,
    }

    TRANSFORMER_SIZES = {
        "dummy": dict(
            num_layers=1,
            mlp_dim=256,
            num_attention_heads=2,
            dropout_rate=0.1,
        ),
        "vanilla": dict(
            num_layers=4,
            mlp_dim=1024,
            num_attention_heads=8,
            dropout_rate=0.1,
        ),
        "vit_s": dict(
            num_layers=12,
            mlp_dim=1536,
            num_attention_heads=6,
            dropout_rate=0.0,
        ),
        "vit_b": dict(
            num_layers=12,
            mlp_dim=3072,
            num_attention_heads=12,
            dropout_rate=0.0,
        ),
        "vit_l": dict(
            num_layers=24,
            mlp_dim=4096,
            num_attention_heads=16,
            dropout_rate=0.1,
        ),
        "vit_h": dict(
            num_layers=32,
            mlp_dim=5120,
            num_attention_heads=16,
            dropout_rate=0.1,
        ),
    }

    TOKEN_DIMS = {
        "dummy": 256,
        "vanilla": 256,
        "vit_s": 384,
        "vit_b": 768,
        "vit_l": 1024,
        "vit_h": 1280,
    }

    return TOKEN_DIMS[transformer_size], {
        **default_params,
        **TRANSFORMER_SIZES[transformer_size],
    }
