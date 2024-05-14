from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any

from einops import rearrange

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from vlnce_baselines.models.octo.base import TokenGroup
from vlnce_baselines.models.octo.diffusion import cosine_beta_schedule, create_diffusion_model
from vlnce_baselines.models.octo.tokenizers import BinTokenizer
from vlnce_baselines.models.octo.transformer import MAPHead
from vlnce_baselines.common.utils import bits2int, int2bits


def masked_mean(x, mask):
    mask = torch.broadcast_to(mask, x.shape)
    return torch.mean((x * mask).type(torch.float)) / torch.clip(torch.mean(mask.type(torch.float)), min=1e-5)


def chunk_actions(actions, pred_horizon):
    """Chunk actions for predicting actions `pred_horizon` steps into the future.

    The resulting actions have shape (batch, actions.shape[-2] - (pred_horizon - 1), pred_horizon, action_dim)

    For example: chunk_actions([a_1, a_2, a_3, a_4, a_5], 3) ->
        [
            [a_1, a_2, a_3],
            [a_2, a_3, a_4],
            [a_3, a_4, a_5],
        ]

    """
    assert (
        actions.ndim == 3
    ), f"Expected actions to have shape (batch, window_size, action_dim), but got shape {actions.shape}"
    window_size = actions.shape[1]
    assert (
        window_size >= pred_horizon
    ), f"pred_horizon {pred_horizon} too large for window size {window_size}"
    chunk_window_size = window_size - (pred_horizon - 1)

    curr_step = torch.arange(chunk_window_size)
    action_offset = torch.arange(pred_horizon)
    chunk_indices = curr_step[:, None] + action_offset[None, :]
    return actions[:, chunk_indices]


def _check_action_window_size(actions, window_size, pred_horizon):
    assert (
        actions.shape[1] >= window_size + pred_horizon - 1
    ), f"""
        To predict actions for window_size {window_size} and future prediction horizon {pred_horizon},
        the ground-truth actions must have at least {window_size + pred_horizon - 1} timesteps, but got shape {actions.shape}.

        Did you make sure to set "future_action_window_size" correctly in the data config?
    """


def continuous_loss(
    pred_value,
    noise,
    actions_flat,
    mask,
    loss_type: str = "mse",
    pred_type='noise',
):
    """
    Args:
        pred_value: shape (batch_dims...)
        ground_truth_value: continuous values w/ shape (batch_dims...)
        mask: broadcastable to ground_truth
    """
    if 'noise' in pred_type:
        if loss_type == "mse":
            loss = torch.square(pred_value - noise)
        elif loss_type == "l1":
            loss = torch.abs(pred_value - noise)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")
    
    if 'action' in pred_type:
        if loss_type == "mse":
            loss = torch.square(pred_value - actions_flat)
        elif loss_type == "l1":
            loss = torch.abs(pred_value - actions_flat)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")
    
    loss = masked_mean(loss, mask)
    return loss, {
        f"{loss_type}_loss": loss
    }


def discrete_loss(
    discrete_tokenizer: BinTokenizer,
    logits,
    ground_truth_value,
    mask,
):
    """
    Args:
        discrete_tokenizer: BinTokenizer to use on ground_truth_value
        logits: shape (batch_dims..., vocab_size)
        ground_truth_value: continuous values in w/ shape (batch_dims...)
        mask: broadcastable to ground_truth_value
    """
    labels = discrete_tokenizer(ground_truth_value)
    labels_one_hot = torch.nn.functional.one_hot(labels, logits.shape[-1])

    loss = -torch.sum(logits * labels_one_hot, axis=-1)
    loss = masked_mean(loss, mask)

    # compute accuracy between predicted actions and target actions
    pred_label = torch.argmax(logits, axis=-1)
    accuracy = pred_label == labels
    accuracy = masked_mean(accuracy, mask)

    # detokenize the predicted actions
    pred_value = discrete_tokenizer.decode(pred_label)
    mse = torch.square(pred_value - ground_truth_value)
    mse = masked_mean(mse, mask)
    return loss, {
        "loss": loss,
        "mse": mse,
        "accuracy": accuracy,
    }

def softmax_cross_entropy_loss(
    pred_logits,
    noise,
    actions_flat,
    mask,
    pred_type='action',
    action_repr='bits',
):
    """
    Args:
        logits: shape (batch_dims..., vocab_size)
        ground_truth_value: continuous values in w/ shape (batch_dims...)
        mask: broadcastable to ground_truth_value
    """
    if 'action' in pred_type:
        if 'bits' in action_repr: # actions_flat is in bits
            labels = bits2int(actions_flat > 0)
        else:
            labels = torch.argmax(actions_flat, axis=-1)
        # labels_one_hot = torch.nn.functional.one_hot(labels, logits.shape[-1])

        cross_ent_loss = torch.nn.CrossEntropyLoss(reduction='none')

        logits_flat = rearrange(
            pred_logits,
            "b w a -> (b w) a"
        )

        labels_flat = rearrange(
            labels,
            "b w -> (b w)"
        )
        loss = cross_ent_loss(logits_flat, labels_flat) 
        loss = rearrange(
            loss,
            "(b w) -> b w",
            b = labels.shape[0],
            w = labels.shape[1]
        )

        loss = masked_mean(loss, mask)

        # compute accuracy between predicted actions and target actions
        pred_label = torch.argmax(pred_logits, axis=-1)
        accuracy = pred_label == labels
        accuracy = masked_mean(accuracy, mask)

        return loss, {
            "cross_ent_loss": loss,
            "accuracy": accuracy,
        }
    else:
        raise NotImplementedError('Softmax cross entropy loss for noise prediction is not implemented.')


class DiffusionActionHead(nn.Module):
    """Predicts actions uses a diffusion process.

    Only a single pass through the transformer is done to obtain an action embedding at each timestep. The
    action is then predicted using a diffusion process conditioned on this embedding. The diffusion model
    architecture is an MLP with residual connections (see `octo.model.components.diffusion`).

    You may create an embedding by either mean-pooling across tokens (use_map=False) or using multi-head
    attention pooling (use_map=True). It is recommended to use MAP when decoding from the observation token
    stream.
    """
    def __init__(self,
        readout_key: str,
        use_map: bool = False,
        pred_horizon: int = 1,
        action_dim: int = 7,
        n_classes: int = 6,
        max_action: float = 5.0,
        min_action: float = -5.0,
        prediction_type: str = 'noise',
        action_repr: str = 'discrete',
        loss_type: str = "mse",
        embedding_size: int = 384,
        # diffusion-specific config with sane defaults
        time_dim: int = 32,
        num_blocks: int = 3,
        dropout_rate: float = 0.1,
        hidden_dim: int = 256,
        use_layer_norm: bool = True,
        diffusion_steps: int = 20,
        device: str = 'cuda',
    ):
        super().__init__()
        self.readout_key = readout_key
        self.use_map = use_map
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.n_classes = n_classes
        self.max_action = max_action
        self.min_action = min_action
        self.action_repr = action_repr
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.embedding_size = embedding_size

        self.time_dim = time_dim
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.diffusion_steps = diffusion_steps
        self.device = device

        if self.use_map:
            self.map_head = MAPHead()

        # create the diffusion model (score network)
        self.diffusion_model = create_diffusion_model(
            self.action_dim * self.pred_horizon,
            self.n_classes * self.pred_horizon,
            time_dim=self.time_dim,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate,
            hidden_dim=self.hidden_dim,
            use_layer_norm=self.use_layer_norm,
            embedding_size=self.embedding_size,
            device=device,
        )

        # create beta schedule
        self.betas = torch.tensor(cosine_beta_schedule(self.diffusion_steps))
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.tensor(
            [torch.prod(self.alphas[: i + 1]) for i in range(self.diffusion_steps)]
        ).to(self.device)

        if 'bits' in self.action_repr:
            self.bits = int2bits(
                torch.arange(self.n_classes),
                self.action_dim,
                out_dtype=torch.float32
            ).to(self.device)
    
    def forward(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        time: Any = None,
        noisy_actions: Any = None,
        train: bool = True,
    ) -> torch.Tensor:
        """Performs a single forward pass through the diffusion model."""
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        if self.use_map:  # Multi-head attention pooling
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        else:  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        # Now, embeddings is (batch_size, window_size, embedding_size)

        if time is None:
            time = torch.zeros((*embeddings.shape[:2], 1), dtype=torch.float32).to(self.device)
        if noisy_actions is None:
            noisy_actions = torch.zeros(
                (*embeddings.shape[:2], self.action_dim * self.pred_horizon),
                dtype=torch.float32,
            ).to(self.device)

        pred_eps = self.diffusion_model(embeddings, noisy_actions, time, train=train)

        if 'bits' in self.action_repr:
            pred_eps_post = torch.einsum('bwd,do->bwo', F.softmax(pred_eps, dim=-1), self.bits)
            pred_eps_post = (pred_eps_post * 2 - 1) * self.max_action
            return pred_eps, pred_eps_post
        else:
            return pred_eps, pred_eps
        
    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions,
        pad_mask,
        train: bool = True,
    ):
        """Computes the loss for the diffusion objective.

        Args:
            transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
                embedding_size)
            actions: shape (batch_size, >= window_size + pred_horizon - 1, action_dim)
            pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep

        Returns:
            loss: float
            metrics: dict
        """
        batch_size, window_size = pad_mask.shape
        _check_action_window_size(actions, window_size, self.pred_horizon)
        actions_chunked = chunk_actions(actions, self.pred_horizon)
        actions_chunked = actions_chunked[:, :window_size]
        # fold action_dim and pred_horizon into one dimension
        actions_flat = rearrange(actions_chunked, "b w p a -> b w (p a)")
        actions_flat = torch.clip(actions_flat, self.min_action, self.max_action)
        
        time = torch.randint(0, self.diffusion_steps, (batch_size, window_size, 1)).to(self.device) # TODO(saumya): why is window dimension added to the actions
        noise = torch.randn(actions_flat.shape).to(self.device)

        alpha_hat = self.alpha_hats[time]
        alpha_1 = torch.sqrt(alpha_hat)
        alpha_2 = torch.sqrt(1 - alpha_hat)
        noisy_actions = alpha_1 * actions_flat + alpha_2 * noise
        
        pred_eps, pred_eps_post = self(
            transformer_outputs, train=train, time=time, noisy_actions=noisy_actions
        )

        if 'mse' in self.loss_type or 'l1' in self.loss_type:
            loss, metrics = continuous_loss(
                pred_eps_post,
                noise,
                actions_flat,
                pad_mask[:, :, None],
                loss_type=self.loss_type,
                pred_type=self.prediction_type
            )
            metrics[f"{self.loss_type}_loss"] = metrics[f"{self.loss_type}_loss"] * self.action_dim
        elif 'softmax_cross_ent' in self.loss_type:
            loss, metrics = softmax_cross_entropy_loss(
                pred_eps,
                noise,
                actions_flat,
                pad_mask,
                pred_type=self.prediction_type,
                action_repr=self.action_repr,
            )
        else:
            raise NotImplementedError("Loss type not defined.")
        
        
        # Sum over action dimension instead of averaging
        loss = loss * self.action_dim
        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        train: bool = True,
        argmax: bool = False,
        sample_shape: tuple = (),
        temperature: float = 1.0,
        device = 'cuda',
    ) -> torch.Tensor:
        """Convenience methods for predicting actions for the final timestep in the window."""
        def scan_fn(current_x, time_now, time_next):
            
            input_time = torch.broadcast_to(time_now, (*current_x.shape[:-1], 1))
            eps_pred, eps_pred_post = self(transformer_outputs, input_time, current_x, train=train)
            
            if 'noise' in self.prediction_type:
                alpha_1 = 1 / torch.sqrt(self.alphas[time_now])
                alpha_2 = (1 - self.alphas[time_now]) / (torch.sqrt(1 - self.alpha_hats[time_now]))
                current_x = alpha_1 * (current_x - alpha_2 * eps_pred_post)
            elif 'action' in self.prediction_type:
                alpha_1 = 1 / (torch.sqrt(1 - self.alpha_hats[time_now]))
                eps = alpha_1 * (current_x - torch.sqrt(self.alpha_hats[time_now]) * eps_pred_post)
                current_x = torch.sqrt(self.alpha_hats[time_next]) * eps_pred_post + (torch.sqrt(1 - self.alpha_hats[time_next])) * eps

            z = torch.randn(current_x.shape).to(device=device)
            current_x = current_x + (time_now > 0) * (torch.sqrt(self.betas[time_now]) * z)

            current_x = torch.clip(current_x, self.min_action, self.max_action)

            return current_x

        def sample_actions(device):
            batch_size, window_size = transformer_outputs[
                self.readout_key
            ].tokens.shape[:2]
            
            actions_flat = torch.randn((batch_size, window_size, self.pred_horizon * self.action_dim)).to(device=device)
            actions_flat = torch.clip(actions_flat, self.min_action, self.max_action)

            steps = torch.arange(self.diffusion_steps - 1, -1, -1).to(device=device)
            for i, x in enumerate(steps):
                actions_flat = scan_fn(actions_flat, steps[i], steps[min(i+1, len(steps)-1)])

            actions = rearrange(
                actions_flat,
                "b w (p a) -> b w p a",
                p=self.pred_horizon,
                a=self.action_dim,
            )
            if 'bits' in self.action_repr:
                bits = actions > 0
                action_tokens = bits2int(bits)
                action_tokens = torch.clip(action_tokens, 0, self.n_classes-1)

            return action_tokens[:,-1]

        actions = sample_actions(device)
        return actions
