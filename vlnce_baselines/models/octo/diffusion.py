# copied from: https://raw.githubusercontent.com/rail-berkeley/bridge_data_v2/main/jaxrl_m/networks/diffusion_nets.py
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

default_init = nn.init.xavier_uniform_


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class ScoreActor(nn.Module):
    def __init__(self,
        time_preprocess: nn.Module,
        cond_encoder: nn.Module,
        reverse_network: nn.Module,
    ):
        super().__init__()
        self.time_preprocess = time_preprocess
        self.cond_encoder = cond_encoder
        self.reverse_network = reverse_network

    def get_trainable_parameters(self):
        return (
            list(self.time_preprocess.parameters()) + \
            self.cond_encoder.get_trainable_parameters() + \
            list(self.reverse_network.parameters())
        )


    def forward(self, obs_enc, actions, time, train=False):
        t_ff = self.time_preprocess(time)
        cond_enc = self.cond_encoder(t_ff, train=train)
        reverse_input = torch.concatenate([cond_enc, obs_enc, actions], axis=-1)
        eps_pred = self.reverse_network(reverse_input, train=train)
        return eps_pred


class FourierFeatures(nn.Module):
    def __init__(self, output_size: int, input_dim: int = 1, learnable: bool = True, device: str = 'cuda'):
        super().__init__()
        self.output_size = output_size
        self.learnable = learnable
        self.device = device

        self.w = nn.Embedding(output_size // 2, input_dim).to(self.device) # input_dim=1 for time
    
    def reset_parameters(self):
        nn.init.normal_(self.w.weight, std=0.2)

    def forward(self, x: torch.Tensor):
        if self.learnable:
            f = 2 * torch.pi * x @ (self.w.weight).T
        else:
            half_dim = self.output_size // 2
            f = torch.log(10000) / (half_dim - 1)
            f = torch.exp(torch.arange(half_dim).to(self.device) * -f)
            f = x * f
        return torch.concatenate([torch.cos(f), torch.sin(f)], axis=-1)


class MLP(nn.Module):
    def __init__(self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: Callable = nn.SiLU(),
        activate_final: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
        device: str = 'cuda'
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.activate_final = activate_final
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        self.device = device

        self.dense_layers = []
        self.dropout_layers = []
        self.layer_norms = []
        _input_dim = input_dim
        for i, size in enumerate(self.hidden_dims):
            self.dense_layers.append(nn.Linear(
                        in_features=_input_dim,
                        out_features=size
                ).to(self.device)
            )
            
            if i + 1 < len(hidden_dims) or activate_final:
                if dropout_rate is not None and dropout_rate > 0:
                    self.dropout_layers.append(
                        nn.Dropout(dropout_rate).to(self.device)
                    )
                if use_layer_norm:
                    self.layer_norms.append(nn.LayerNorm(_input_dim).to(self.device))
            _input_dim = size

    def get_trainable_parameters(self):
        mlp_params = [param for layer in (self.dense_layers + self.dropout_layers + self.layer_norms) for param in layer.parameters()]
        return mlp_params

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        
        for i, size in enumerate(self.hidden_dims):
            x = self.dense_layers[i](x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = self.dropout_layers[i](x)
                if self.use_layer_norm:
                    x = self.layer_norms[i](x)
                x = self.activation(x)
        return x


class MLPResNetBlock(nn.Module):
    def __init__(self,
        features: int,
        act: Callable,
        dropout_rate: float = None,
        use_layer_norm: bool = False,
        device: str = 'cuda'
    ):  
        super().__init__()
        self.act = act
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        self.device = device

        self.dropout = nn.Dropout(dropout_rate).to(self.device)
        self.layer_norm = nn.LayerNorm(features).to(self.device)
        self.dense_layer1 = nn.Linear(
            in_features=features,
            out_features=features * 4
        ).to(self.device)
        self.dense_layer2 = nn.Linear(
            in_features=features * 4,
            out_features=features
        ).to(self.device)

        self.dense_layer3 = nn.Linear(
            in_features=features,
            out_features=features
        ).to(self.device)

    def forward(self, x, train: bool = False):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = self.dropout(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        x = self.dense_layer1(x)
        x = self.act(x)
        x = self.dense_layer2(x)

        if residual.shape != x.shape:
            residual = self.dense_layer3(residual)

        return residual + x


class MLPResNet(nn.Module):
    def __init__(self,
        num_blocks: int,
        inp_dim: int,
        out_dim: int,
        dropout_rate: float = None,
        use_layer_norm: bool = False,
        hidden_dim: int = 256,
        activation: Callable = nn.SiLU(),
        device: str = 'cuda',
    ):  
        super().__init__()
        self.num_blocks = num_blocks
        self.activation = activation
        self.device = device

        self.dense_layer1 = nn.Linear(
            in_features=inp_dim,
            out_features=hidden_dim
        ).to(self.device)

        self.mlp_resnet_block = MLPResNetBlock(
            hidden_dim,
            act=activation,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
            device=self.device,
        )

        self.dense_layer2 = nn.Linear(
            in_features=hidden_dim,
            out_features=out_dim
        ).to(self.device)

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        x = self.dense_layer1(x)
        for _ in range(self.num_blocks):
            x = self.mlp_resnet_block(x, train=train)
        x = self.activation(x)
        x = self.dense_layer2(x)
        return x


def create_diffusion_model(
    out_dim: int,
    time_dim: int,
    num_blocks: int,
    dropout_rate: float,
    hidden_dim: int,
    use_layer_norm: bool,
    embedding_size: int,
    device,
):  
    inp_dim = time_dim + embedding_size + out_dim
    return ScoreActor(
        FourierFeatures(time_dim, learnable=True, device=device),
        MLP(time_dim, (2 * time_dim, time_dim), device=device),
        MLPResNet(
            num_blocks,
            inp_dim,
            out_dim,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
            device=device,
        ),
    ).to(device)
