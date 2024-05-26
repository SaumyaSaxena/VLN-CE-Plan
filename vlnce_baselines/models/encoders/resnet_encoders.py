from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from gym import Space, spaces
from habitat.core.simulator import Observations
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from torch import Tensor

from vlnce_baselines.common.utils import single_frame_box_shape

class VlnResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        output_size: int = 128,
        checkpoint: str = "NONE",
        backbone: str = "resnet50",
        resnet_baseplanes: int = 32,
        normalize_visual_inputs: bool = False,
        trainable: bool = False,
        spatial_output: bool = False,
    ) -> None:
        super().__init__()
        self.trainable = trainable

        self.visual_encoder = ResNetEncoder(
            spaces.Dict(
                {
                    "depth": single_frame_box_shape(
                        observation_space.spaces["depth"]
                    )
                }
            ),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint)

            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue

                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v

            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), output_size
                ),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)

    def forward(self, observations: Observations) -> Tensor:
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            if self.trainable:
                x = self.visual_encoder(observations)
            else:
                with torch.no_grad():
                    x = self.visual_encoder(observations)
            

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )
            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)


class TorchVisionResNet(nn.Module):
    """TorchVision ResNet pre-trained on ImageNet. The standard average
    pooling can be replaced with spatial average pooling. The final fc layer
    is replaced with a new fc layer of a specified output size.
    """

    def __init__(
        self,
        output_size: int,
        resnet_version: str = "resnet50",
        normalize_visual_inputs: bool = False,
        trainable: bool = False,
        spatial_output: bool = False,
        single_spatial_filter: bool = True,
    ) -> None:
        super().__init__()
        self.trainable = trainable
        self.normalize_visual_inputs = normalize_visual_inputs
        self.spatial_output = spatial_output
        resnet = getattr(models, resnet_version)(pretrained=True)
        modules = list(resnet.children())
        self.resnet_layer_size = modules[-1].in_features
        self.cnn = nn.Sequential(*modules[:-1])

        for param in self.cnn.parameters():
            param.requires_grad_(trainable)
        self.cnn.train(trainable)

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.resnet_layer_size, output_size),
                nn.ReLU(),
            )
        else:

            class SpatialAvgPool(nn.Module):
                def forward(self, x):
                    x = F.adaptive_avg_pool2d(x, (4, 4))

                    return x

            if single_spatial_filter:
                self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
            self.cnn.avgpool = SpatialAvgPool()
            self.spatial_embeddings = nn.Embedding(4 * 4, 64)
            self.output_shape = (
                self.resnet_layer_size + self.spatial_embeddings.embedding_dim,
                4,
                4,
            )

    def forward(self, observations: Observations) -> Tensor:
        def normalize(imgs: Tensor) -> Tensor:
            """Normalizes a batch of images by:
                1) scaling pixel values to be in the range 0-1
                2) subtracting the ImageNet mean
                3) dividing by the ImageNet variance
                TODO: could be nice to calculate mean and variance for Habitat MP3D scenes.
                    Method: compute for training split with oracle path follower.
            Args:
                imgs: must have pixel values ranging from 0-255. Assumes a size of [Bx3xHxW]
            https://github.com/pratogab/batch-transforms/blob/master/batch_transforms.py
            """
            imgs = imgs.contiguous() / 255.0
            if self.normalize_visual_inputs:
                mean_norm = torch.tensor([0.485, 0.456, 0.406]).to(
                    device=imgs.device
                )[None, :, None, None]
                std_norm = torch.tensor([0.229, 0.224, 0.225]).to(
                    device=imgs.device
                )[None, :, None, None]
                return imgs.sub(mean_norm).div(std_norm)
            else:
                return imgs

        if "rgb_features" in observations:
            resnet_output = observations["rgb_features"]
        else:
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
            rgb_observations = observations["rgb"].permute(0, 3, 1, 2)
            if self.trainable:
                resnet_output = self.cnn(normalize(rgb_observations))
            else:
                with torch.no_grad():
                    resnet_output = self.cnn(normalize(rgb_observations))

        if self.spatial_output:
            b, c, h, w = resnet_output.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=resnet_output.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )
            return torch.cat([resnet_output, spatial_features], dim=1)
        else:
            return self.fc(resnet_output)  # returns [BATCH x OUTPUT_DIM]


class TorchVisionResNet50(TorchVisionResNet):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, resnet_version="resnet50", **kwargs)


class TorchVisionResNet18(TorchVisionResNet):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, resnet_version="resnet18", **kwargs)

class VlnResnetRGBDEncoder(nn.Module):
    def __init__(self,
        normalize_rgb = False,
        DEPTH_ENCODER = None,
        RGB_ENCODER = None,
        device: str = 'cuda',
    ):
        super().__init__()
        depth_space = spaces.Box(
            low=DEPTH_ENCODER.observation_space.low,
            high=DEPTH_ENCODER.observation_space.high, 
            shape=DEPTH_ENCODER.observation_space.shape, 
            dtype=np.float32
        )
        observation_space = spaces.Dict({
            'depth': depth_space,
        })
        self.depth_encoder = eval(DEPTH_ENCODER.cnn_type
        )(
            observation_space,
            output_size=DEPTH_ENCODER.output_size,
            checkpoint=DEPTH_ENCODER.ddppo_checkpoint,
            backbone=DEPTH_ENCODER.backbone,
            trainable=DEPTH_ENCODER.trainable,
            spatial_output=True,
        )

        self.rgb_encoder = eval(RGB_ENCODER.cnn_type
        )(
            RGB_ENCODER.output_size,
            normalize_visual_inputs=normalize_rgb,
            trainable=RGB_ENCODER.trainable,
            spatial_output=True,
        )

        self.num_features = self.depth_encoder.output_shape[0] + self.rgb_encoder.output_shape[0]

    def forward(self, observation):
        obs = {}
        obs['depth'] = observation[...,3:] # channel last
        obs['rgb'] = observation[...,:3] # channel first
        depth_embedding = self.depth_encoder(obs)
        rgb_embedding = self.rgb_encoder(obs)
        return torch.cat([depth_embedding, rgb_embedding], dim=1).permute(0,2,3,1) # permute to change to (b,h,w,c) format