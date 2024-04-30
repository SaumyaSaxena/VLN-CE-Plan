import sys, os
sys.path.append('/home/saumyas/Projects/VLN-CE-Plan')

import abc
import torch
from torch import nn
from typing import Any, Optional, Tuple, Dict
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
import numpy as np

from vlnce_baselines.models.octo.octo_module import OctoModule
from vlnce_baselines.models.octo.text_processing import TextProcessor
from vlnce_baselines.models.octo.spec import ModuleSpec

# @baseline_registry.register_policy
class OctoPolicy(nn.Module, metaclass=abc.ABCMeta):

    def __init__(
        self,
        module: OctoModule,
        seed: int,
        text_processor: TextProcessor,
        config: Dict,
        example_batch: Dict,
        dataset_statistics: Dict,
        device: str = 'cuda'
    ):
        super().__init__()
        self.module = module.to(device)
        self.seed = seed
        self.text_processor = text_processor
        self.config = config
        self.example_batch = example_batch
        self.dataset_statistics = dataset_statistics
        self.device = device
    
    @classmethod
    def from_config(
        cls,
        config,
        example_batch,
        dataset_statistics,
        device: str = 'cuda'
    ):
        """Initializes a model with a fresh set of weights from a given config + example_batch.

        Args:
            config (Dict[str, Any]): Config dict. The only required key is "model", but other configuration
                may be saved for posterity.
            example_batch (Dict[str, Any]): Example batch.
            text_processor (Any, optional): Preprocessor for text inputs.
            verbose (bool, optional): Whether to print out a summary of the model.
            rng (Optional[PRNGKey], optional): RNG key for initializing the model.
            dataset_statistics (Optional[Dict[str, Any]], optional): Dataset statistics.
        """
        module = OctoModule.create(**config["model"], device=device)
        text_processor = ModuleSpec.instantiate(config["text_processor"], device)()

        return cls(
            module=module,
            seed=config['seed'],
            text_processor=text_processor,
            example_batch=example_batch,
            config=config,
            dataset_statistics=dataset_statistics,
            device=device
        )
    
    def get_trainable_parameters(self):
        params = self.module.get_trainable_parameters()
        return params

    def sample_actions(
        self,
        observations,
        tasks,
        pad_mask = None,
        train: bool = False,
        argmax: bool = False,
        sample_shape: Tuple[int, ...] = (),
        temperature: float = 1.0,
    ):
        """Samples actions from the model. See `action_heads.py` for more info.

        Args:
            observations: dictionary of arrays of shape (batch_size, window_size, *)
            tasks: dict of tasks of shape (batch_size, *)
            pad_mask: (batch_size, window_size) Boolean mask that is False when the timestep corresponds to padding
            train: whether to run in train mode
            ...see `action_heads.py` for the rest of the kwargs.
        Returns:
            actions: (*sample_shape, batch_size, pred_horizon, action_dim)
        """
        if pad_mask is None:
            pad_mask = observations["pad_mask"]
        transformer_outputs, _ = self.run_transformer(
            observations, tasks, pad_mask, train=train
        )
        return self.module.heads["action"].predict_action(
            transformer_outputs,
            train=train,
            argmax=argmax,
            sample_shape=sample_shape,
            temperature=temperature,
        )

    def create_tasks(self, goals=None, texts=None):
        """Creates tasks dict from goals and texts.

        Args:
            goals: if not None, dict of arrays with shape (batch_size, *)
            texts: if not None, list of texts of length batch_size

        Omit images to run the language-conditioned model, and omit texts to run the
        goal-conditioned model.
        """
        assert goals is not None or texts is not None
        tasks = {"pad_mask_dict": {}}
        if goals is not None:
            tasks.update(goals)
            tasks["pad_mask_dict"].update(
                {k: torch.ones(v.shape[:1], dtype=bool) for k, v in goals.items()}
            )
        else:
            batch_size = len(texts)
            tasks.update(
                {
                    k: torch.from_numpy(np.zeros((batch_size, *v.shape[1:]), dtype=v.dtype))
                    for k, v in self.example_batch["task"].items()
                    if k not in ("pad_mask_dict", "language_instruction")
                }
            )
            tasks["pad_mask_dict"].update(
                {
                    k: torch.zeros(batch_size, dtype=bool)
                    for k in tasks.keys()
                    if k != "pad_mask_dict"
                }
            )

        if texts is not None:
            assert self.text_processor is not None
            tasks["language_instruction"] = texts
            tasks["pad_mask_dict"]["language_instruction"] = torch.ones(
                len(texts), dtype=bool
            )
        else:
            batch_size = jax.tree_leaves(goals)[0].shape[0]
            tasks["language_instruction"] = [""] * batch_size
            tasks["pad_mask_dict"]["language_instruction"] = torch.zeros(
                batch_size, dtype=bool
            )

        if self.text_processor is not None:
            tasks["language_instruction"] = self.text_processor.encode(
                tasks["language_instruction"]
            )
        else:
            del tasks["language_instruction"]

        # _verify_shapes(tasks, "tasks", self.example_batch["task"], starting_dim=1)
        return tasks
    
    def run_transformer(
        self, observations, tasks, pad_mask, train: bool = False
    ):
        """Runs the transformer, but does shape checking on the inputs.

        Args:
            observations: dictionary of arrays of shape (batch_size, window_size, *shape).
                Shape must be consistent with self.example_batch["observation"]
            tasks: dict of tasks of shape (batch_size, *shape)
                Shape must be consistent with self.example_batch["task"]
            pad_mask: (batch_size, window_size) Boolean mask that is False when the timestep corresponds to padding
            train: whether to run in train mode
        """
        # _verify_shapes(
        #     observations,
        #     "observations",
        #     self.example_batch["observation"],
        #     starting_dim=2,
        # )
        # _verify_shapes(tasks, "tasks", self.example_batch["task"], starting_dim=1)

        return self.module(
            observations,
            tasks,
            pad_mask,
            train=train,
        )

    def loss_fn(self, params, batch, rng, train=True):
        bound_module = self.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # action head knows to pull out the "action" readout_key
            batch["action"],
            pad_mask=batch["observation"]["pad_mask"],
            train=train,
        )
        return action_loss, action_metrics