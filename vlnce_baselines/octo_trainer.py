import os
import time
import warnings
from typing import List
import datetime
from PIL import Image
import numpy as np

from habitat import Config
from gym import Space

import torch
import tqdm
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.recollect_trainer import RecollectTrainer
from vlnce_baselines.common.recollection_dataset import (
    TeacherRecollectionDataset,
)
from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
from vlnce_baselines.dagger_trainer import collate_fn
from vlnce_baselines.common.utils import extract_instruction_tokens
from habitat_extensions.utils import generate_video, observations_to_image, text_to_append, append_text_to_image_fixed_height

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    # import tensorflow as tf  # noqa: F401

from vlnce_baselines.models.octo.spec import ModuleSpec
from absl import flags
from ml_collections import config_flags
FLAGS = flags.FLAGS
flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")
config_dir = os.path.join(os.path.dirname(__file__), "configs")
config_flags.DEFINE_config_file(
    "config",
    os.path.join(config_dir, "config.py:transformer_bc"),
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
NUM_ACTIONS_FOR_VIS = 8

import openai

@baseline_registry.register_trainer(name="octo_trainer")
class OctoTrainer(RecollectTrainer):
    """A Teacher Forcing trainer that re-collects episodes from simulation
    rather than saving them all to disk. Included as starter code for the
    RxR-Habitat Challenge but can also train R2R agents.
    """

    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        os.makedirs(
            os.path.dirname(
                self.config.IL.OCTO_TRAINER.trajectories_file
            ),
            exist_ok=True,
        )
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ) -> None:
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)

        text_processor = ModuleSpec.instantiate(FLAGS.config.text_processor)()
        # consolidate dataset statistics into one big dict
        dataset_statistics = {
            dataset_kwargs["name"]: statistics
            for dataset_kwargs, statistics in zip(
                FLAGS.config.dataset_kwargs["dataset_kwargs_list"],
                train_data.dataset_statistics,
            )
        }

        # self.policy = policy.from_config(
        #     config=config,
        #     observation_space=observation_space,
        #     action_space=action_space,
        # )
        self.policy = policy.from_config(
            FLAGS.config.to_dict(),
            example_batch,
            text_processor,
            verbose=True,
            rng=init_rng,
            dataset_statistics=dataset_statistics,
        )
        self.policy.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.IL.lr
        )
        if load_from_ckpt:
            ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            self.policy.load_state_dict(ckpt_dict["state_dict"])
            if config.IL.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                self.start_epoch = ckpt_dict["epoch"] + 1
                self.step_id = ckpt_dict["step_id"]
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params}. Trainable: {params_t}")
        logger.info("Finished setting up policy.")

    def save_checkpoint(self, epoch: int, step_id: int, ckpt_save_dir) -> None:
        torch.save(
            obj={
                "state_dict": self.policy.state_dict(),
                "config": self.config,
                "optim_state": self.optimizer.state_dict(),
                "epoch": epoch,
                "step_id": step_id,
            },
            f=os.path.join(ckpt_save_dir, f"ckpt.{epoch}.pth"),
        )

    def train(self) -> None:
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = split
        self.config.IL.OCTO_TRAINER.gt_path = (
            self.config.TASK_CONFIG.TASK.NDTW.GT_PATH
        )
        self.config.IL.OCTO_TRAINER.gt_file = (
            self.config.TASK_CONFIG.TASK.NDTW.GT_PATH
        )
        self.config.use_pbar = not is_slurm_batch_job()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        self.config.freeze()

        dataset = TeacherRecollectionDataset(self.config)
        diter = iter(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=False,
                drop_last=True,
                num_workers=1,
            )
        )
        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=dataset.observation_space,
            action_space=dataset.action_space,
        )
        
        trainable_size = np.sum([np.prod(p.shape) for p in self.policy.parameters() if p.requires_grad])
        all_params_size = np.sum([np.prod(p.shape) for p in self.policy.parameters()])
        print(f"all: {all_params_size}, trainable: {trainable_size}")
    
        if trainable_size > 1e9:
            print(f'{float(trainable_size) / 1e9:.2f} G ({trainable_size})')
        elif trainable_size > 1e6:
            print(f'{float(trainable_size) / 1e6:.2f} M ({trainable_size})')
        elif trainable_size > 1e3:
            print(f'{float(trainable_size) / 1e3:.2f} K ({trainable_size})')
        else:
            print(f'{float(trainable_size):.2f}')

        if self.config.IL.OCTO_TRAINER.effective_batch_size > 0:
            assert (
                self.config.IL.OCTO_TRAINER.effective_batch_size
                % self.config.IL.batch_size
                == 0
            ), (
                "Gradient accumulation: effective_batch_size"
                " should be a multiple of batch_size."
            )

        current_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        tb_dir = f'{self.config.TENSORBOARD_DIR}/{current_time}/tb_logs'
        os.makedirs(tb_dir, exist_ok=False)
        ckpt_save_dir = f'{self.config.CHECKPOINT_FOLDER}/{current_time}/checkpoints'
        os.makedirs(ckpt_save_dir, exist_ok=False)

        with TensorboardWriter(
            tb_dir,
            flush_secs=self.flush_secs,
            purge_step=0,
        ) as writer:

            AuxLosses.activate()
            batches_per_epoch = dataset.length // dataset.batch_size

            for epoch in range(self.start_epoch, self.config.IL.epochs):
                epoch_time = time.time()
                epoch_str = f"{epoch + 1}/{self.config.IL.epochs}"

                t = (
                    tqdm.trange(
                        batches_per_epoch, leave=False, dynamic_ncols=True
                    )
                    if self.config.use_pbar
                    else range(batches_per_epoch)
                )

                for batch_idx in t:
                    batch_time = time.time()
                    batch_str = f"{batch_idx + 1}/{batches_per_epoch}"

                    (
                        observations_batch,
                        prev_actions_batch,
                        not_done_masks,
                        corrected_actions_batch,
                        weights_batch,
                    ) = next(diter)

                    observations_batch = apply_obs_transforms_batch(
                        {
                            k: v.to(device=self.device, non_blocking=True)
                            for k, v in observations_batch.items()
                        },
                        dataset.obs_transforms,
                    )

                    prev_actions_batch = prev_actions_batch.to(
                        device=self.device, non_blocking=True
                    )
                    not_done_masks = not_done_masks.to(
                        device=self.device, non_blocking=True
                    )
                    corrected_actions_batch = corrected_actions_batch.to(
                        device=self.device, non_blocking=True
                    )
                    weights_batch = weights_batch.to(
                        device=self.device, non_blocking=True
                    )

                    # gradient accumulation
                    if (
                        self.config.IL.OCTO_TRAINER.effective_batch_size
                        > 0
                    ):
                        loss_accumulation_scalar = (
                            self.config.IL.OCTO_TRAINER.effective_batch_size
                            // self.config.IL.batch_size
                        )
                        step_grad = bool(
                            self.step_id % loss_accumulation_scalar
                        )
                    else:
                        loss_accumulation_scalar = 1
                        step_grad = True

                    loss, action_loss, aux_loss = self._update_agent(
                        observations_batch,
                        prev_actions_batch,
                        not_done_masks,
                        corrected_actions_batch,
                        weights_batch,
                        step_grad=step_grad,
                        loss_accumulation_scalar=loss_accumulation_scalar,
                    )

                    if self.config.use_pbar:
                        t.set_postfix(
                            {
                                "Epoch": epoch_str,
                                "Loss": round(loss, 4),
                                "ActionLoss": round(action_loss, 4),
                                "AuxLoss": round(aux_loss, 4),
                            }
                        )
                    else:
                        if aux_loss != 0.0:
                            aux_s = (
                                f" [ActionLoss: {round(action_loss, 4)}]"
                                + f" [AuxLoss: {round(aux_loss, 4)}]"
                            )
                        else:
                            aux_s = ""
                        logger.info(
                            f"[Epoch: {epoch_str}] [Batch: {batch_str}]"
                            + f" [BatchTime: {round(time.time() - batch_time, 2)}s]"
                            + f" [EpochTime: {round(time.time() - epoch_time)}s]"
                            + f" [Loss: {round(loss, 4)}]"
                            + aux_s
                        )
                    writer.add_scalar("loss", loss, self.step_id)
                    writer.add_scalar("action_loss", action_loss, self.step_id)
                    writer.add_scalar("aux_loss", aux_loss, self.step_id)
                    self.step_id += 1  # noqa: SIM113

                self.save_checkpoint(epoch, self.step_id, ckpt_save_dir)

            AuxLosses.deactivate()
            dataset.close_sims()
