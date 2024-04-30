import os
import time
import warnings
from typing import List
import datetime
from PIL import Image
import numpy as np

from habitat import Config
from gym import Space

from collections import defaultdict

import torch
import tqdm
import wandb

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry


from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.recollect_trainer import RecollectTrainer

from vlnce_baselines.common.recollection_dataset import (
    OctoTeacherRecollectionDataset,
)
from vlnce_baselines.common.utils import create_optimizer_with_params, create_schedulder_with_params
from vlnce_baselines.common.utils import TopKLogger

from vlnce_baselines.models.octo.octo_utils import get_octo_data
from vlnce_baselines.models.octo_policy import OctoPolicy

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    # import tensorflow as tf  # noqa: F401

from omegaconf import OmegaConf

def collate_fn(batch):
    """Each sample in batch: (
        obs,
        tasks,
        prev_actions,
        oracle_actions,
        inflec_weight,
    )
    """

    transposed = list(zip(*batch))
    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])
    tasks_batch = list(transposed[4])
    
    B = len(prev_actions_batch)

    # Transpose observations
    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(
                observations_batch[bid][sensor]
            )
    observations_batch = new_observations_batch
    for sensor in observations_batch:
        observations_batch[sensor] = torch.cat(
            observations_batch[sensor], dim=0
        )

    # Transpose tasks
    new_tasks_batch = defaultdict(list)
    for sensor in tasks_batch[0].keys():
        if sensor not in ('language_instruction', 'pad_mask_dict'):
            for bid in range(B):
                new_tasks_batch[sensor].append(
                    tasks_batch[bid][sensor]
                )
    new_tasks_batch['language_instruction'] = defaultdict(list)
    for sensor in tasks_batch[0]['language_instruction'].keys():
        for bid in range(B):
            new_tasks_batch['language_instruction'][sensor].append(
                tasks_batch[bid]['language_instruction'][sensor]
            )
    new_tasks_batch['pad_mask_dict'] = defaultdict(list)
    for sensor in tasks_batch[0]['pad_mask_dict'].keys():
        for bid in range(B):
            new_tasks_batch['pad_mask_dict'][sensor].append(
                tasks_batch[bid]['pad_mask_dict'][sensor]
            )
    
    tasks_batch = new_tasks_batch
    for sensor in tasks_batch.keys():
        if sensor not in ('language_instruction', 'pad_mask_dict'):
            tasks_batch[sensor] = torch.cat(
                tasks_batch[sensor], dim=0
            )
    for sensor in tasks_batch['language_instruction'].keys():
        tasks_batch['language_instruction'][sensor] = torch.cat(
            tasks_batch['language_instruction'][sensor], dim=0
        )
    for sensor in tasks_batch['pad_mask_dict'].keys():
        tasks_batch['pad_mask_dict'][sensor] = torch.cat(
            tasks_batch['pad_mask_dict'][sensor], dim=0
        )

    corrected_actions_batch = torch.cat(corrected_actions_batch, dim=0).flatten()
    final_obs_batch = dict()
    final_obs_batch["observation"] = dict()
    # appending rgb and depth together and creating history dimension. Shape: (b, history=1, h, w, 4)
    final_obs_batch["observation"]['image_primary'] = (torch.cat((observations_batch['rgb'], observations_batch['depth']), dim=-1)).unsqueeze(1)
    batch_size = observations_batch['rgb'].shape[0]
    # Pad mask for the whole batch with history=1
    final_obs_batch["observation"]['pad_mask'] = torch.tensor([True]*batch_size).unsqueeze(1)
    final_obs_batch["observation"]['pad_mask_dict'] = dict()
    final_obs_batch["observation"]['pad_mask_dict']['image_primary'] = torch.tensor([True]*batch_size).unsqueeze(1)

    final_obs_batch['task'] = tasks_batch
    final_obs_batch['action'] = torch.nn.functional.one_hot(corrected_actions_batch, num_classes=6).unsqueeze(1)

    return final_obs_batch
    
@baseline_registry.register_trainer(name="octo_trainer")
class OctoRecollectTrainer(RecollectTrainer):
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

    def get_current_learning_rate(self, epoch):
        '''Get current learning rate for logging.'''
        lr = None
        if self.scheduler:
            if self.scheduler_is_timm:
                lr = self.scheduler.get_epoch_values(epoch)[0]
            else:
                lr = self.scheduler.get_last_lr()[0]
        return lr

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
    ) -> None:
        
        self.octo_config = OmegaConf.load(config['OCTO_CONFIG_PATH'])
        example_batch, dataset_statistics = get_octo_data(self.octo_config.pretrained_ckpt_path)
        self.policy = OctoPolicy.from_config(self.octo_config, example_batch, dataset_statistics, self.device)
        self.policy.to(self.device)
        # wandb.config.update(opt)

        self.optimizer = create_optimizer_with_params(config.train.optimizer, self.policy.get_trainable_parameters())

        # Create scheduler
        if config.train.scheduler.use:
            self.scheduler, scheduler_extra_dict = create_schedulder_with_params(config.train.scheduler, self.optimizer)
            # If True update scheduler at epochs else after every step
            self.scheduler_t_in_epochs = scheduler_extra_dict['t_in_epochs']
            self.scheduler_is_timm = scheduler_extra_dict['timm_scheduler']
        else:
            self.scheduler = None

        self.use_grad_clip = config.train.grad_clip.use
        if self.use_grad_clip:
            self.grad_clip_norm = config.train.grad_clip.norm

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

    def save_checkpoint(
        self, 
        epoch: int,
        step_id: int,
        ckpt_save_dir,
        metric,
        upload_to_wandb: bool,
    ) -> None:

        save_file_name = os.path.join(ckpt_save_dir, f'ckpt_epoch_{epoch}_step_{step_id}_action_loss_{metric:.3f}.pth')
        
        status = self.topk_logger.push(save_file_name, -1*metric)
        if status:
            torch.save(
                obj={
                    "state_dict": self.policy.state_dict(),
                    "config": self.config,
                    "octo_config": self.octo_config,
                    "optim_state": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "step_id": step_id,
                },
                f=save_file_name,
            )
            if upload_to_wandb:
                wandb.save(save_file_name, base_path=os.path.join(ckpt_save_dir, '..'))

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
        self.config.train.scheduler.timm_cosine.epochs = self.config.IL.epochs # TODO: why is this not updating in cfg
        self.config.freeze()

        current_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        tb_dir = f'{self.config.TENSORBOARD_DIR}/{current_time}/tb_logs'
        os.makedirs(tb_dir, exist_ok=False)
        ckpt_save_dir = f'{self.config.CHECKPOINT_FOLDER}/{current_time}/checkpoints'
        os.makedirs(ckpt_save_dir, exist_ok=False)
        wandb_dir = f'{self.config.CHECKPOINT_FOLDER}/{current_time}/wandb'
        os.makedirs(wandb_dir, exist_ok=False)

        self.topk_logger = TopKLogger(self.config['wandb']['saver'].get('save_top_k', 5))

        if 'wandb' in self.config.IL.OCTO_TRAINER.logger_type:
            self.wandb_run = wandb.init(project=self.config.wandb.project, 
                            entity=self.config.wandb.entity, 
                            group=self.config.wandb.group,
                            name=f'{self.config.wandb.name}_{current_time}',
                            dir=wandb_dir)

        if 'tb' in self.config.IL.OCTO_TRAINER.logger_type:
            tb_writer = TensorboardWriter(
                tb_dir,
                flush_secs=self.flush_secs,
                purge_step=0,
            )

        self._initialize_policy(
                self.config,
                self.config.IL.load_from_ckpt
            )
        self.policy.train()

        dataset = OctoTeacherRecollectionDataset(self.policy, self.config)
        
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
        
        trainable_size = np.sum([np.prod(p.shape) for p in self.policy.get_trainable_parameters() if p.requires_grad])
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

            epoch_logs = {}
            epoch_logs['loss'] = []
            epoch_logs['optim/lr'] = []
            epoch_logs['start_train_step'] = self.step_id

            # self.save_checkpoint(epoch, self.step_id, ckpt_save_dir, 0, self.config.wandb.saver.upload)

            for batch_idx in t:
                batch_time = time.time()
                batch_str = f"{batch_idx + 1}/{batches_per_epoch}"

                batch = next(diter)
                for k, v in batch["observation"].items():
                    if "pad_mask_dict" not in k:
                        batch["observation"][k] = v.to(device=self.device, non_blocking=True)
                
                for k, v in batch["observation"]['pad_mask_dict'].items():
                    batch["observation"]['pad_mask_dict'][k] = v.to(device=self.device, non_blocking=True)
                
                batch['action'] = batch['action'].to(device=self.device, non_blocking=True)

                for k, v in batch["task"].items():
                    if k not in ("pad_mask_dict", "language_instruction"):
                        batch["task"][k] = v.to(device=self.device, non_blocking=True)
                
                for k, v in batch["task"]["language_instruction"].items():
                    batch["task"]["language_instruction"][k] = v.to(device=self.device, non_blocking=True)
                
                for k, v in batch["task"]["pad_mask_dict"].items():
                    batch["task"]["pad_mask_dict"][k] = v.to(device=self.device, non_blocking=True)

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

                action_loss, loss_dict = self._update_agent(
                    batch,
                    step_grad=step_grad,
                    loss_accumulation_scalar=loss_accumulation_scalar,
                )

                if self.config.use_pbar:
                    t.set_postfix(
                        {
                            "Epoch": epoch_str,
                            # "Loss": round(loss, 4),
                            "ActionLoss": round(action_loss, 4),
                            # "AuxLoss": round(aux_loss, 4),
                        }
                    )
                
                    logger.info(
                        f"[Epoch: {epoch_str}] [Batch: {batch_str}]"
                        + f" [BatchTime: {round(time.time() - batch_time, 2)}s]"
                        + f" [EpochTime: {round(time.time() - epoch_time)}s]"
                        + f" [Loss: {round(action_loss, 4)}]"
                        # + aux_s
                    )

                if 'tb' in self.config.IL.OCTO_TRAINER.logger_type:
                    # tb_writer.add_scalar("loss", loss, self.step_id)
                    tb_writer.add_scalar("action_loss", action_loss, self.step_id)
                    # tb_writer.add_scalar("aux_loss", aux_loss, self.step_id)
                
                if 'wandb' in self.config.IL.OCTO_TRAINER.logger_type:
                    optim_logs = {'epoch': epoch}
                    if self.scheduler:
                        optim_logs['optim/lr'] = self.get_current_learning_rate(epoch)

                    log_dict = {'train_loss': action_loss}
                    log_dict.update(loss_dict)
                    log_dict.update(optim_logs)
                    wandb.log(log_dict, step=self.step_id)

                epoch_logs['loss'].append(action_loss)
                for k, v in loss_dict.items():
                    if epoch_logs.get(k) is None:
                        epoch_logs[k] = []
                    epoch_logs[k].append(v)
                if self.scheduler is not None:
                    epoch_logs['optim/lr'].append(self.get_current_learning_rate(epoch))

                self.step_id += 1  # noqa: SIM113
            
            # Saving/updates at epoch level
            if self.scheduler and self.scheduler_t_in_epochs:
                self.scheduler.step(epoch)
            epoch_logs['end_train_step'] = self.step_id
            metric = sum(epoch_logs['loss'])/len(epoch_logs['loss'])
            self.save_checkpoint(epoch, self.step_id, ckpt_save_dir, metric, self.config.wandb.saver.upload)

        AuxLosses.deactivate()
        dataset.close_sims()
        if 'tb' in self.config.IL.OCTO_TRAINER.logger_type:
            tb_writer.exit()


    def _update_agent(self, 
        batch,
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
    ):
        transformer_embeddings = self.policy.module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["pad_mask"]
        )
        action_loss, loss_dict = self.policy.module.heads["action"].loss(
            transformer_embeddings,  # Action head knows to pull out the action readout_key
            batch["action"],
            pad_mask=batch["observation"]["pad_mask"]
        )
        # action_loss = ((weights * action_loss).sum(0) / weights.sum(0)).mean()

        # aux_mask = (weights > 0).view(-1)
        # aux_loss = AuxLosses.reduce(aux_mask)

        # loss = action_loss + aux_loss
        # loss = loss / loss_accumulation_scalar
        action_loss.backward()

        if step_grad:
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.get_trainable_parameters(), 
                    self.grad_clip_norm)
            self.optimizer.step()

            if self.scheduler and not self.scheduler_t_in_epochs:
                self.scheduler.step()

            self.optimizer.zero_grad()

        # if isinstance(aux_loss, torch.Tensor):
        #     aux_loss = aux_loss.item()
        return action_loss.item(), loss_dict