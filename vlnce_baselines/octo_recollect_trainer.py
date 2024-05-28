import os
import time
import warnings
from typing import List
import datetime
from PIL import Image
import numpy as np
import json

from habitat import Config
from gym import Space

from collections import defaultdict
from einops import rearrange

import torch
import tqdm
import wandb

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class

from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    get_active_obs_transforms,
    apply_obs_transforms_obs_space,
)

from habitat.utils.visualizations.utils import append_text_to_image

from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.recollect_trainer import RecollectTrainer


from vlnce_baselines.common import recollection_dataset

from vlnce_baselines.common.utils import (
    create_optimizer_with_params, 
    create_schedulder_with_params, 
    extract_instruction_tokens,
)
from vlnce_baselines.common.utils import TopKLogger

from vlnce_baselines.models.octo.octo_utils import get_octo_data
from vlnce_baselines.models.octo_policy import OctoPolicy
from habitat_extensions.utils import generate_video, observations_to_image, text_to_append

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    # import tensorflow as tf  # noqa: F401

from omegaconf import OmegaConf

class TrajCollator(object):
    def __init__(self):
        pass
    
    def __call__(self, batch):
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
        # prev_actions_batch = list(transposed[1])
        # corrected_actions_batch = list(transposed[2])
        actions_for_training = list(transposed[3])
        # weights_batch = list(transposed[4])
        instructions_batch = list(transposed[5])
        
        return (observations_batch, actions_for_training, instructions_batch)

class TimeStepCollator(object):
    def __init__(self):
        pass
    
    def __call__(self, batch):

        transposed = list(zip(*batch))

        rgb_batch = list(transposed[0])
        depth_batch = list(transposed[1])
        rxr_instruction = list(transposed[2])
        # prev_actions = list(transposed[3])
        # oracle_actions = list(transposed[4])
        actions_for_training = list(transposed[5])
        instructions_batch = list(transposed[6])
        
        return (
            rgb_batch,
            depth_batch,
            rxr_instruction,
            actions_for_training,
            instructions_batch
        )
    
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
        else:
            lr = self.optimizer.param_groups[0]['lr']
        return lr

    def update_octo_config(self):
        if 'discrete' in self.octo_config.model.heads.action.kwargs.action_repr:
            self.octo_config.model.heads.action.kwargs.action_dim = 1
            self.octo_config.model.heads.action.kwargs.n_classes = 6
            self.octo_config.model.heads.action.kwargs.max_action = 5
            self.octo_config.model.heads.action.kwargs.min_action = 0
        if 'one-hot' in self.octo_config.model.heads.action.kwargs.action_repr:
            self.octo_config.model.heads.action.kwargs.action_dim = 6
            self.octo_config.model.heads.action.kwargs.n_classes = 6
            self.octo_config.model.heads.action.kwargs.max_action = 1
            self.octo_config.model.heads.action.kwargs.min_action = 0
        if 'bits' in self.octo_config.model.heads.action.kwargs.action_repr:
            nbits = int(np.ceil(np.log2(6)))
            self.octo_config.model.heads.action.kwargs.action_dim = nbits
            self.octo_config.model.heads.action.kwargs.n_classes = 6
            self.octo_config.model.heads.action.kwargs.max_action = self.config.IL.OCTO_TRAINER.scale_bits
            self.octo_config.model.heads.action.kwargs.min_action = -self.config.IL.OCTO_TRAINER.scale_bits
        
        obs_transforms = get_active_obs_transforms(self.config)
        env = get_env_class(self.config.ENV_NAME)(config=self.config)
        observation_space = apply_obs_transforms_obs_space(
            env.observation_space, obs_transforms
        )
        self.octo_config.image_encoder_kwargs.VlnResnetRGBDEncoder.kwargs.DEPTH_ENCODER.observation_space.shape=observation_space.spaces['depth'].shape
        self.octo_config.image_encoder_kwargs.VlnResnetRGBDEncoder.kwargs.DEPTH_ENCODER.observation_space.low=float(observation_space.spaces['depth'].low[0,0,0])
        self.octo_config.image_encoder_kwargs.VlnResnetRGBDEncoder.kwargs.DEPTH_ENCODER.observation_space.high=float(observation_space.spaces['depth'].high[0,0,0])

        encoder_name = self.octo_config.model.observation_tokenizers.primary.kwargs.encoder.name
        self.octo_config.model.observation_tokenizers.primary.kwargs.encoder.module = self.octo_config['image_encoder_kwargs'][encoder_name]['module']
        self.octo_config.model.observation_tokenizers.primary.kwargs.encoder.kwargs = self.octo_config['image_encoder_kwargs'][encoder_name]['kwargs']

        self.octo_config.model.observation_tokenizers.primary.kwargs.num_tokens = self.octo_config['image_encoder_kwargs'][encoder_name]['num_tokens']
    
    @staticmethod
    def _pause_envs(envs_to_pause, envs, batch, task, rgb_frames=None):
        # pausing envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            batch["observation"]['image_primary'] = batch["observation"]['image_primary'][state_index]
            batch["observation"]['bert_tokens'] = batch["observation"]['bert_tokens'][state_index]
            batch["observation"]['pad_mask'] = batch["observation"]['pad_mask'][state_index]
            batch["observation"]['pad_mask_dict']['image_primary'] = batch["observation"]['pad_mask_dict']['image_primary'][state_index]

            if "language_instruction" in task.keys():
                for k, v in task["language_instruction"].items():
                    task["language_instruction"][k] = v[state_index]
            task["pad_mask_dict"]["language_instruction"] = task["pad_mask_dict"]["language_instruction"][state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (envs, batch, task, rgb_frames)
    
    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        eval_dir = '',
    ) -> None:
        if load_from_ckpt:
            if 'wandb' in self.config.IL.OCTO_TRAINER.logger_type:
                logger.info(f"Run path: {self.config.EVAL.wandb_load.run_path}. File: {self.config.EVAL.wandb_load.file}")
                ckpt_file = wandb.restore(str(self.config.EVAL.wandb_load.file), run_path=self.config.EVAL.wandb_load.run_path,
                    root=eval_dir, replace=False)
                ckpt = torch.load(ckpt_file.name, map_location=self.device)
            
            if 'tb' in self.config.IL.OCTO_TRAINER.logger_type:
                logger.info(f"checkpoint_path: {self.config.EVAL.tb_load.EVAL_CKPT_PATH_DIR}")
                ckpt = self.load_checkpoint(self.config.EVAL.tb_load.EVAL_CKPT_PATH_DIR, map_location=self.device)
            self.octo_config = ckpt['octo_config']
        else:
            self.octo_config = OmegaConf.load(config['OCTO_CONFIG_PATH'])

        self.update_octo_config()
        if 'wandb' in self.config.IL.OCTO_TRAINER.logger_type:
            wandb.config.update({'octo_config': dict(self.octo_config)})
            wandb.config.update({'main_config': config})

        example_batch, dataset_statistics = get_octo_data(self.octo_config.pretrained_ckpt_path)
        self.policy = OctoPolicy.from_config(self.octo_config, example_batch, dataset_statistics, self.device)
        self.policy.to(self.device)
        
        self.print_parameters(self.policy)
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
            self.policy.load_state_dict(ckpt["state_dict"])
            logger.info(f"Loaded weights from checkpoint")

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

    def chunk_and_transform_traj_batch(self, obs_batch, obs_transforms):
        
        if 'OctoTeacherRecollectionDataset' in self.config.DATASET:
            observations_batch, action_batch, instructions_batch = obs_batch
            B = len(observations_batch)
            transformed_rgbd_batch = [
                apply_obs_transforms_batch(
                    {
                        'rgb': observations_batch[bid]['rgb'].to(device=self.device, non_blocking=True),
                        'depth': observations_batch[bid]['depth'].to(device=self.device, non_blocking=True),
                    },
                    obs_transforms,
                ) for bid in range(B)
            ]

            new_observations_batch = defaultdict(list)
            for sensor in observations_batch[0]:
                for bid in range(B):
                    traj_len = observations_batch[bid][sensor].shape[0] - (self.octo_config.window_size - 1)
                    if 'rxr_instruction' in sensor: # don't chunk bert_tokens
                        new_observations_batch[sensor].append(
                            observations_batch[bid][sensor][:traj_len]
                        )
                    else:
                        curr_step = torch.arange(traj_len)
                        obs_offset = torch.arange(self.octo_config.window_size)
                        chunk_indices = curr_step[:, None] + obs_offset[None, :]

                        new_observations_batch[sensor].append(
                            transformed_rgbd_batch[bid][sensor][chunk_indices]
                        )

            observations_batch = new_observations_batch
            for sensor in observations_batch:
                observations_batch[sensor] = torch.cat(
                    observations_batch[sensor], dim=0
                ).to(device=self.device, non_blocking=True)

            # corrected_actions_batch = torch.cat(corrected_actions_batch, dim=0).flatten()
            actions_with_horizon = []
            for bid in range(B):
                traj_len = action_batch[bid].shape[0] - (self.octo_config.window_size - 1)
                curr_step = torch.arange(traj_len)
                obs_offset = torch.arange(self.octo_config.window_size + self.octo_config.pred_horizon - 1)
                chunk_indices = torch.minimum(
                    torch.tensor(action_batch[bid].shape[0]-1),
                    (curr_step[:, None] + obs_offset[None, :])
                ) # repeat last action which should be STOP action
                actions_with_horizon.append(action_batch[bid][chunk_indices])
            actions_with_horizon = torch.cat(actions_with_horizon, dim=0).to(device=self.device, non_blocking=True)

            instructions_batch_flat = []
            for row in instructions_batch:
                traj_len = len(row) - (self.octo_config.window_size - 1)
                instructions_batch_flat += row[:traj_len]
        elif 'OctoTimeStepsTeacherRecollectionDataset' in self.config.DATASET:
            rgb_batch, depth_batch, rxr_instruction_batch, action_batch, instructions_batch_flat = obs_batch
            B = len(rgb_batch)
            start = time.time()
            rgb_batch = torch.stack(
                [b.to(device=self.device, non_blocking=True) for b in rgb_batch], 
                dim=0)
            depth_batch = torch.stack(
                [b.to(device=self.device, non_blocking=True) for b in depth_batch], 
                dim=0)

            print("Stack time:", time.time()-start)
            start = time.time()
            observations_batch = apply_obs_transforms_batch(
                {
                    'rgb':  rearrange(rgb_batch, "b l h w c -> (b l) h w c"),
                    'depth': rearrange(depth_batch, "b l h w c -> (b l) h w c"),
                },
                obs_transforms,
            )
            print("Transform time:", time.time()-start)
            
            start = time.time()
            observations_batch['rgb'] = rearrange(observations_batch['rgb'], "(b l) h w c -> b l h w c", b = B, l = self.octo_config.window_size)
            observations_batch['depth'] = rearrange(observations_batch['depth'], "(b l) h w c -> b l h w c", b = B, l = self.octo_config.window_size)
            print("rearrange time:", time.time()-start)
            
            start = time.time()
            observations_batch['rxr_instruction'] = torch.stack(
                [b.to(device=self.device, non_blocking=True) for b in rxr_instruction_batch], 
                dim=0)
            actions_with_horizon = torch.stack(
                [b.to(device=self.device, non_blocking=True) for b in action_batch], 
                dim=0)
            
            print("Stack2 time:", time.time()-start)
        else:
            raise NotImplementedError('Dataset chunker not implemented.')
        
        final_obs_batch = dict()
        final_obs_batch["observation"] = dict()
        # rgb and depth Shape: (b, window_size, h, w, 4)
        final_obs_batch["observation"]['image_primary'] = (torch.cat((observations_batch['rgb'], observations_batch['depth']), dim=-1))
        
        # rxr_instruction Shape: (b, n_tokens, features)
        final_obs_batch["observation"]['bert_tokens'] = observations_batch['rxr_instruction']

        batch_size = observations_batch['rgb'].shape[0]
        # Pad mask for the whole batch with window_size
        final_obs_batch["observation"]['pad_mask'] = torch.ones((batch_size,self.octo_config.window_size), dtype=bool).to(device=self.device, non_blocking=True)
        final_obs_batch["observation"]['pad_mask_dict'] = dict()
        final_obs_batch["observation"]['pad_mask_dict']['image_primary'] = torch.ones((batch_size,self.octo_config.window_size), dtype=bool).to(device=self.device, non_blocking=True)

        final_obs_batch['action'] = actions_with_horizon # add prediction horizon  Shape: (b, pred_horizon=1, action_dim)

        if "t5" in self.octo_config.model.task_tokenizers.language.kwargs.encoder:
            task = {"language_instruction": [], "pad_mask_dict": {"language_instruction": []}}
            task["language_instruction"] = self.policy.text_processor.encode(instructions_batch_flat)
            for k, v in task["language_instruction"].items():
                task["language_instruction"][k] = v.to(device=self.device, non_blocking=True)
        else:
            task = {"pad_mask_dict": {"language_instruction": []}}
        
        task["pad_mask_dict"]["language_instruction"] = torch.ones(
            len(instructions_batch_flat), dtype=bool
        ).to(device=self.device, non_blocking=True)

        return final_obs_batch, task
    
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
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        self.config.train.scheduler.timm_cosine.epochs = self.config.IL.epochs # TODO(saumya): why is this not updating in cfg
        
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

        # Note: Always initialize policy before dataset
        dataset = getattr(
            recollection_dataset,
            self.config.get('DATASET', 'OctoTeacherRecollectionDataset')
        )(self.config, self.octo_config)
        
        if 'OctoTeacherRecollectionDataset' in self.config.DATASET:
            collate_fn = TrajCollator()
        elif 'OctoTimeStepsTeacherRecollectionDataset' in self.config.DATASET:
            collate_fn = TimeStepCollator()
        else:
            raise NotImplementedError('Collator not implemented.')        
        
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

        if self.config.IL.OCTO_TRAINER.effective_batch_size > 0:
            assert (
                self.config.IL.OCTO_TRAINER.effective_batch_size
                % self.config.IL.batch_size
                == 0
            ), (
                "Gradient accumulation: effective_batch_size"
                " should be a multiple of batch_size."
            )

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

            for batch_idx in t:
                batch_time = time.time()
                batch_str = f"{batch_idx + 1}/{batches_per_epoch}"
                
                start = time.time()
                obs_batch = next(diter)
                
                print("batch time", time.time()-start)

                start = time.time()
                batch, task = self.chunk_and_transform_traj_batch(
                    obs_batch,
                    dataset.obs_transforms
                )
                print("chunk time", time.time()-start)
                del obs_batch

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
                
                action_loss, metrics_dict = self._update_agent(
                    batch,
                    task,
                    step_grad=step_grad,
                    loss_accumulation_scalar=loss_accumulation_scalar,
                )
                # self.print_gpu_memory_usage("After update")
                del batch, task
                # self.print_gpu_memory_usage("After deleting batch")
                # with torch.cuda.device(self.device):
                #     torch.cuda.empty_cache()
                # self.print_gpu_memory_usage("After emptying cache")

                if self.config.use_pbar:
                    t.set_postfix(
                        {
                            "Epoch": epoch_str,
                            "ActionLoss": round(action_loss, 4),
                            "BatchTime": round(time.time() - batch_time, 2),
                        }
                    )

                if 'tb' in self.config.IL.OCTO_TRAINER.logger_type:
                    tb_writer.add_scalar("action_loss", action_loss, self.step_id)
                
                if 'wandb' in self.config.IL.OCTO_TRAINER.logger_type:
                    optim_logs = {'epoch': epoch}
                    optim_logs['optim/lr'] = self.get_current_learning_rate(epoch)

                    log_dict = {'train_loss': action_loss}
                    log_dict.update(metrics_dict)
                    log_dict.update(optim_logs)
                    wandb.log(log_dict, step=self.step_id)

                epoch_logs['loss'].append(action_loss)
                for k, v in metrics_dict.items():
                    if epoch_logs.get(k) is None:
                        epoch_logs[k] = []
                    epoch_logs[k].append(v.item())

                self.step_id += 1  # noqa: SIM113
                del action_loss, metrics_dict
                
            
            # Saving/updates at epoch level
            if self.scheduler and self.scheduler_t_in_epochs:
                self.scheduler.step(epoch)
            epoch_logs['end_train_step'] = self.step_id
            metric = sum(epoch_logs['loss'])/len(epoch_logs['loss'])
            self.save_checkpoint(epoch, self.step_id, ckpt_save_dir, metric, self.config.wandb.saver.upload)

        dataset.close_sims()
        if 'tb' in self.config.IL.OCTO_TRAINER.logger_type:
            tb_writer.exit()


    def _update_agent(self, 
        batch,
        task,
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
    ):
        transformer_embeddings = self.policy.module.octo_transformer(
            batch["observation"],
            task,
            batch["observation"]["pad_mask"]
        )
        action_loss, loss_dict = self.policy.module.heads[0].loss(
            transformer_embeddings,  # Action head knows to pull out the action readout_key
            batch["action"],
            pad_mask=batch["observation"]["pad_mask"]
        )

        action_loss = action_loss / loss_accumulation_scalar
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

        return action_loss.item(), loss_dict
    
    def _setup_eval_config(self, checkpoint_config: Config) -> Config:
        r"""Sets up and returns a merged config for evaluation. Config
            object saved from checkpoint is merged into config file specified
            at evaluation time with the following overwrite priority:
                  eval_opts > ckpt_opts > ckpt_cfg > eval_cfg
            Evaluation should be done based on the training config (ckpt_cfg) and only EVAL cfg settings should be updated
            If the saved config is outdated, only the eval config is returned.

        Args:
            checkpoint_config: saved config from checkpoint.

        Returns:
            Config: merged config for eval.
        """

        config = self.config.clone()
        ckpt_cmd_opts = checkpoint_config.get('CMD_TRAILING_OPTS', [])
        eval_cmd_opts = config.CMD_TRAILING_OPTS
        eval_config = Config({'EVAL': config.EVAL})
        eval_task_config = Config({'TASK_CONFIG': config.EVAL.EVAL_TASK_CONFIG})
        eval_common_config = Config(config.EVAL.COMMON)

        config.merge_from_other_cfg(checkpoint_config['config']) # Evaluation should be done based on the training config
        config.merge_from_other_cfg(eval_config) # EVAL cfg settings should be updated
        config.merge_from_other_cfg(eval_task_config) # EVAL_TASK cfg settings should be updated
        config.merge_from_other_cfg(eval_common_config) # EVAL_TASK cfg settings should be updated
        config.merge_from_list(ckpt_cmd_opts)
        config.merge_from_list(eval_cmd_opts)

        config.defrost()

        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = self.config.SENSORS
        config.freeze()

        return config

    def _create_batch(self, obs, instructions):

        rgb, depth, bert_tokens = [], [], []
        for o in obs:
            rgb.append(o['rgb'])
            depth.append(o['depth'])
            bert_tokens.append(o['rxr_instruction'][:self.octo_config.bert_max_tokens,:])

        batch = dict()
        batch["observation"] = dict()

        # appending rgb and depth together and creating history dimension. Shape: (b, history=1, h, w, 4)
        rgb = torch.from_numpy(np.stack(rgb, axis=0)).to(device=self.device, non_blocking=True)
        depth = torch.from_numpy(np.stack(depth, axis=0)).to(device=self.device, non_blocking=True)
        batch["observation"]['bert_tokens'] = torch.from_numpy(np.stack(bert_tokens, axis=0)).to(device=self.device, non_blocking=True)
        
        batch_size =rgb.shape[0]
        # Pad mask for the whole batch with history=1
        batch["observation"]['pad_mask'] = torch.ones((batch_size,1), dtype=bool).to(device=self.device, non_blocking=True)
        batch["observation"]['pad_mask_dict'] = dict()
        batch["observation"]['pad_mask_dict']['image_primary'] = torch.ones((batch_size,1), dtype=bool).to(device=self.device, non_blocking=True)


        if "t5" in self.octo_config.model.task_tokenizers.language.kwargs.encoder:
            task = {"language_instruction": [], "pad_mask_dict": {"language_instruction": []}}
            task["language_instruction"] = self.policy.text_processor.encode(instructions)
            for k, v in task["language_instruction"].items():
                task["language_instruction"][k] = v.to(device=self.device, non_blocking=True)
        else:
            task = {"pad_mask_dict": {"language_instruction": []}}
        
        task["pad_mask_dict"]["language_instruction"] = torch.ones(
            len(instructions), dtype=bool
        ).to(device=self.device, non_blocking=True)

        observations_batch = apply_obs_transforms_batch(
            {
                'rgb': rgb,
                'depth': depth,
            },
            self.obs_transforms,
        )
        batch["observation"]['image_primary'] = (torch.cat((observations_batch['rgb'], observations_batch['depth']), dim=-1)).unsqueeze(1)
        return batch, task

    def eval(self, context=False) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer
        or BaseILTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.EVAL.COMMON.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        current_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if 'wandb' in self.config.EVAL.logger_type:
            eval_dir = f'{self.config.CHECKPOINT_FOLDER}_evals/{current_time}/evals'
            os.makedirs(eval_dir, exist_ok=True)
            wandb_dir = f'{self.config.CHECKPOINT_FOLDER}_evals/{current_time}/wandb'
            os.makedirs(wandb_dir, exist_ok=False)

            self.wandb_run = wandb.init(project=self.config.EVAL.wandb_load.project, 
                            entity=self.config.EVAL.wandb_load.entity, 
                            group=self.config.EVAL.wandb_load.group,
                            name=f'{self.config.EVAL.wandb_load.name}_{self.config.EVAL.SPLIT}_{current_time}',
                            dir=wandb_dir)
            wandb.config.update({"run_path": self.config.EVAL.wandb_load.run_path,
                                  "run_file": self.config.EVAL.wandb_load.file})
            tb_writer = None

        if 'tb' in self.config.EVAL.logger_type:
            tb_dir = f'{self.config.EVAL.EVAL_LOG_DIR}/tb_logs'
            os.makedirs(tb_dir, exist_ok=True)
            eval_dir = f'{self.config.EVAL.EVAL_LOG_DIR}/evals'
            os.makedirs(eval_dir, exist_ok=True)

            tb_writer = TensorboardWriter(
                tb_dir,
                flush_secs=self.flush_secs,
            )
            assert (
                len(self.config.EVAL.EVAL_LOG_DIR) > 0
            ), "Must specify a tensorboard directory for logging"
        
        if context:
            self._eval_checkpoint_context(
                self.config.EVAL.EVAL_CKPT_PATH_DIR,
                tb_writer,
                checkpoint_index=ckpt_idx,
            )
        else:
            self._eval_checkpoint(
                tb_writer,
                eval_dir,
            )

    def _eval_checkpoint(
        self,
        tb_writer: TensorboardWriter,
        eval_dir,
    ) -> None:
        """Evaluates a single checkpoint.

        Args:
            writer: tensorboard writer object. None if using wandb
        """
        if tb_writer is None:
            logger.info(f"Run path: {self.config.EVAL.wandb_load.run_path}. File: {self.config.EVAL.wandb_load.file}")
            ckpt_file = wandb.restore(str(self.config.EVAL.wandb_load.file), run_path=self.config.EVAL.wandb_load.run_path,
                root=eval_dir, replace=False)
            ckpt = torch.load(ckpt_file.name, map_location=self.device)
        else:
            logger.info(f"checkpoint_path: {self.config.EVAL.EVAL_CKPT_PATH_DIR}")
            ckpt = self.load_checkpoint(self.config.EVAL.EVAL_CKPT_PATH_DIR, map_location='cpu')
            
        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self.config.clone()
            config = self._setup_eval_config(ckpt)

        self.obs_transforms = get_active_obs_transforms(config)

        split = config.EVAL.SPLIT
        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = split
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
        config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.use_pbar = not is_slurm_batch_job()

        if config.EVAL.SAVE_VIDEO:
            video_dir = f'{eval_dir}/videos_ckpt/{split}_{config.TASK_CONFIG.TASK.SUCCESS_DISTANCE}'
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")

        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                eval_dir,
                f"stats_ckpt_{split}_SD{config.TASK_CONFIG.TASK.SUCCESS_DISTANCE}.json",
            )
            if os.path.exists(fname):
                logger.info("skipping -- evaluation exists.")
                return

        envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )

        self._initialize_policy(
            config,
            load_from_ckpt=True,
            eval_dir=eval_dir,
        )
        self.policy.eval()

        observations = envs.reset()

        current_episodes = envs.current_episodes()
        instructions = [current_episodes[i].instruction for i in range(envs.num_envs)]

        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        
        batch, task = self._create_batch(observations, instructions)

        stats_episodes = {}
        rgb_frames = [[] for _ in range(envs.num_envs)]

        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)

        pbar = tqdm.tqdm(total=num_eps) if config.use_pbar else None
        log_str = (
            f"[Ckpt:]"
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        actions_all_envs = [[] for _ in range(envs.num_envs)]
        step = 0
        while envs.num_envs > 0 and len(stats_episodes) < num_eps:
            current_episodes = envs.current_episodes()
            with torch.no_grad():
                actions = self.policy.sample_actions(
                    batch["observation"],
                    task,
                    batch["observation"]["pad_mask"],
                    argmax = True,
                )
            outputs = envs.step([a[0].item() for a in actions])
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            log_keys = [k for k in infos[0].keys() if 'agent' not in k]
            # reset envs and observations if necessary
            for i in range(envs.num_envs):
                if config.EVAL.SAVE_VIDEO:
                    frame = observations_to_image(observations[i], infos[i])
                    _text = text_to_append(current_episodes[i].instruction)
                    frame = append_text_to_image(frame, _text)
                    rgb_frames[i].append(frame)
                    actions_all_envs[i].append(actions[i][0].item())

                done_eval_i = envs.call_at(i, "get_done_eval", None)
                # if not dones[i]:
                if not done_eval_i:
                    continue
                step += 1

                ep_id = current_episodes[i].episode_id
                stats_episodes[ep_id] = infos[i]
                
                observations[i] = envs.reset_at(i)[0]

                if config.use_pbar:
                    pbar.update()
                else:
                    logger.info(
                        log_str.format(
                            evaluated=len(stats_episodes),
                            total=num_eps,
                            time=round(time.time() - start_time),
                        )
                    )

                if config.EVAL.SAVE_VIDEO:
                    generate_video(
                        video_option=config.EVAL.VIDEO_SAVE_LOC,
                        video_dir=video_dir,
                        images=rgb_frames[i],
                        episode_id=ep_id,
                        checkpoint_idx=0,
                        metrics={"spl": stats_episodes[ep_id]["spl_rxr_no_stop"]},
                        tb_writer=tb_writer,
                    )
                    del stats_episodes[ep_id]["top_down_map_vlnce"]
                    rgb_frames[i] = []
                actions_all_envs = [[] for _ in range(envs.num_envs)]

                if 'wandb' in self.config.EVAL.logger_type:
                    log_dict = {}
                    for m in log_keys:
                        log_dict[m] = stats_episodes[ep_id][m]
                        log_dict[f'{m}_aggr'] = np.sum(stats_episodes[k][m] for k in stats_episodes.keys()) / len(stats_episodes.keys())
                    log_dict["episode_id"] = int(ep_id)
                    wandb.log(log_dict, step=step)

            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )

            instructions = [current_episodes[i].instruction for i in range(envs.num_envs)]
            batch, task = self._create_batch(observations, instructions)

            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            (envs, batch, task, rgb_frames) = self._pause_envs(envs_to_pause, envs, batch, task, rgb_frames)

        envs.close()
        if config.use_pbar:
            pbar.close()

        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for k in log_keys:
            aggregated_stats[f'{k}_aggr'] = (
                    np.sum(v[k] for v in stats_episodes.values()) / num_episodes
                )
        
        if 'wandb' in self.config.EVAL.logger_type:
            wandb.log(aggregated_stats, step=step+1)

        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(aggregated_stats, f, indent=4)
        
        if 'tb' in self.config.EVAL.logger_type:
            logger.info(f"Episodes evaluated: {num_episodes}")
            for k, v in aggregated_stats.items():
                logger.info(f"{k}: {v:.6f}")
                tb_writer.add_scalar(f"eval_{split}_{k}", v, 0)