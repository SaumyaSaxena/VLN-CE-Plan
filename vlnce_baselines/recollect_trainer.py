import os
import time
import warnings
from typing import List
import datetime
from PIL import Image

import torch
import tqdm
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
from habitat.utils.visualizations.utils import append_text_to_image

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.recollection_dataset import (
    TeacherRecollectionDataset,
)
from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
from vlnce_baselines.dagger_trainer import collate_fn
from habitat_extensions.utils import generate_video, observations_to_image, text_to_append

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    # import tensorflow as tf  # noqa: F401


@baseline_registry.register_trainer(name="recollect_trainer")
class RecollectTrainer(BaseVLNCETrainer):
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
                self.config.IL.RECOLLECT_TRAINER.trajectories_file
            ),
            exist_ok=True,
        )
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

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
        self.config.IL.RECOLLECT_TRAINER.gt_path = (
            self.config.TASK_CONFIG.TASK.NDTW.GT_PATH
        )
        self.config.IL.RECOLLECT_TRAINER.gt_file = (
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

        if self.config.IL.RECOLLECT_TRAINER.effective_batch_size > 0:
            assert (
                self.config.IL.RECOLLECT_TRAINER.effective_batch_size
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
                        self.config.IL.RECOLLECT_TRAINER.effective_batch_size
                        > 0
                    ):
                        loss_accumulation_scalar = (
                            self.config.IL.RECOLLECT_TRAINER.effective_batch_size
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

    def _eval_checkpoint_context(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        """Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint
        """
        logger.info(f"checkpoint_path: {checkpoint_path}")

        self.ll_model = self.config.EVAL.LL_MODEL
        self.vl_model = self.config.EVAL.VL_MODEL
        self.video_subsample = self.config.EVAL.VIDEO_SUBSAMPLE

        self.load_llm(self.config.EVAL.LL_MODEL)
        self.load_vlm(self.config.EVAL.VL_MODEL)

        config = self.config.clone()
        if self.config.EVAL.USE_CKPT_CONFIG:
            ckpt = self.load_checkpoint(checkpoint_path, map_location="cpu")
            config = self._setup_eval_config(ckpt)

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
        config.IL.ckpt_to_load = checkpoint_path
        config.use_pbar = not is_slurm_batch_job()

        eval_dir = f'{config.EVAL.EVAL_LOG_DIR}/evals'

        if config.EVAL.SAVE_VIDEO:
            assert (
                len(config.EVAL.VIDEO_SAVE_LOC) > 0
            ), "Must specify a video save location "
            video_dir = f'{eval_dir}/videos_ckpt{checkpoint_index}/{split}_{config.TASK_CONFIG.TASK.SUCCESS_DISTANCE}'
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")

        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            
            fname = os.path.join(
                eval_dir,
                f"stats_ckpt_{checkpoint_index}_{split}_SD{config.TASK_CONFIG.TASK.SUCCESS_DISTANCE}.json",
            )
            if os.path.exists(fname):
                logger.info("skipping -- evaluation exists.")
                return

        envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )
        observation_space, action_space = self._get_spaces(config, envs=envs)

        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.eval()

        observations = envs.reset()

        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rnn_states = torch.zeros(
            envs.num_envs,
            self.policy.net.num_recurrent_layers,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            envs.num_envs, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        stats_episodes = {}

        rgb_frames = [[] for _ in range(envs.num_envs)]
        rgb_frames_to_save = [[] for _ in range(envs.num_envs)]
        summary = [' ' for _ in range(envs.num_envs)]
        high_level_instruction = [' ' for _ in range(envs.num_envs)]

        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)

        pbar = tqdm.tqdm(total=num_eps) if config.use_pbar else None
        log_str = (
            f"[Ckpt: {checkpoint_index}]"
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        while envs.num_envs > 0 and len(stats_episodes) < num_eps:
            current_episodes = envs.current_episodes()

            with torch.no_grad():
                actions, rnn_states = self.policy.act(
                    batch,
                    rnn_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=not config.EVAL.SAMPLE,
                )
                prev_actions.copy_(actions)

            outputs = envs.step([a[0].item() for a in actions])
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            not_done_masks = torch.tensor(
                [[0] if done else [1] for done in dones],
                dtype=torch.uint8,
                device=self.device,
            )

            # reset envs and observations if necessary
            for i in range(envs.num_envs):
                frame = observations_to_image(observations[i], infos[i], append_depth=False, append_map=False)
                rgb_frames[i].append(frame)
                high_level_instruction[i] = current_episodes[i].instruction.high_level_instruction
                import ipdb; ipdb.set_trace()
                if config.EVAL.SAVE_VIDEO:
                    frame = observations_to_image(observations[i], infos[i])
                    _text = text_to_append(current_episodes[i].instruction)
                    frame = append_text_to_image(frame, _text)
                    rgb_frames_to_save[i].append(frame)
                
                if envs[i]._elapsed_steps % self.config.EVAL.MAX_STEPS_PER_SUBTASK:
                    next_instruction, summary[i] = self.find_next_instruction(
                        summary[i], 
                        rgb_frames[i], 
                        high_level_instruction[i])

                if not dones[i]:
                    continue

                ep_id = current_episodes[i].episode_id
                stats_episodes[ep_id] = infos[i]
                observations[i] = envs.reset_at(i)[0]
                prev_actions[i] = torch.zeros(1, dtype=torch.long)

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
                        images=rgb_frames_to_save[i],
                        episode_id=ep_id,
                        checkpoint_idx=checkpoint_index,
                        metrics={"spl": stats_episodes[ep_id]["spl_rxr"]},
                        tb_writer=writer,
                    )
                    del stats_episodes[ep_id]["top_down_map_vlnce"]
                    rgb_frames_to_save[i] = []
                rgb_frames[i], summary[i], high_level_instruction[i] = [], ' ', ' '

            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            (
                envs,
                rnn_states,
                not_done_masks,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                envs,
                rnn_states,
                not_done_masks,
                prev_actions,
                batch,
                rgb_frames,
            )

        envs.close()
        if config.use_pbar:
            pbar.close()

        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for k in next(iter(stats_episodes.values())).keys():
            if k == 'agent_rotation' or k == 'agent_position':
                continue
            else:
                aggregated_stats[k] = (
                    np.sum(v[k] for v in stats_episodes.values()) / num_episodes
                )

        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(aggregated_stats, f, indent=4)

        logger.info(f"Episodes evaluated: {num_episodes}")
        checkpoint_num = checkpoint_index + 1
        for k, v in aggregated_stats.items():
            logger.info(f"{k}: {v:.6f}")
            writer.add_scalar(f"eval_{split}_{k}", v, checkpoint_num)
    

    def load_vlm(self, name):
        if 'gemini' in name:
            from vlnce_baselines.common.utils import get_gemini_vision_model
            self.gemini_vlm = get_gemini_vision_model()
        elif 'blip' in name:
            from vlnce_baselines.common.utils import get_blip2_model
            self.blip2_processor, self.blip2_model = get_blip2_model()
        else:
            raise NotImplementedError(f" VLM: {name} not available.")

    def load_llm(self, name):
        if 'gemini' in name:
            from vlnce_baselines.common.utils import get_gemini_text_model
            self.gemini_llm = get_gemini_text_model()
        elif 'gpt' in name:
            pass
        else:
            raise NotImplementedError(f" LLM: {name} not available.")
    
    def llm_summary(self, summary_text):
        llm_summary_prompt = "A person made the following consecutive observations while " \
             "navigating a house. Very concisely describe the layout of the house seen so far "\
                 "based on the observations."
        
        if 'gemini' in self.ll_model:
            new_summary = self.gemini_llm.generate_content(f"{llm_summary_prompt} Observations: {summary_text}")
            new_summary.resolve()
        elif 'gpt' in self.ll_models:
            import openai
            messages=[
                {"role": "system", "content": llm_summary_prompt},
                {"role": "user", "content": f"Observations: '{summary_text}'"}
            ]
            new_summary = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
            new_summary = new_summary['choices'][0]['message']['content']
        
        return new_summary


    def summarize_video(self, prev_summary, video):
        describe_img_prompt = "Question: You are in an indoor environment." \
              "What do you see? Give only high level details. Answer:"
        
        video_description = [prev_summary]
        for i, im in enumerate(video):
            if i % self.video_subsample == 0:
                image = Image.fromarray(im).convert('RGB')
                if 'blip' in self.vl_model:
                    inputs = self.blip2_processor(image, text=describe_img_prompt, return_tensors="pt").to(self.device, torch.float16)
                    generated_ids = self.blip2_model.generate(**inputs, max_new_tokens=40)
                    generated_text = self.blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    video_description.append(generated_text)
                elif 'gemini' in self.vl_model:
                    generated_text = self.gemini_vlm.generate_content([describe_img_prompt, image])
                    generated_text.resolve()
                    video_description.append(generated_text.text)
        
        # Summarize episode
        video_description.append(" ' ")
        video_description_input = " ".join(video_description)
        new_summary = self.llm_summary(" ".join(video_description_input))
        return new_summary
    
    def get_next_instruction_from_context(self, current_img, high_level_instruction, summary):
        next_instruction_prompt = f"You are a robot navigating a house. "\
            "You get a high level instruction ({high_level_instruction}) "\
            "of where to go and a summary ({summary}) of what you have seen in the house so far. "\
            "The image shows where you are right now. To which visible landmark "\
            "should you move next to get nearer to high level goal."
        
        if 'blip' in self.vl_model:
            inputs = self.blip2_processor(
                current_img, 
                text=next_instruction_prompt, 
                return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.blip2_model.generate(**inputs, max_new_tokens=40)
            next_instruction = self.blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        elif 'gemini' in self.vl_model:
            next_instruction = self.gemini_vlm.generate_content([next_instruction_prompt, current_img])
            next_instruction.resolve()
            next_instruction = next_instruction.text
        return next_instruction

    def find_next_instruction(self, summary, video, high_level_instruction):
        new_summary = self.summarize_video(summary, video)
        next_instruction = self.get_next_instruction_from_context(video[-1], high_level_instruction, new_summary)
        return next_instruction, new_summary



