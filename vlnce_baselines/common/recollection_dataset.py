import gzip
import json
from collections import defaultdict, deque

import numpy as np
import torch
import tqdm
import time
from itertools import islice

from gym import Space
from habitat.config.default import Config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)

from habitat_extensions.task import ALL_ROLES_MASK, RxRVLNCEDatasetV1
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import extract_instruction_tokens, int2bits

from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)

class TeacherRecollectionDataset(torch.utils.data.IterableDataset):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._preload = deque()

        self.envs = None
        self._env_observations = None

        if config.IL.use_iw:
            self.inflec_weights = torch.tensor(
                [1.0, config.IL.inflection_weight_coef]
            )
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        if self.config.IL.RECOLLECT_TRAINER.preload_trajectories_file:
            with gzip.open(
                config.IL.RECOLLECT_TRAINER.trajectories_file, "rt"
            ) as f:
                self.trajectories = json.load(f)
        else:
            self.trajectories = self.collect_dataset()

        self.initialize_sims()

    def initialize_sims(self):

        assert (
            self.config.IL.RECOLLECT_TRAINER.preload_size >= self.config.IL.batch_size
        ), "preload size must be greater than batch size."

        config = self.config.clone()
        config.defrost()
        config.TASK_CONFIG.MEASUREMENTS = []
        config.freeze()

        self.envs = construct_envs(
            config,
            get_env_class(config.ENV_NAME),
            episodes_allowed=list(self.trajectories.keys()),
        )
        self.length = sum(self.envs.number_of_episodes)
        self.obs_transforms = get_active_obs_transforms(self.config)
        self._observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], self.obs_transforms
        )
        self.env_step = [0 for _ in range(self.envs.num_envs)]
        self._env_observations = [[] for _ in range(self.envs.num_envs)]

        observations = self.envs.reset()
        observations = extract_instruction_tokens(
            observations,
            self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
        )
        for i, ep in enumerate(self.envs.current_episodes()):
            path_step = self.trajectories[ep.episode_id][0]
            self._env_observations[i].append(
                (
                    observations[i],
                    path_step[0],  # prev_action
                    path_step[2],  # oracle_action
                )
            )

    @property
    def batch_size(self):
        return self.config.IL.batch_size

    @property
    def observation_space(self) -> Space:
        assert self.envs is not None, "Simulator must first be loaded."
        assert self._observation_space is not None
        return self._observation_space

    @property
    def action_space(self) -> Space:
        assert self.envs is not None, "Simulator must first be loaded."
        return self.envs.action_spaces[0]

    def close_sims(self):
        self.envs.close()
        del self.envs
        del self._env_observations
        self.envs = None
        self._env_observations = None

    def collect_dataset(self):
        """Uses the ground truth trajectories to create a teacher forcing
        datset for a given split. Loads both guide and follower episodes.
        """
        trajectories = defaultdict(list)
        split = self.config.TASK_CONFIG.DATASET.SPLIT

        if "{role}" in self.config.IL.RECOLLECT_TRAINER.gt_file:
            gt_data = {}
            for role in RxRVLNCEDatasetV1.annotation_roles:
                if (
                    ALL_ROLES_MASK not in self.config.TASK_CONFIG.DATASET.ROLES
                    and role not in self.config.TASK_CONFIG.DATASET.ROLES
                ):
                    continue

                with gzip.open(
                    self.config.IL.RECOLLECT_TRAINER.gt_file.format(
                        split=split, role=role
                    ),
                    "rt",
                ) as f:
                    gt_data.update(json.load(f))
        else:
            with gzip.open(
                self.config.IL.RECOLLECT_TRAINER.gt_path.format(split=split)
            ) as f:
                gt_data = json.load(f)

        t = (
            tqdm.tqdm(gt_data.items(), "GT Collection")
            if self.config.use_pbar
            else gt_data.items()
        )

        for episode_id, trajectory in t:
            if (
                self.config.IL.RECOLLECT_TRAINER.max_traj_len != -1
                and len(trajectory["actions"])
                > self.config.IL.RECOLLECT_TRAINER.max_traj_len
            ):
                continue

            for i, action in enumerate(trajectory["actions"]):
                prev_action = (
                    trajectories[episode_id][i - 1][1]
                    if i
                    else HabitatSimActions.STOP
                )

                # [prev_action, action, oracle_action]
                trajectories[episode_id].append([prev_action, action, action])

        with gzip.open(
            self.config.IL.RECOLLECT_TRAINER.trajectories_file, "wt"
        ) as f:
            f.write(json.dumps(trajectories))
        return trajectories

    def _load_next(self):
        """
        Episode length is currently not considered. We were previously batching episodes
        together with similar lengths. Not sure if we need to bring that back.
        """

        if len(self._preload):
            return self._preload.popleft()

        while (
            len(self._preload) < self.config.IL.RECOLLECT_TRAINER.preload_size
        ):
            current_episodes = self.envs.current_episodes()
            prev_eps = current_episodes

            # get the next action for each env
            actions = [
                self.trajectories[ep.episode_id][self.env_step[i]][1]
                for i, ep in enumerate(current_episodes)
            ]

            outputs = self.envs.step(actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            
            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )

            current_episodes = self.envs.current_episodes()

            for i in range(self.envs.num_envs):
                self.env_step[i] += 1
                if dones[i]:
                    assert len(self._env_observations[i]) == len(
                        self.trajectories[prev_eps[i].episode_id]
                    ), "Collected episode does not match the step count of trajectory"
                    self._preload.append(
                        (
                            [o[0] for o in self._env_observations[i]],
                            [o[1] for o in self._env_observations[i]],
                            [o[2] for o in self._env_observations[i]],
                        )
                    )
                    self._env_observations[i] = []
                    self.env_step[i] = 0

                path_step = self.trajectories[current_episodes[i].episode_id][
                    self.env_step[i]
                ]
                self._env_observations[i].append(
                    (
                        observations[i],
                        path_step[0],  # prev_action
                        path_step[2],  # oracle_action
                    )
                )
                assert (
                    len(self._env_observations[i])
                    <= self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
                ), "Trajectories should be no more than the maximum episode steps."

        return self._preload.popleft()

    def __next__(self):
        """Takes about 1s to once self._load_next() has finished with a batch
        size of 5. For this reason, we probably don't need to use extra workers.
        """
        x = self._load_next()
        obs, prev_actions, oracle_actions = x

        # transpose obs
        obs_t = defaultdict(list)
        for k in obs[0]:
            for i in range(len(obs)):
                obs_t[k].append(obs[i][k])

            obs_t[k] = np.array(obs_t[k])

        for k, v in obs_t.items():
            obs_t[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs_t,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            assert (
                worker_info.num_workers == 1
            ), "multiple workers not supported."

        return self

class OctoTeacherRecollectionDataset(TeacherRecollectionDataset):
    def __init__(self, config: Config, octo_config):
        self.octo_config = octo_config
        self.action_repr = octo_config.model.heads.action.kwargs.action_repr
        super().__init__(config)

    def collect_dataset(self):
        """Uses the ground truth trajectories to create a teacher forcing
        datset for a given split. Loads both guide and follower episodes.
        """
        trajectories = defaultdict(list)
        split = self.config.TASK_CONFIG.DATASET.SPLIT

        if "{role}" in self.config.IL.OCTO_TRAINER.gt_file:
            gt_data = {}
            for role in RxRVLNCEDatasetV1.annotation_roles:
                if (
                    ALL_ROLES_MASK not in self.config.TASK_CONFIG.DATASET.ROLES
                    and role not in self.config.TASK_CONFIG.DATASET.ROLES
                ):
                    continue

                with gzip.open(
                    self.config.IL.OCTO_TRAINER.gt_file.format(
                        split=split, role=role
                    ),
                    "rt",
                ) as f:
                    gt_data.update(json.load(f))
        else:
            with gzip.open(
                self.config.IL.OCTO_TRAINER.gt_path.format(split=split)
            ) as f:
                gt_data = json.load(f)

        t = (
            tqdm.tqdm(gt_data.items(), "GT Collection")
            if self.config.use_pbar
            else gt_data.items()
        )

        for episode_id, trajectory in t:
            if (
                self.config.IL.OCTO_TRAINER.max_traj_len != -1
                and len(trajectory["actions"])
                > self.config.IL.OCTO_TRAINER.max_traj_len
            ):
                continue

            for i, action in enumerate(trajectory["actions"]):
                prev_action = (
                    trajectories[episode_id][i - 1][1]
                    if i
                    else HabitatSimActions.STOP
                )

                # [prev_action, action, oracle_action]
                trajectories[episode_id].append([prev_action, action, action])

        with gzip.open(
            self.config.IL.OCTO_TRAINER.trajectories_file, "wt"
        ) as f:
            f.write(json.dumps(trajectories))
        return trajectories
    
    def initialize_sims(self):

        assert (
            self.config.IL.OCTO_TRAINER.preload_size >= self.config.IL.batch_size
        ), "preload size must be greater than batch size."

        config = self.config.clone()
        config.defrost()
        config.TASK_CONFIG.MEASUREMENTS = []
        config.freeze()

        # temporary
        # split = self.config.TASK_CONFIG.DATASET.SPLIT
        # if "{role}" in self.config.TASK_CONFIG.DATASET.DATA_PATH:
        #     data = {}
        #     for role in RxRVLNCEDatasetV1.annotation_roles:
        #         if (
        #             ALL_ROLES_MASK not in self.config.TASK_CONFIG.DATASET.ROLES
        #             and role not in self.config.TASK_CONFIG.DATASET.ROLES
        #         ):
        #             continue

        #         with gzip.open(
        #             self.config.TASK_CONFIG.DATASET.DATA_PATH.format(
        #                 split=split, role=role
        #             ),
        #             "rt",
        #         ) as f:
        #             data.update(json.load(f))
        # dataset_size = len(data['episodes'])
        # episodes_allowed = [data['episodes'][i]['episode_id'] for i in range(dataset_size) if data['episodes'][0]['scene_id'] in data['episodes'][i]['scene_id']]
        
        episodes_allowed = list(self.trajectories.keys())[:54312]
        self.envs = construct_envs(
            config,
            get_env_class(config.ENV_NAME),
            episodes_allowed=episodes_allowed,
        )

        self.length = sum(self.envs.number_of_episodes)
        self.obs_transforms = get_active_obs_transforms(self.config)
        self._observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], self.obs_transforms
        )
        self.env_step = [0 for _ in range(self.envs.num_envs)]
        self._env_observations = [[] for _ in range(self.envs.num_envs)]

        observations = self.envs.reset()
        observations = extract_instruction_tokens(
            observations,
            self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
        )

        self.actions = {} 
        for ep_id in self.trajectories.keys():
            trajectory = np.array(self.trajectories[ep_id])
            if 'one-hot' in self.action_repr:
                one_hot_trajectory = np.zeros((trajectory.shape[0],6), dtype=np.float32)
                one_hot_trajectory[np.arange(trajectory.shape[0]), trajectory[:,1]] = 1
                self.actions[ep_id] = one_hot_trajectory
            elif 'discrete' in self.action_repr:
                self.actions[ep_id] = trajectory[:,1]
            elif 'bits' in  self.action_repr:
                nbits = self.octo_config.model.heads.action.kwargs.action_dim
                self.actions[ep_id] = (int2bits(trajectory[:,1], nbits) * 2 - 1) * self.config.IL.OCTO_TRAINER.scale_bits
            else:
                raise NotImplementedError('action representation not defined')

        for i, ep in enumerate(self.envs.current_episodes()):
            path_step = self.trajectories[ep.episode_id][0]
            
            self._env_observations[i].append(
                (
                    observations[i],
                    path_step[0],  # prev_action
                    path_step[2],  # oracle_action
                    self.actions[ep.episode_id][0],  # oracle_action
                    ep.instruction.instruction_text
                )
            )

    def _load_next(self):
        """
        Episode length is currently not considered. We were previously batching episodes
        together with similar lengths. Not sure if we need to bring that back.
        """

        if len(self._preload):
            return self._preload.popleft()

        while (
            len(self._preload) < self.config.IL.OCTO_TRAINER.preload_size
        ):
            current_episodes = self.envs.current_episodes()
            prev_eps = current_episodes

            # get the next action for each env
            actions = [
                self.trajectories[ep.episode_id][self.env_step[i]][1]
                for i, ep in enumerate(current_episodes)
            ]

            outputs = self.envs.step(actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            
            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )

            current_episodes = self.envs.current_episodes()

            for i in range(self.envs.num_envs):
                self.env_step[i] += 1
                if dones[i]:
                    assert len(self._env_observations[i]) == len(
                        self.trajectories[prev_eps[i].episode_id]
                    ), "Collected episode does not match the step count of trajectory"
                    self._preload.append(
                        (
                            [o[0] for o in self._env_observations[i]],
                            [o[1] for o in self._env_observations[i]],
                            [o[2] for o in self._env_observations[i]],
                            [o[3] for o in self._env_observations[i]],
                            [o[4] for o in self._env_observations[i]],
                        )
                    )
                    self._env_observations[i] = []
                    self.env_step[i] = 0

                path_step = self.trajectories[current_episodes[i].episode_id][
                    self.env_step[i]
                ]

                self._env_observations[i].append(
                    (
                        observations[i],
                        path_step[0],  # prev_action
                        path_step[2],  # oracle_action
                        self.actions[current_episodes[i].episode_id][self.env_step[i]],
                        current_episodes[i].instruction.instruction_text
                    )
                )
                assert (
                    len(self._env_observations[i])
                    <= self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
                ), "Trajectories should be no more than the maximum episode steps."

        return self._preload.popleft()

    def __next__(self):
        """Takes about 1s to once self._load_next() has finished with a batch
        size of 5. For this reason, we probably don't need to use extra workers.
        """
        x = self._load_next()
        obs, prev_actions, oracle_actions, one_hot_action, instructions = x

        # transpose obs
        obs_t = defaultdict(list)
        for k in obs[0]:
            for i in range(len(obs)):
                obs_t[k].append(obs[i][k])

            obs_t[k] = np.array(obs_t[k])

        for k, v in obs_t.items():
            obs_t[k] = torch.from_numpy(np.copy(v))

        obs_t['rxr_instruction'] = obs_t['rxr_instruction'][:,:self.octo_config.bert_max_tokens,:]
        
        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))
        one_hot_action = torch.from_numpy(np.copy(one_hot_action)).to(torch.float32)

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs_t,
            prev_actions,
            oracle_actions,
            one_hot_action,
            self.inflec_weights[inflections],
            instructions,
        )
    
class OctoTimeStepsTeacherRecollectionDataset(OctoTeacherRecollectionDataset):
    def __init__(self, config: Config, octo_config):
        super().__init__(config, octo_config)
        self.rgb = [[] for _ in range(self.envs.num_envs)]
        self.depth = [[] for _ in range(self.envs.num_envs)]
        self.bert_features = [[] for _ in range(self.envs.num_envs)]
        self.prev_actions = [[] for _ in range(self.envs.num_envs)]
        self.oracle_actions = [[] for _ in range(self.envs.num_envs)]
        self.one_hot_action = [[] for _ in range(self.envs.num_envs)]
        self.instructions = [[] for _ in range(self.envs.num_envs)]
        for i, ep in enumerate(self.envs.current_episodes()):
            env_obs = self._env_observations[i][0]
            self.rgb[i].append(env_obs[0]['rgb'])
            self.depth[i].append(env_obs[0]['depth'])
            self.bert_features[i].append(env_obs[0]['rxr_instruction'])
            self.prev_actions[i].append(env_obs[1])
            self.oracle_actions[i].append(env_obs[2])
            self.one_hot_action[i].append(env_obs[3])
            self.instructions[i].append(env_obs[4])
        
        self.length = 0
        for key, value in self.trajectories.items():
            self.length += len(value)
            
    def add_obsi_to_queue(self, i):
        traj_len = len(self.rgb[i]) - (self.octo_config.window_size - 1)
        curr_step = np.arange(traj_len)
        obs_offset = np.arange(self.octo_config.window_size)
        chunk_indices = curr_step[:, None] + obs_offset[None, :]

        rgb = np.array(self.rgb[i])[chunk_indices]
        depth =np.array(self.depth[i])[chunk_indices]
        rxr_instruction = np.array(self.bert_features[i])[:traj_len,:self.octo_config.bert_max_tokens,:]

        action_offset = np.arange(self.octo_config.window_size + self.octo_config.pred_horizon - 1)
        chunk_indices = np.minimum(
            len(self.rgb[i])-1,
            (curr_step[:, None] + action_offset[None, :])
        ) # repeat last action which should be STOP action
        prev_actions = np.array(self.prev_actions[i])[chunk_indices]
        oracle_actions = np.array(self.oracle_actions[i])[chunk_indices]
        one_hot_action = np.array(self.one_hot_action[i])[chunk_indices]
        instructions = self.instructions[i][:traj_len]

        self._preload.extend(
            [
                (
                    rgb[t],
                    depth[t],
                    rxr_instruction[t],
                    prev_actions[t],
                    oracle_actions[t],
                    one_hot_action[t],
                    instructions[t]
                ) for t in range(traj_len)
            ]
        )
        # self._preload.extend(list(zip(
        #     rgb,
        #     depth,
        #     rxr_instruction,
        #     prev_actions,
        #     oracle_actions,
        #     one_hot_action,
        #     instructions
        # )))
        # for t in range(traj_len):
        #     self._preload.append(
        #         (
        #             rgb[t],
        #             depth[t],
        #             rxr_instruction[t],
        #             prev_actions[t],
        #             oracle_actions[t],
        #             one_hot_action[t],
        #             instructions[t]
        #         )
        #     )

    def add_obsi_to_queue_torch(self, i):
        traj_len = len(self.rgb[i]) - (self.octo_config.window_size - 1)
        curr_step = torch.arange(traj_len)
        obs_offset = torch.arange(self.octo_config.window_size)
        chunk_indices = curr_step[:, None] + obs_offset[None, :]

        rgb = torch.from_numpy(np.array(self.rgb[i]))[chunk_indices].to(torch.float32)
        depth = torch.from_numpy(np.array(self.depth[i]))[chunk_indices].to(torch.float32)
        rxr_instruction = torch.from_numpy(np.array(self.bert_features[i])[:traj_len,:self.octo_config.bert_max_tokens,:]).to(torch.float32)

        action_offset = torch.arange(self.octo_config.window_size + self.octo_config.pred_horizon - 1)
        chunk_indices = torch.minimum(
            torch.tensor(len(self.rgb[i])-1),
            (curr_step[:, None] + action_offset[None, :])
        ) # repeat last action which should be STOP action
        prev_actions = torch.from_numpy(np.array(self.prev_actions[i]))[chunk_indices].to(torch.float32)
        oracle_actions = torch.from_numpy(np.array(self.oracle_actions[i]))[chunk_indices].to(torch.float32)
        one_hot_action = torch.from_numpy(np.array(self.one_hot_action[i]))[chunk_indices].to(torch.float32)
        instructions = self.instructions[i][:traj_len]

        self._preload.extend(
            [
                (
                    rgb[t],
                    depth[t],
                    rxr_instruction[t],
                    prev_actions[t],
                    oracle_actions[t],
                    one_hot_action[t],
                    instructions[t]
                ) for t in range(traj_len)
            ]
        )
        # self._preload.extend(list(zip(
        #     rgb,
        #     depth,
        #     rxr_instruction,
        #     prev_actions,
        #     oracle_actions,
        #     one_hot_action,
        #     instructions
        # )))
        # for t in range(traj_len):
        #     self._preload.append(
        #         (
        #             rgb[t],
        #             depth[t],
        #             rxr_instruction[t],
        #             prev_actions[t],
        #             oracle_actions[t],
        #             one_hot_action[t],
        #             instructions[t]
        #         )
        #     )

    def _pop_batch(self):
        # Since num_workers > 0 is not implemented yet, checking if popping larger slices is faster
        # sample_batch = list(islice(self._preload, 0, self.config.IL.batch_size))
        # self._preload = deque(islice(self._preload, self.config.IL.batch_size, None))
        sample_batch = self._preload.popleft()
        return sample_batch
        
    def _load_next(self):

        if len(self._preload) > self.config.IL.batch_size:
            return self._pop_batch()

        # add_queue_time = 0.
        # preload_time = time.time()
        while (
            len(self._preload) < self.config.IL.OCTO_TRAINER.preload_size
        ):
            current_episodes = self.envs.current_episodes()
            prev_eps = current_episodes

            # get the next action for each env
            actions = [
                self.trajectories[ep.episode_id][self.env_step[i]][1]
                for i, ep in enumerate(current_episodes)
            ]

            outputs = self.envs.step(actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            
            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )

            current_episodes = self.envs.current_episodes()

            for i in range(self.envs.num_envs):
                self.env_step[i] += 1
                if dones[i]:
                    assert len(self.prev_actions[i]) == len(
                        self.trajectories[prev_eps[i].episode_id]
                    ), "Collected episode does not match the step count of trajectory"
                    # start = time.time()
                    self.add_obsi_to_queue_torch(i)
                    # add_queue_time += time.time()-start

                    self.rgb[i] = []
                    self.depth[i] = []
                    self.bert_features[i] = []
                    self.prev_actions[i] = []
                    self.oracle_actions[i] = []
                    self.one_hot_action[i] = []
                    self.instructions[i] = []
                    self.env_step[i] = 0

                path_step = self.trajectories[current_episodes[i].episode_id][
                    self.env_step[i]
                ]
               
                self.rgb[i].append(observations[i]['rgb'])
                self.depth[i].append(observations[i]['depth'])
                self.bert_features[i].append(observations[i]['rxr_instruction'])
                self.prev_actions[i].append(path_step[0])
                self.oracle_actions[i].append(path_step[2])
                self.one_hot_action[i].append(self.actions[current_episodes[i].episode_id][self.env_step[i]])
                self.instructions[i].append(current_episodes[i].instruction.instruction_text)
                assert (
                    len(self.prev_actions[i])
                    <= self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
                ), "Trajectories should be no more than the maximum episode steps."
        
        # print("Add to queue time: ", add_queue_time)
        # print("Preload time: ", time.time()-preload_time)
        return self._pop_batch()
    
    def __next__(self):
        return self._load_next()
    
    # @property
    # def batch_size(self):
    #     return 1