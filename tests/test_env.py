import sys
sys.path.append('/home/sax1rng/Projects/VLN-CE-Plan')  

from habitat import Config, logger
from gym import Space
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.environments import get_env_class
from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false

def _get_spaces(config: Config, envs: Optional[Any] = None) -> Tuple[Space]:
        """Gets both the observation space and action space.

        Args:
            config (Config): The config specifies the observation transforms.
            envs (Any, optional): An existing Environment. If None, an
                environment is created using the config.

        Returns:
            observation space, action space
        """
        if envs is not None:
            observation_space = envs.observation_spaces[0]
            action_space = envs.action_spaces[0]

        else:
            env = get_env_class(config.ENV_NAME)(config=config)
            observation_space = env.observation_space
            action_space = env.action_space

        obs_transforms = get_active_obs_transforms(config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, obs_transforms
        )
        return observation_space, action_space

if __name__== "__main__":
    # Create config
    envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )
    observation_space, action_space = _get_spaces(config, envs=envs)

    observations = envs.reset()

    outputs = envs.step([a[0].item() for a in actions])

    observations, _, dones, infos = [list(x) for x in zip(*outputs)]

    for i in range(envs.num_envs):
        if config.EVAL.SAVE_VIDEO:
            frame = observations_to_image(observations[i], infos[i])
            _text = text_to_append(current_episodes[i].instruction)
            frame = append_text_to_image(frame, _text)
            rgb_frames[i].append(frame)
