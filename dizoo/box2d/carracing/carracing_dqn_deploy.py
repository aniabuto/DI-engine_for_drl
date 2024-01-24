import gym
import torch
from easydict import EasyDict

from ding.config import compile_config
from ding.envs import DingEnvWrapper
from ding.model import DQN
from ding.policy import DQNPolicy, single_env_forward_wrapper
from dizoo.box2d.carracing.config.carracing_qrdqn_config import create_config, main_config
from dizoo.box2d.carracing.envs.carracing_env import CarRacingEnv


def main(main_config: EasyDict, create_config: EasyDict, ckpt_path: str):
    main_config.exp_name = f'carracing_qrdqn_seed0_deploy'
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    env = CarRacingEnv(cfg.env)
    env.enable_save_replay(replay_path=f'./{main_config.exp_name}/video')
    model = DQN(**cfg.policy.model)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    policy = DQNPolicy(cfg.policy, model=model).eval_mode
    forward_fn = single_env_forward_wrapper(policy.forward)
    obs = env.reset()
    returns = 0.
    while True:
        action = forward_fn(obs)
        obs, rew, done, info = env.step(action)
        returns += rew
        if done:
            break
    print(f'Deploy is finished, final episode return is: {returns}')


if __name__ == "__main__":
    main(
        main_config=main_config,
        create_config=create_config,
        ckpt_path=f'./carracing_qrdqn_seed0/ckpt/ckpt_best.pth.tar'
    )
