import pytest
import numpy as np
from easydict import EasyDict
from carracing_env import CarRacingEnv


# @pytest.mark.envtest
# @pytest.mark.parametrize('cfg', [EasyDict({'env_id': 'CarRacing-v2', 'continuous': False, 'act_scale': False,
#                                            'replay_path': None, 'save_replay_gif': False,
#                                            'replay_path_gif': None, 'action_clip': False})])
# class TestCarRacing:
#
#     def test_naive(self, cfg):
#         env = CarRacingEnv(cfg)
#         env.seed(314)
#         assert env._seed == 314
#         obs = env.reset()
#         assert obs.shape == (3, 96, 96)
#         for i in range(10):
#             random_action = env.random_action()
#             timestep = env.step(random_action)
#             print(timestep)
#             assert isinstance(timestep.obs, np.ndarray)
#             assert isinstance(timestep.done, bool)
#             assert timestep.obs.shape == (3, 96, 96)
#             assert timestep.reward.shape == (1, )
#             assert timestep.reward >= env.reward_space.low
#             assert timestep.reward <= env.reward_space.high
#         print(env.observation_space, env.action_space, env.reward_space)
#         env.close()

import os
import gym
from tensorboardX import SummaryWriter
from easydict import EasyDict
from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs import SyncSubprocessEnvManager, DingEnvWrapper, BaseEnvManager
from ding.envs.env_wrappers import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    EvalEpisodeReturnWrapper
from ding.policy import QRDQNPolicy
from ding.model import QRDQN
from ding.utils import set_pkg_seed

from ding.config import compile_config
from ding.data import DequeBuffer
from ding.envs import BaseEnvManagerV2, DingEnvWrapper
from ding.framework import ding_init, task
from ding.framework.context import OnlineRLContext
from ding.rl_utils import get_epsilon_greedy_fn
from dizoo.box2d.carracing.config.carracing_qrdqn_config import main_config
from ding.framework.middleware import CkptSaver, OffPolicyLearner, StepCollector, data_pusher, eps_greedy_handler, \
    interaction_evaluator, online_logger
from ditk import logging


def main():
    filename = '{}/log.txt'.format(main_config.exp_name)
    logging.getLogger(with_files=[filename]).setLevel(logging.INFO)

    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    ding_init(cfg)

    collector_env = BaseEnvManagerV2(
        env_fn=[lambda: CarRacingEnv(cfg.env) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
    )
    evaluator_env = BaseEnvManagerV2(
        env_fn=[lambda: CarRacingEnv(cfg.env) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
    )

    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    model = QRDQN(**cfg.policy.model)
    buffer = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
    policy = QRDQNPolicy(cfg.policy, model=model)

    with task.start(async_mode=False, ctx=OnlineRLContext()):
        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(data_pusher(cfg, buffer))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer))
        task.use(online_logger(train_show_freq=10))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=1000))
        task.run()


if __name__ == '__main__':
    main()