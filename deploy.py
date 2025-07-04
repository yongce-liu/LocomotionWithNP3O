import numpy as np
import torch

# Unitree Go2 SDK imports
from unitree_sdk2py.core import channel  # Pub, Sub, FactoryInitializer
from unitree_sdk2py.go2.sport import sport_client

# policy utils
from utils.helpers import class_to_dict

# rl-training configs
from utils.task_registry import task_registry
from configs.go2_constraint_him import (
    Go2ConstraintHimRoughCfg,
    Go2ConstraintHimRoughCfgPPO,
)
from envs import LeggedRobot
from modules import ActorCriticRMA
from utils import get_args


class trajectoryPublisher:
    def __init__(self):
        pass


class Dog:
    def __init__(self, loaded_policy: ActorCriticRMA):
        self.policy = loaded_policy

    def command(self):
        actions = self.policy.act_teacher(self.obs)
        pass

    def main_loop(self):
        pass


def load_policy(args, policy_path):
    task_registry.register(
        name=args.task,
        task_class=LeggedRobot,
        env_cfg=Go2ConstraintHimRoughCfg(),
        train_cfg=Go2ConstraintHimRoughCfgPPO(),
    )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    policy_cfg_dict = class_to_dict(train_cfg.policy)
    runner_cfg_dict = class_to_dict(train_cfg.runner)
    actor_critic_class = eval(runner_cfg_dict["policy_class_name"])
    policy: ActorCriticRMA = actor_critic_class(
        env.cfg.env.n_proprio,
        env.cfg.env.n_scan,
        env.num_obs,
        env.cfg.env.n_priv_latent,
        env.cfg.env.history_len,
        env.num_actions,
        **policy_cfg_dict,
    )
    model_dict = torch.load(policy_path)
    policy.load_state_dict(model_dict["model_state_dict"])
    policy.half()
    policy.eval()
    policy = policy.to(env.device)
    policy.save_torch_jit_policy("model.pt", env.device)
    return policy


if __name__ == "__main__":
    args = get_args()
    # task, sim_device
    policy: ActorCriticRMA = load_policy(args, "./checkpoint/model_10000.pt")

    test_dog = Dog(loaded_policy=policy)
