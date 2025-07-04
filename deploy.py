import numpy as np
import torch

# Unitree Go2 SDK imports
from unitree_sdk2py.core import channel  # Pub, Sub, FactoryInitializer
from unitree_sdk2py.go2.sport import sport_client

# policy utils
from modules import ActorCriticRMA
from tools.utils import Policy, class_to_dict

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


def load_policy(policy_path, device="cpu"):
    n_scan = 187
    n_priv_latent =  4 + 1 + 12 + 12 + 12 + 6 + 1 + 4 + 1 - 3 + 3 - 3 + 4 - 7
    n_proprio = 45 + 3
    history_len = 10
    n_actions = 12
    num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent

    _policy = Policy()
    policy_cfg_dict = class_to_dict(_policy)
    policy= ActorCriticRMA(
        num_prop = n_proprio,
        num_scan = n_scan,
        num_critic_obs = num_observations,
        num_priv_latent = n_priv_latent,
        num_hist = history_len,
        num_actions = n_actions,
        **policy_cfg_dict,
    )
    model_dict = torch.load(policy_path)
    policy.load_state_dict(model_dict["model_state_dict"])
    policy.half()
    policy.eval()
    policy = policy.to(device)
    policy.save_torch_jit_policy("model.pt", device)
    return policy


if __name__ == "__main__":
    # task, sim_device
    policy= load_policy("./checkpoint/model_10000.pt")

    test_dog = Dog(loaded_policy=policy)
