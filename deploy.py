import torch

torch.set_grad_enabled(False)
import argparse
import time

from unitree_sdk2py.core import channel  # Pub, Sub, FactoryInitializer
from modules import ActorCriticBarlowTwins
from utils.tools import class_to_dict, Policy
from utils.client import UnitreeGo2


class trajectoryPublisher:
    def __init__(self):
        pass


def load_policy(policy_path, device="cpu"):
    n_scan = 187
    n_priv_latent = 4 + 1 + 12 + 12 + 12 + 6 + 1 + 4 + 1 - 3 + 3 - 3 + 4 - 7
    n_proprio = 45 + 3
    history_len = 10
    n_actions = 12
    num_observations = n_proprio + n_scan + history_len * n_proprio + n_priv_latent

    _policy = Policy()
    policy_cfg_dict = class_to_dict(_policy)
    policy = ActorCriticBarlowTwins(
        num_prop=n_proprio,
        num_scan=n_scan,
        num_critic_obs=num_observations,
        num_priv_latent=n_priv_latent,
        num_hist=history_len,
        num_actions=n_actions,
        **policy_cfg_dict,
    )
    model_dict = torch.load(
        policy_path, map_location=torch.device(device), weights_only=False
    )
    policy.load_state_dict(model_dict["model_state_dict"])
    # policy.half()
    policy.eval()
    policy = policy.to(device)
    # policy.save_torch_jit_policy("model.pt", device)
    return policy


if __name__ == "__main__":
    # task, sim_device
    args = argparse.ArgumentParser()
    args.add_argument("--path", type=str, default="./checkpoint/model_10000.pt")
    args.add_argument("--device", type=str, default="cpu")
    args.add_argument("--address", type=str, default="lo")

    params = args.parse_args()
    policy = load_policy(params.path, device=params.device)
    channel.ChannelFactoryInitialize(1, params.address)
    test_dog = UnitreeGo2(loaded_policy=policy)
    while True:
        test_dog.step(torch.tensor([0.0, 0.0, 0.0]))
        time.sleep(0.01)
