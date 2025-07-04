import numpy as np
import argparse
import torch

torch.set_grad_enabled(False)

# Unitree Go2 SDK imports
from unitree_sdk2py.core import channel  # Pub, Sub, FactoryInitializer
from unitree_sdk2py.go2.sport import sport_client
import unitree_sdk2py.idl.unitree_go.msg.dds_ as unitree_msg_dds

# policy utils
from modules import ActorCriticBarlowTwins
from tmp.tools import class_to_dict, Policy, Normalization
from tmp.math import quat_apply_inverse


class trajectoryPublisher:
    def __init__(self):
        pass


class Dog:
    """
    Default Joint angles for the dog.
    unitree go2 sdk order:
    FR_hip_joint 0
    FR_thigh_joint 1
    FR_calf_joint 2
    FL_hip_joint 3
    FL_thigh_joint 4
    FL_calf_joint 5
    RR_hip_joint 6
    RR_thigh_joint 7
    RR_calf_joint 8
    RL_hip_joint 9
    RL_thigh_joint 10
    RL_calf_joint 11
    """

    def __init__(self, loaded_policy: ActorCriticBarlowTwins):
        self.policy = loaded_policy
        self.obs = np.zeros((self.policy.num_obs,))
        self.used_obs_idx = range(3, self.policy.num_prop)
        self.base_ang_vel = None
        self.base_quat = None
        self.dof_pos = None
        self.dof_vel = None
        self._init_default_vars()
        self._init_handlers()

    def _init_default_vars(self):
        """
        Initialize default variables for the dog.
        """
        self.gravity_vec = torch.tensor([0.0, 0.0, -1], dtype=torch.float32)
        self.obs_scales = Normalization.obs_scales
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
            ],
        )  # TODO change this

        self.default_dof_pos = torch.tensor(
            [
                -0.1,  # FR_hip_joint
                0.8,  # FR_thigh_joint
                -1.5,  # FR_calf_joint
                0.1,  # FL_hip_joint
                0.8,  # FL_thigh_joint
                -1.5,  # FL_calf_joint
                -0.1,  # RR_hip_joint
                1.0,  # RR_thigh_joint
                -1.5,  # RR_calf_joint
                0.1,  # RL_hip_joint
                1.0,  # RL_thigh_joint
                -1.5,  # RL_calf_joint
            ]
        )  # rad
        self.start_joint_angles = torch.tensor(
            [
                0.0,  # FR_hip_joint
                0.9,  # FR_thigh_joint
                -1.8,  # FR_calf_joint
                0.0,  # FL_hip_joint
                0.9,  # FL_thigh_joint
                -1.8,  # FL_calf_joint
                0.0,  # RR_hip_joint
                0.9,  # RR_thigh_joint
                -1.8,  # RR_calf_joint
                0.0,  # RL_hip_joint
                0.9,  # RL_thigh_joint
                -1.8,  # RL_calf_joint
            ]
        )

    def _init_handlers(self):
        """
        Initialize the handlers for the dog.
        This method is currently a placeholder and does not perform any actions.
        """
        self.low_state_suber = channel.ChannelSubscriber(
            "rt/lowstate", unitree_msg_dds.LowState_
        )
        self.low_state_suber.Init(self.LowStateHandler, 10)
        self.high_state_suber = channel.ChannelSubscriber(
            "rt/sportmodestate", unitree_msg_dds.SportModeState_
        )
        self.low_cmd_puber = channel.ChannelPublisher(
            "rt/lowcmd", unitree_msg_dds.LowCmd_
        )
    
    def LowStateHandler(self, msg: unitree_msg_dds.LowState_):
        imu_state = msg.imu_state
        self.base_ang_vel = imu_state.gyroscope
        self.base_quat = imu_state.quaternion
        motor_states = msg.motor_state
        self.dof_pos = [_s.q for _s in motor_states]
        self.dof_vel = [_s.dq for _s in motor_states]
        print("LowStateHandler called")

    def compute_obs(self, cmd: torch.Tensor):
        self.obs[self.used_obs_idx] = torch.cat(
            self.base_ang_vel * self.obs_scales.ang_vel,
            quat_apply_inverse(self.base_quat, self.gravity_vec),
            cmd[:3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
        )

    def get_act(self, cmd: torch.Tensor):
        """
        Get action from the policy based on the command and current observation.
        :param cmd: Command array
        :return: Action array
        """
        self.compute_obs(cmd)
        obs_tensor = (
            torch.tensor(self.obs, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.policy.device)
        )
        action = self.policy.act_teacher(obs_tensor)
        return action.cpu().numpy()[0]

    def _wrap_action(self, act):
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
    model_dict = torch.load(policy_path, map_location=torch.device(device), weights_only=False)
    policy.load_state_dict(model_dict["model_state_dict"])
    policy.half()
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
    test_dog = Dog(loaded_policy=policy)
    while True:
        pass
