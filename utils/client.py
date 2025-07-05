import numpy as np
import torch
import time

# Unitree Go2 SDK imports
from unitree_sdk2py.core import channel  # Pub, Sub, FactoryInitializer
import unitree_sdk2py.idl.unitree_go.msg.dds_ as unitree_msg_dds
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

# policy utils
from modules import ActorCriticBarlowTwins
from .tools import Normalization
from .math import quat_apply_inverse


class UnitreeGo2:
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
        self.decimation = 10
        self.dt = 0.002
        self.obs = torch.zeros((self.policy.num_hist+1, self.policy.num_prop))
        self.obs_cur = torch.zeros((1, self.policy.num_prop))
        self.act_buf = torch.zeros((self.policy.num_actions,))
        self.base_ang_vel = None
        self.base_quat = None  # Quaternion [w, x, y, z]
        self.dof_pos = None
        self.dof_vel = None
        self.if_read_state = False  # Flag to check if state is read
        self._init_default_vars()
        self._init_handlers()
        while not self.if_read_state:
            time.sleep(0.01)

    def _init_default_vars(self):
        """
        Initialize default variables for the dog.
        """
        self.gravity_vec = torch.tensor([0.0, 0.0, -1], dtype=torch.float32)
        self.obs_scales = Normalization.obs_scales
        self.clip_actions = Normalization.clip_actions
        self.clip_obs = Normalization.clip_observations
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
            ],
        )  # TODO change this

        self.default_dof_pos = torch.tensor(
            [ # action when no action, action = 0.0
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
    #     self.default_dof_pos = torch.tensor([
    #             0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455,
    # 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375])
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

        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0
        for i in range(20):
            cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = 0.0
        self.cmd = cmd

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
        self.low_cmd_puber.Init()
        self.crc = CRC()

    def LowStateHandler(self, msg: unitree_msg_dds.LowState_):
        imu_state = msg.imu_state
        self.base_ang_vel: list = imu_state.gyroscope
        self.base_quat: list = imu_state.quaternion
        motor_states = msg.motor_state
        self.dof_pos: list = [motor_states[i].q for i in range(12)]
        self.dof_vel: list = [motor_states[i].dq for i in range(12)]
        self.if_read_state = True
        # print("LowStateHandler called")
        # print("Base Angular Velocity: {}".format(self.base_ang_vel))
        # print("Base Quaternion: {}".format(self.base_quat))
        # print("Degrees of Freedom Position: {}".format(self.dof_pos))
        # print("Degrees of Freedom Velocity: {}".format(self.dof_vel))
        # print(type(self.base_ang_vel), type(self.base_quat))
        # print(type(self.dof_pos), type(self.dof_vel))

    def update_obs(self, des_act: torch.tensor):
        # print(
        #     torch.tensor(self.base_ang_vel) * self.obs_scales.ang_vel,
        #     quat_apply_inverse(torch.tensor(self.base_quat), self.gravity_vec),
        #     cmd[:3] * self.commands_scale,
        #     (torch.tensor(self.dof_pos) - self.default_dof_pos)
        #     * self.obs_scales.dof_pos,
        #     torch.tensor(self.dof_vel) * self.obs_scales.dof_vel,
        #     self.act_buf,
        # )
        self.obs_cur[:, 3:] = torch.cat(
            (
                torch.tensor(self.base_ang_vel) * self.obs_scales.ang_vel,
                quat_apply_inverse(torch.tensor(self.base_quat), self.gravity_vec),
                des_act[:3] * self.commands_scale,
                (torch.tensor(self.dof_pos) - self.default_dof_pos)
                * self.obs_scales.dof_pos,
                torch.tensor(self.dof_vel) * self.obs_scales.dof_vel,
                self.act_buf,
            )
        )
        self.obs = torch.cat([self.obs_cur,
                            self.obs[:-1, :]])
        self.obs = torch.clip(
            self.obs,
            -self.clip_obs,
            self.clip_obs,
        )

    def step(self, des_act: torch.Tensor):
        """
        Get action from the policy based on the command and current observation.
        :param cmd: Command array
        :return: Action array
        """
        self.update_obs(des_act)
        action = self.policy.act_teacher(self.obs.view(-1).unsqueeze(0)).squeeze(0)
        self.act_buf = action
        action = torch.clip(
            action,
            -self.clip_actions,
            self.clip_actions,
        )
        print("Action: ", action)
        # action = self._compute_torques(action)
        self._act2cmd(action)
        self.cmd.crc = self.crc.Crc(self.cmd)
        for _ in range(self.decimation):
            self.low_cmd_puber.Write(self.cmd)
            time.sleep(self.dt)
        # return motor_cmd

    def _act2cmd(self, act):
        act = act.cpu().squeeze(0)
        for i in range(12):
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 50.0
            self.cmd.motor_cmd[i].dq = act[i]
            self.cmd.motor_cmd[i].kd = 5.0
            self.cmd.motor_cmd[i].tau = 0

    # def _compute_torques(self, actions):
    #     """ Compute torques from actions.
    #         Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
    #         [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

    #     Args:
    #         actions (torch.Tensor): Actions

    #     Returns:
    #         [torch.Tensor]: Torques sent to the simulation
    #     """
    #     if self.cfg.control.use_filter:
    #         actions = self._low_pass_action_filter(actions)

    #     #pd controller
    #     actions_scaled = actions[:, :12] * self.cfg.control.action_scale
    #     actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction

    #     # if self.cfg.domain_rand.randomize_lag_timesteps:
    #     #     self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
    #     #     joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
    #     # else:
    #     #     joint_pos_target = actions_scaled + self.default_dof_pos

    #     if self.cfg.domain_rand.randomize_lag_timesteps:
    #         self.lag_buffer = torch.cat([self.lag_buffer[:,1:,:].clone(),actions_scaled.unsqueeze(1).clone()],dim=1)
    #         joint_pos_target = self.lag_buffer[self.num_envs_indexes,self.randomized_lag,:] + self.default_dof_pos
    #     else:
    #         joint_pos_target = actions_scaled + self.default_dof_pos

    #     # joint_pos_target = torch.clamp(joint_pos_target,self.dof_pos-1,self.dof_pos+1)

    #     control_type = self.cfg.control.control_type
    #     if control_type=="P":
    #         if not self.cfg.domain_rand.randomize_kpkd:  # TODO add strength to gain directly
    #             torques = self.p_gains*(joint_pos_target- self.dof_pos) - self.d_gains*self.dof_vel
    #         else:
    #             torques = self.kp_factor * self.p_gains*(joint_pos_target - self.dof_pos) - self.kd_factor * self.d_gains*self.dof_vel
    #     elif control_type=="V":
    #         torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
    #     elif control_type=="T":
    #         torques = actions_scaled
    #     else:
    #         raise NameError(f"Unknown controller type: {control_type}")
        
    #     torques = torques * self.motor_strength
    #     return torch.clip(torques, -self.torque_limits, self.torque_limits)