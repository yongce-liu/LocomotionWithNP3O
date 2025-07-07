def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class Policy:
    init_noise_std = 1.0
    continue_from_last_std = True
    scan_encoder_dims = None  # [128, 64, 32]
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    # priv_encoder_dims = [64, 20]
    priv_encoder_dims = []
    activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    # only for 'ActorCriticRecurrent':
    rnn_type = "lstm"
    rnn_hidden_size = 512
    rnn_num_layers = 1

    tanh_encoder_output = False
    num_costs = 3

    teacher_act = True
    imi_flag = True

import torch
class Normalization:
    class obs_scales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        height_measurements = 5.0
    
    class act_scales:
        action_scale = 0.25
        hip_scale_reduction = 0.5 
        # FR_hip_joint, FR_thigh_joint, FR_calf_joint
        # FL_hip_joint, FL_thigh_joint, FL_calf_joint
        # RR_hip_joint, RR_thigh_joint, RR_calf_joint
        # RL_hip_joint, RL_thigh_joint, RL_calf_joint
        dof_pos_clip = torch.tensor([[-0.837758, -3.49066, -2.72271,
                                      -0.837758, -3.49066, -2.72271,
                                      -0.837758, -4.53786, -2.72271,
                                      -0.837758, -4.53786, -2.72271],
                                     [0.837758, 1.5708, 0.837758,
                                      0.837758, 1.5708, 0.837758,
                                      0.837758, 0.523599, 0.837758,
                                      0.837758, 0.523599, 0.837758]]) * 0.9
    class ctrl_params:
        Kp = 50.0
        Kd = 2.0
    
    clip_observations = 100.0
    # clip_actions = 1.2
    clip_actions = 100
