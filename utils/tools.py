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


class Normalization:
    class obs_scales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        height_measurements = 5.0

    clip_observations = 100.0
    # clip_actions = 1.2
    clip_actions = 100
