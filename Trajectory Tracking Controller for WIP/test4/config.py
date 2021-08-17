class Config(object):
    batch_size = 32
    valid_size = 128
    logging_frequency = 50
    model_save_path = "./model.pth"
    max_grad_norm = 10000


class WIP_LINEARConfig(Config):
    total_time = 10
    delta_t = 0.02
    dim = 4
    lr_value = 1e-1
    num_iterations = 6000
    num_hiddens = [32, 128, 32]
    y_init_range = [10, 20]
    z_init_range = [-0.1, 0.1]
    zmax = 1
    umax = 9
    DELTA_CLIP = 50.0
    weight_decay = 1e-5

    lstm_num_layers = 1
    lstm_hidden_size = 16


def get_config(name):
    try:
        return globals()[name + 'Config']
    except KeyError:
        raise KeyError("Config for the required problem not found.")
