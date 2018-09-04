class AgentConfig(object):
    scale = 10000
    display = False

    max_step = 5000 * scale

    # TODO use psutil do get total RAM and ndarray.nbytes and try
    # to calculate the maximum size of the replay memory

    # this is for gym
    # memory_size = 100 * scale
    # this is for marlo
    memory_size = 10 * scale

    batch_size = 32
    random_start = 30
    cnn_format = 'NCHW'
    discount = 0.99
    target_q_update_step = 1 * scale
    learning_rate = 0.00025
    learning_rate_minimum = 0.00025
    learning_rate_decay = 0.96
    learning_rate_decay_step = 5 * scale

    ep_end = 0.1
    ep_start = 1.
    ep_end_t = memory_size

    history_length = 4
    train_frequency = 4
    learn_start = 5. * scale

    min_delta = -1
    max_delta = 1

    double_q = False
    dueling = False

    _test_step = 5 * scale
    _save_step = _test_step * 10


class EnvironmentConfig(object):

    def __init__(self, env_name, screen_dim, max_reward=1., min_reward=-1.):
        self.env_name = env_name
        self.screen_width = screen_dim
        self.screen_height = screen_dim
        self.max_reward = max_reward
        self.min_reward = min_reward


class DQNConfig(AgentConfig, EnvironmentConfig):
    model = ''
    backend = 'tf'
    env_type = 'detail'
    action_repeat = 1


def get_config(FLAGS):
    if FLAGS.marlo:
        config = DQNConfig('MarLo-FindTheGoal-v0', 400)
    else:
        config = DQNConfig('Breakout-v0', 84)

    if FLAGS.use_gpu:
        config.cnn_format = 'NHWC'
    else:
        config.cnn_format = 'NCHW'

    return config
