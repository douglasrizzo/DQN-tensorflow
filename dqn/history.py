import numpy as np


class History:

    def __init__(self, config):
        batch_size = config.batch_size
        history_length = config.history_length
        screen_height = config.screen_height
        screen_width = config.screen_width

        self.history = np.zeros([history_length, screen_height, screen_width], dtype=np.float32)
        self.cnn_format = config.cnn_format

    def add(self, screen):
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen

    def reset(self):
        self.history *= 0

    def get(self):
        if self.cnn_format == 'NHWC':
            return np.transpose(self.history, (1, 2, 0))
        else:
            return self.history
