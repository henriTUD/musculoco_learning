from mushroom_rl.policy import Policy
import numpy as np


class RandomGaussianPolicy(Policy):

    def __init__(self, a_shape, mu=0.0, std=0.8):
        self.a_shape = a_shape
        self.mu = mu
        self.std = std

    def __call__(self, *args):
        raise NotImplementedError

    def draw_action(self, state):
        return np.random.normal(self.mu, self.std, self.a_shape)
