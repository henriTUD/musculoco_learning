
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.policy.torch_policy import GaussianTorchPolicy, TorchPolicy

from itertools import chain

import numpy as np

import torch
import torch.nn as nn


class OptionalGaussianTorchPolicy(GaussianTorchPolicy):
    """
    Torch policy implementing a Gaussian policy with trainable standard
    deviation. The standard deviation is not state-dependent.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.deterministic = False

        self._add_save_attr(
            deterministic='primitive',
        )

    def draw_action_t(self, state):
        if self.deterministic:
            return self._mu(state, **self._predict_params, output_tensor=True)
        else:
            return self.distribution_t(state).sample().detach()

    def get_as_normal_dist(self, state):
        mu = self._mu(state, **self._predict_params, output_tensor=True)
        stds = torch.exp(self._log_sigma)
        return torch.distributions.Normal(mu, stds)


class FixedStdGaussianTorchPolicy(TorchPolicy):

    def __init__(self, network, input_shape, output_shape, std_0=1.,
                 use_cuda=False, **params):
        super().__init__(use_cuda)

        self._action_dim = output_shape[0]

        self._mu = Regressor(TorchApproximator, input_shape, output_shape,
                             network=network, use_cuda=use_cuda, **params)
        self._predict_params = dict()

        log_sigma_init = (torch.ones(self._action_dim) * np.log(std_0)).float()

        if self._use_cuda:
            log_sigma_init = log_sigma_init.cuda()

        self._log_sigma = log_sigma_init

        self.deterministic = False

        self._add_save_attr(
            _action_dim='primitive',
            _mu='mushroom',
            _predict_params='pickle',
            _log_sigma='torch',
            deterministic='primitive',
        )

    def draw_action_t(self, state):
        if self.deterministic:
            return self._mu(state, **self._predict_params, output_tensor=True).detach()
        else:
            return self.distribution_t(state).sample().detach()

    def log_prob_t(self, state, action):
        return self.distribution_t(state).log_prob(action)[:, None]

    def entropy_t(self, state=None):
        return self._action_dim / 2 * np.log(2 * np.pi * np.e) + torch.sum(self._log_sigma)

    def distribution_t(self, state):
        mu, chol_sigma = self.get_mean_and_chol(state)
        return torch.distributions.MultivariateNormal(loc=mu, scale_tril=chol_sigma, validate_args=False)

    def get_mean_and_chol(self, state):
        assert torch.all(torch.exp(self._log_sigma) > 0)
        return self._mu(state, **self._predict_params, output_tensor=True), torch.diag(torch.exp(self._log_sigma))

    def set_weights(self, weights):
        self._mu.set_weights(weights)

    def get_weights(self):
        mu_weights = self._mu.get_weights()

        return np.concatenate([mu_weights])

    def parameters(self):
        return chain(self._mu.model.network.parameters())