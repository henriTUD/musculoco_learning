from mushroom_rl.approximators import Regressor
from mushroom_rl.policy import TorchPolicy
from mushroom_rl.approximators.parametric import TorchApproximator

import torch
import torch.nn as nn
from torch.distributions import TransformedDistribution, AffineTransform

from torch.distributions.beta import Beta

from itertools import chain

import numpy as np


class BetaDistributionTorchPolicy(TorchPolicy):

    def __init__(self, network, input_shape, output_shape, softplus_offset, ab_offset, use_cuda=False, **params):
        super().__init__(use_cuda)

        self._action_dim = output_shape[0]

        self.softplus_offset = softplus_offset
        self.ab_offset = ab_offset

        self._eps_log_prob = 1e-6

        self._network = Regressor(TorchApproximator, input_shape, (2*output_shape[0],),
                                        network=network, use_cuda=use_cuda, **params)

        self._predict_params = dict()

        self._network_size = self._network.weights_size

        self._alpha_selector = torch.arange(start=0, end=self._action_dim, step=1)
        self._beta_selector = torch.arange(start=self._action_dim, end=2*self._action_dim, step=1)

        print(self._alpha_selector)
        print(self._beta_selector)

        self._add_save_attr(
            _action_dim='primitive',
            _network='mushroom',
            _predict_params='pickle',
            _network_size='primitive',
            _alpha_selector='torch',
            _beta_selector='torch',
            softplus_offset='primitive',
            ab_offset='primitive'
        )

    def draw_action_t(self, state):
        return self.distribution_t(state).sample().detach()

    def log_prob_t(self, state, action):
        dist = self.distribution_t(state)
        clamped_action = torch.clamp(action, min=1e-6, max=1.0-1e-6)
        return torch.sum(dist.log_prob(clamped_action)[:, None], dim=2)

    def entropy_t(self, state):
        analytic_entropy = self.distribution_t(state).entropy().sum(dim=1).mean()
        return analytic_entropy

    def distribution_t(self, state):
        alpha, beta = self.parameterize_beta_distribution(state)
        dist = Beta(alpha, beta)
        return dist

    def parameterize_beta_distribution(self, state):
        pred = self._network(state, **self._predict_params, output_tensor=True)

        out = torch.nn.functional.softplus(pred + self.softplus_offset)

        alpha = torch.index_select(out, 1, self._alpha_selector) + self.ab_offset
        beta = torch.index_select(out, 1, self._beta_selector) + self.ab_offset

        return alpha, beta

    def set_weights(self, weights):
        self._network.set_weights(weights)

    def get_weights(self):
        return np.concatenate([self._network.get_weights()])

    def parameters(self):
        return chain(self._network.model.network.parameters())


class MeanParameterizedBetaDistributionTorchPolicy(TorchPolicy):

    def __init__(self, network, input_shape, output_shape, enforce_unimodal=True,
                 raw_std_0=1., detach_mean=True, use_cuda=False, **params):
        super().__init__(use_cuda)

        self._action_dim = output_shape[0]

        self._mu = Regressor(TorchApproximator, input_shape, output_shape,
                             network=network, use_cuda=use_cuda, **params)
        self._predict_params = dict()

        self.enforce_unimodal = enforce_unimodal
        self.detach_mean = detach_mean

        self._raw_sigma = nn.Parameter(torch.ones(self._action_dim) * raw_std_0)
        #TODO: Cuda handling

        self.deterministic = False

        self._add_save_attr(
            _action_dim='primitive',
            _mu='mushroom',
            _predict_params='pickle',
            _raw_sigma='torch',
            deterministic='primitive',
            detach_mean='primitive',
        )

    def draw_action_t(self, state):
        if not self.deterministic:
            return self.distribution_t(state).sample().detach()
        else:
            return self.get_mean(state)

    def log_prob_t(self, state, action):
        return torch.sum(self.distribution_t(state).log_prob(action)[:, None], dim=2)

    def entropy_t(self, state):
        analytic_entropy = self.distribution_t(state).entropy().sum(dim=1).mean()
        return analytic_entropy

    def distribution_t(self, state):
        mean = self.get_mean(state)
        alpha, beta = self.parameterize_beta_distribution(mean)
        return Beta(alpha, beta)

    def get_mean(self, state):
        return self._mu(state, **self._predict_params, output_tensor=True)

    def activate_sigma(self, mean):
        if self.enforce_unimodal:
            max_var = mean * torch.min(((mean*(1-mean))/(1+mean)), (torch.square(1-mean)/(2-mean)))
        else:
            max_var = mean * (1 - mean)
        max_sigma = torch.sqrt(max_var)
        return torch.sigmoid(self._raw_sigma) * max_sigma

    def parameterize_beta_distribution(self, mean):
        clamped_mean = torch.clamp(mean, min=1e-5,
                                   max=1. - 1e-5)
        if self.detach_mean:
            detached_clamped_mean = clamped_mean.clone().detach()
            sigma = self.activate_sigma(detached_clamped_mean)
        else:
            sigma = self.activate_sigma(clamped_mean)
        alpha = (((1 - clamped_mean)/(torch.square(sigma))) - (1/clamped_mean)) * torch.square(clamped_mean)
        beta = alpha * ((1/clamped_mean) - 1)

        return alpha, beta

    def set_weights(self, weights):
        raw_sigma_data = torch.from_numpy(weights[-self._action_dim:])
        if self.use_cuda:
            raw_sigma_data = raw_sigma_data.cuda()
        self._raw_sigma.data = raw_sigma_data
        self._mu.set_weights(weights[:-self._action_dim])

    def get_weights(self):
        mu_weights = self._mu.get_weights()
        raw_sigma_weights = self._raw_sigma.data.detach().cpu().numpy()

        return np.concatenate([mu_weights, raw_sigma_weights])

    def parameters(self):
        return chain(self._mu.model.network.parameters(), [self._raw_sigma])


class SeparateNetworkBetaDistributionTorchPolicy(TorchPolicy):

    def __init__(self, network, input_shape, output_shape,
                 n_entropy_samples=1, use_cuda=False, **params):
        super().__init__(use_cuda)

        self._action_dim = output_shape[0]

        self._alpha_network = Regressor(TorchApproximator, input_shape, output_shape,
                             network=network, use_cuda=use_cuda, **params)

        self._beta_network = Regressor(TorchApproximator, input_shape, output_shape,
                             network=network, use_cuda=use_cuda, **params)

        self._predict_params = dict()

        self.n_entropy_samples = 1

        self._network_size = self._alpha_network.weights_size

        self._add_save_attr(
            _action_dim='primitive',
            _alpha_network='mushroom',
            _beta_network='mushroom',
            _predict_params='pickle',
            n_entropy_samples='primitive',
            _network_size='primitive'
        )

    def draw_action_t(self, state):
        return self.distribution_t(state).sample().detach()

    def log_prob_t(self, state, action):
        return torch.sum(self.distribution_t(state).log_prob(action)[:, None], dim=2)

    def entropy_t(self, state):
        sampled_action = self.distribution_t(state).rsample((self.n_entropy_samples,))
        state_repeated = state.repeat((self.n_entropy_samples, 1))
        entropy = -1 * torch.mean(self.log_prob_t(state_repeated, sampled_action.view(-1, self._action_dim)))
        return entropy

    def distribution_t(self, state):
        alpha, beta = self.parameterize_beta_distribution(state)
        return Beta(alpha, beta)

    def parameterize_beta_distribution(self, state):
        alpha = self._alpha_network(state, **self._predict_params, output_tensor=True)
        beta = self._beta_network(state, **self._predict_params, output_tensor=True)

        alpha = torch.nn.functional.softplus(alpha) + 1.
        beta = torch.nn.functional.softplus(beta) + 1.

        return alpha, beta

    def set_weights(self, weights):
        self._alpha_network.set_weights(weights[:self._network_size])
        self._beta_network.set_weights(weights[self._network_size:])

    def get_weights(self):
        alpha_weights = self._alpha_network.get_weights()
        beta_weights = self._beta_network.get_weights()

        return np.concatenate([alpha_weights, beta_weights])

    def parameters(self):
        return chain(self._alpha_network.model.network.parameters(), self._beta_network.model.network.parameters())