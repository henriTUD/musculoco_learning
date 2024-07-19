from mushroom_rl.approximators import Regressor
from mushroom_rl.policy import TorchPolicy
from mushroom_rl.approximators.parametric import TorchApproximator

import torch
import torch.nn as nn

import numpy as np

from itertools import chain
from musculoco_il.util.torch_models import LinearLayerWrapper

from imitation_lib.utils import NormcInitializer


class LatentExplorationPolicy(TorchPolicy):

    def __init__(self, network, input_shape, output_shape, latent_shape,
                 std_a_0=1., std_x_0=1., learn_latent_layer=False,
                 use_cuda=False, **params):
        super().__init__(use_cuda)

        self._state_dim = input_shape[0]
        self._action_dim = output_shape[0]
        self._latent_dim = latent_shape[0]

        self._learn_latent_layer = learn_latent_layer

        self._mu = Regressor(TorchApproximator, input_shape, latent_shape,
                             network=network, use_cuda=use_cuda, **params)

        self._latent_layer = Regressor(TorchApproximator, latent_shape, output_shape,
                                       network=LinearLayerWrapper, initializer=NormcInitializer(0.001))

        self._predict_params = dict()

        log_sigma_a_init = (torch.ones(self._action_dim) * np.log(std_a_0)).float()
        log_sigma_x_init = (torch.ones(self._latent_dim) * np.log(std_x_0)).float()

        if self._use_cuda:
            log_sigma_a_init = log_sigma_a_init.cuda()
            log_sigma_x_init = log_sigma_x_init.cuda()

        self._log_sigma_a = nn.Parameter(log_sigma_a_init)
        print(torch.exp(2 * self._log_sigma_a))
        self._log_sigma_x = nn.Parameter(log_sigma_x_init)
        print(torch.exp(2 * self._log_sigma_x))

        self.deterministic = False

        # Weight Sizes
        self._mu_w_size = self._mu.weights_size
        self._latent_layer_w_size = self._latent_layer.weights_size

        self._add_save_attr(
            _action_dim='primitive',
            _state_dim='primitive',
            _latent_dim='primitive',
            _learn_latent_layer='primitive',
            _mu='mushroom',
            _latent_layer='mushroom',
            _predict_params='pickle',
            _log_sigma_a='torch',
            _log_sigma_x='torch',
            deterministic='primitive',
            mu_w_size='primitive',
            latent_layer_w_size='primitive',
        )

    def get_covariance_matrix(self):
        a_sigma_mat = torch.diag(torch.exp(2 * self._log_sigma_a))
        x_sigma_mat = torch.diag(torch.exp(2 * self._log_sigma_x))

        w_mat = next(self._latent_layer.model.network.parameters())
        if not self._learn_latent_layer:
            w_mat = w_mat.clone().detach()

        res = torch.matmul(torch.matmul(w_mat, x_sigma_mat), w_mat.T) + a_sigma_mat

        return res

    def draw_action_t(self, state):
        if not self.deterministic:
            return self.distribution_t(state).sample().detach()
        else:
            return self.get_mean(state)

    def log_prob_t(self, state, action):
        return self.distribution_t(state).log_prob(action)[:, None]

    def action_entropy(self):
        return self._action_dim / 2 * np.log(2 * np.pi * np.e) + torch.sum(self._log_sigma_a)

    def latent_entropy(self):
        return self._latent_dim / 2 * np.log(2 * np.pi * np.e) + torch.sum(self._log_sigma_x)

    def entropy_t(self, state):
        cov_mat = self.get_covariance_matrix()
        return torch.distributions.MultivariateNormal(loc=torch.zeros(self._action_dim),
                                                      covariance_matrix=cov_mat, validate_args=False).entropy()

    def distribution_t(self, state):
        mu = self.get_mean(state)
        sigma = self.get_covariance_matrix()
        return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma, validate_args=False)

    def get_mean(self, state):
        latent_state = self._mu(state, **self._predict_params, output_tensor=True)
        mu = self._latent_layer(latent_state, output_tensor=True)
        return mu

    def set_weights(self, weights):
        self._mu.set_weights(weights[:self._mu_w_size])
        self._latent_layer.set_weights(weights[self._mu_w_size:self._mu_w_size+self._latent_layer_w_size])

        log_sigma_a_data = torch.from_numpy(weights[self._mu_w_size+self._latent_layer_w_size:
                                                  self._mu_w_size+self._latent_layer_w_size+self._action_dim])
        if self.use_cuda:
            log_sigma_a_data = log_sigma_a_data.cuda()
        self._log_sigma_a.data = log_sigma_a_data

        log_sigma_x_data = torch.from_numpy(weights[-self._latent_dim:])
        if self.use_cuda:
            log_sigma_x_data = log_sigma_x_data.cuda()
        self._log_sigma_x.data = log_sigma_x_data

    def get_weights(self):
        mu_weights = self._mu.get_weights()
        latent_weights = self._latent_layer.get_weights()
        sigma_a_weights = self._log_sigma_a.data.detach().cpu().numpy()
        sigma_x_weights = self._log_sigma_x.data.detach().cpu().numpy()

        return np.concatenate([mu_weights, latent_weights, sigma_a_weights, sigma_x_weights])

    def parameters(self):
        return chain(self._mu.model.network.parameters(), self._latent_layer.model.network.parameters(),
                     [self._log_sigma_a], [self._log_sigma_x])


class LATTICEPolicy(TorchPolicy):

    def __init__(self, network, input_shape, output_shape, latent_shape,
                 std_a_0=1., std_x_0=1., learn_latent_layer=False, detach_latent_state=True,
                 use_expln=True,
                 alpha=1.0, resampling_n=2, min_std=0.05, max_std=1.5, epsilon=1e-6,
                 use_cuda=False, **params):
        super().__init__(use_cuda)

        self._state_dim = input_shape[0]
        self._action_dim = output_shape[0]
        self._latent_dim = latent_shape[0]

        self._resampling_n = resampling_n
        self._step_counter = 0

        self._use_expln = use_expln
        self._min_std = min_std
        self._max_std = max_std
        self._epsilon = epsilon

        self._learn_latent_layer = learn_latent_layer
        self._detach_latent_state = detach_latent_state

        self._alpha = alpha

        self._mu = Regressor(TorchApproximator, input_shape, latent_shape,
                             network=network, use_cuda=use_cuda, **params)

        self._latent_layer = Regressor(TorchApproximator, latent_shape, output_shape,
                                       network=LinearLayerWrapper, initializer=NormcInitializer(0.001))

        self._predict_params = dict()

        log_sigma_a_init = (torch.ones(self._action_dim * self._latent_dim) * np.log(std_a_0)).float()
        log_sigma_x_init = (torch.ones(self._latent_dim * self._latent_dim) * np.log(std_x_0)).float()

        self._log_sigma_a_flat = nn.Parameter(log_sigma_a_init)
        self._log_sigma_x_flat = nn.Parameter(log_sigma_x_init)

        self._P_action_flat = torch.zeros(self._action_dim * self._latent_dim)
        self._P_latent_flat = torch.zeros(self._latent_dim * self._latent_dim)

        self.deterministic = False

        # Weight Sizes
        self._mu_w_size = self._mu.weights_size
        self._latent_layer_w_size = self._latent_layer.weights_size
        self._sigma_a_size = self._action_dim * self._latent_dim
        self._sigma_x_size = self._latent_dim * self._latent_dim

        self._add_save_attr(
            _action_dim='primitive',
            _state_dim='primitive',
            _latent_dim='primitive',
            _resampling_n='primitive',
            _learn_latent_layer='primitive',
            _detach_latent_state='primitive',
            _alpha='primitive',
            _use_expln='primitive',
            _mu='mushroom',
            _latent_layer='mushroom',
            _predict_params='pickle',
            _log_sigma_a_flat='torch',
            _log_sigma_x_flat='torch',
            deterministic='primitive',
            mu_w_size='primitive',
            latent_layer_w_size='primitive',
            _sigma_a_size='primitive',
            _sigma_x_size='primitive',
            _min_std='primitive',
            _max_std='primitive',
            _epsilon='primitive',
        )

    def reset(self):
        self._step_counter = 0

    def get_covariance_matrix(self, latent_state):
        if self._detach_latent_state:
            latent_state = latent_state.clone().detach()

        w_mat = next(self._latent_layer.model.network.parameters())
        if not self._learn_latent_layer:
            w_mat = w_mat.clone().detach()

        S_a = self.action_flat_to_mat(self.activate_log_std(self._log_sigma_a_flat)) #.repeat(state.shape[0], 1, 1)
        S_x = self.latent_flat_to_mat(self.activate_log_std(self._log_sigma_x_flat)) #.repeat(state.shape[0], 1, 1)

        #latent_state = latent_state[:, :, None]
        #a_diag = torch.matmul(S_a**2, latent_state**2).squeeze()
        a_diag = torch.matmul(latent_state[:, None, :]**2, S_a.T**2).squeeze()
        a_component = torch.eye(self._action_dim) * a_diag[:, None, :]

        #latent_diag = torch.matmul(S_x**2, latent_state**2).squeeze()
        latent_diag = torch.matmul(latent_state[:, None, :]**2, S_x.T**2).squeeze()
        latent_component = torch.eye(self._latent_dim) * latent_diag[:, None, :]
        latent_component = self._alpha**2 * torch.matmul(torch.matmul(w_mat, latent_component), w_mat.T)
        return a_component + latent_component

    def activate_log_std(self, log_value):
        log_std = log_value.clip(min=np.log(self._min_std), max=np.log(self._max_std))
        log_std = log_std - 0.5 * np.log(self._latent_dim)

        if self._use_expln:
            # From gSDE paper, it allows to keep variance
            # above zero and prevent it from growing too fast
            below_threshold = torch.exp(log_std) * (log_std <= 0)
            # Avoid NaN: zeros values that are below zero
            safe_log_std = log_std * (log_std > 0) + self._epsilon
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            # Use normal exponential
            std = torch.exp(log_std)

        return std

    def latent_flat_to_mat(self, flat_rep):
        return torch.reshape(flat_rep, (self._latent_dim, self._latent_dim))

    def action_flat_to_mat(self, flat_rep):
        return torch.reshape(flat_rep, (self._action_dim, self._latent_dim))

    def handle_time_correlation(self):
        if self._step_counter % self._resampling_n == 0:
            self._P_action_flat = torch.normal(torch.zeros(self._action_dim * self._latent_dim),
                                               self.activate_log_std(self._log_sigma_a_flat))
            self._P_latent_flat = torch.normal(torch.zeros(self._latent_dim * self._latent_dim),
                                               self.activate_log_std(self._log_sigma_x_flat))
        self._step_counter += 1

    def state_dependent_exploration(self, state):
        latent_state = self._mu(state, **self._predict_params, output_tensor=True)

        action_eps = torch.matmul(
            self.action_flat_to_mat(self._P_action_flat), latent_state.squeeze())

        sampled_latent = latent_state.squeeze() + self._alpha * torch.matmul(
            self.latent_flat_to_mat(self._P_latent_flat), latent_state.squeeze())
        return self._latent_layer(sampled_latent, output_tensor=True) + action_eps

    def draw_action_t(self, state):
        if not self.deterministic:
            self.handle_time_correlation()
            return self.state_dependent_exploration(state)
        else:
            latent_state = self._mu(state, **self._predict_params, output_tensor=True)
            return self.get_mean(latent_state)

    def log_prob_t(self, state, action):
        return self.distribution_t(state).log_prob(action)[:, None]

    def distribution_t(self, state):
        latent_state = self._mu(state, **self._predict_params, output_tensor=True)
        mu = self.get_mean(latent_state)
        sigma = self.get_covariance_matrix(latent_state)
        return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma, validate_args=False)

    def get_mean(self, latent_state):
        mu = self._latent_layer(latent_state, output_tensor=True)
        return mu

    def entropy_t(self, state):
        analytic_entropy = self.distribution_t(state).entropy().mean()
        return analytic_entropy

    def set_weights(self, weights):
        self._mu.set_weights(weights[:self._mu_w_size])
        self._latent_layer.set_weights(weights[self._mu_w_size:self._mu_w_size + self._latent_layer_w_size])

        log_sigma_a_data = torch.from_numpy(weights[self._mu_w_size + self._latent_layer_w_size:
                                                    self._mu_w_size + self._latent_layer_w_size + self._sigma_a_size])
        if self.use_cuda:
            log_sigma_a_data = log_sigma_a_data.cuda()
        self._log_sigma_a_flat.data = log_sigma_a_data

        log_sigma_x_data = torch.from_numpy(weights[-self._sigma_x_size:])
        if self.use_cuda:
            log_sigma_x_data = log_sigma_x_data.cuda()
        self._log_sigma_x_flat.data = log_sigma_x_data

    def get_weights(self):
        mu_weights = self._mu.get_weights()
        latent_weights = self._latent_layer.get_weights()
        sigma_a_weights = self._log_sigma_a_flat.data.detach().cpu().numpy()
        sigma_x_weights = self._log_sigma_x_flat.data.detach().cpu().numpy()

        return np.concatenate([mu_weights, latent_weights, sigma_a_weights, sigma_x_weights])

    def parameters(self):
        return chain(self._mu.model.network.parameters(), self._latent_layer.model.network.parameters(),
                     [self._log_sigma_a_flat], [self._log_sigma_x_flat])


class HybridLATTICEPolicy(TorchPolicy):

    def __init__(self, network, input_shape, output_shape, latent_shape,
                 std_a_0=1., std_x_0=1., learn_latent_layer=False, detach_latent_state=True,
                 use_expln=True, epsilon=1e-6,
                 alpha=1.0, resampling_n=2, min_std=0.05, max_std=1.5,
                 use_cuda=False, **params):
        super().__init__(use_cuda)

        self._state_dim = input_shape[0]
        self._action_dim = output_shape[0]
        self._latent_dim = latent_shape[0]

        self._resampling_n = resampling_n
        self._step_counter = 0

        self._use_expln = use_expln
        self._min_std = min_std
        self._max_std = max_std

        self._learn_latent_layer = learn_latent_layer
        self._detach_latent_state = detach_latent_state

        self._alpha = alpha
        self._epsilon = epsilon

        self._mu = Regressor(TorchApproximator, input_shape, latent_shape,
                             network=network, use_cuda=use_cuda, **params)

        self._latent_layer = Regressor(TorchApproximator, latent_shape, output_shape,
                                       network=LinearLayerWrapper, initializer=NormcInitializer(0.001))

        self._predict_params = dict()

        log_sigma_a_init = (torch.ones(self._action_dim) * np.log(std_a_0)).float()
        log_sigma_x_init = (torch.ones(self._latent_dim * self._latent_dim) * np.log(std_x_0)).float()

        self._log_sigma_a = nn.Parameter(log_sigma_a_init)
        self._log_sigma_x_flat = nn.Parameter(log_sigma_x_init)

        self._P_latent_flat = torch.zeros(self._latent_dim * self._latent_dim)

        self.deterministic = False

        # Weight Sizes
        self._mu_w_size = self._mu.weights_size
        self._latent_layer_w_size = self._latent_layer.weights_size
        self._sigma_a_size = self._action_dim
        self._sigma_x_size = self._latent_dim * self._latent_dim

        self._add_save_attr(
            _action_dim='primitive',
            _state_dim='primitive',
            _latent_dim='primitive',
            _learn_latent_layer='primitive',
            _mu='mushroom',
            _latent_layer='mushroom',
            _predict_params='pickle',
            _log_sigma_a='torch',
            _log_sigma_x_flat='torch',
            deterministic='primitive',
            _alpha='primitive',
            mu_w_size='primitive',
            _sigma_a_size='primitive',
            _sigma_x_size='primitive',
            latent_layer_w_size='primitive',
            _resampling_n='primitive',
            _use_expln='primitive',
            _min_std='primitive',
            _max_std='primitive',
            _detach_latent_state='primitive',
            _epsilon='primitive',
        )

    def reset(self):
        self._step_counter = 0

    def get_covariance_matrix(self, latent_state):
        if self._detach_latent_state:
            latent_state = latent_state.clone().detach()

        w_mat = next(self._latent_layer.model.network.parameters())
        if not self._learn_latent_layer:
            w_mat = w_mat.clone().detach()

        a_component = torch.diag(torch.exp(2 * self._log_sigma_a))

        S_x = self.latent_flat_to_mat(self.activate_log_std(self._log_sigma_x_flat))

        latent_diag = torch.matmul(latent_state[:, None, :] ** 2, S_x.T ** 2).squeeze()
        latent_component = torch.eye(self._latent_dim) * latent_diag[:, None, :]
        latent_component = self._alpha ** 2 * torch.matmul(torch.matmul(w_mat, latent_component), w_mat.T)

        return a_component + latent_component

    def activate_log_std(self, log_value):
        log_std = log_value.clip(min=np.log(self._min_std), max=np.log(self._max_std))
        log_std = log_std - 0.5 * np.log(self._latent_dim)

        if self._use_expln:
            # From gSDE paper, it allows to keep variance
            # above zero and prevent it from growing too fast
            below_threshold = torch.exp(log_std) * (log_std <= 0)
            # Avoid NaN: zeros values that are below zero
            safe_log_std = log_std * (log_std > 0) + self._epsilon
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            # Use normal exponential
            std = torch.exp(log_std)

        return std

    def latent_flat_to_mat(self, flat_rep):
        return torch.reshape(flat_rep, (self._latent_dim, self._latent_dim))

    def handle_time_correlation(self):
        if self._step_counter % self._resampling_n == 0:
            self._P_latent_flat = torch.normal(torch.zeros(self._latent_dim * self._latent_dim),
                                               self.activate_log_std(self._log_sigma_x_flat))
        self._step_counter += 1

    def state_dependent_exploration(self, state):
        latent_state = self._mu(state, **self._predict_params, output_tensor=True)

        action_eps = torch.normal(torch.zeros(self._action_dim),
                                  torch.exp(self._log_sigma_a))

        sampled_latent = latent_state.squeeze() + self._alpha*torch.matmul(
                                            self.latent_flat_to_mat(self._P_latent_flat), latent_state.squeeze())
        return self._latent_layer(sampled_latent, output_tensor=True) + action_eps

    def draw_action_t(self, state):
        if not self.deterministic:
            self.handle_time_correlation()
            return self.state_dependent_exploration(state)
        else:
            latent_state = self._mu(state, **self._predict_params, output_tensor=True)
            return self.get_mean(latent_state)

    def log_prob_t(self, state, action):
        return self.distribution_t(state).log_prob(action)[:, None]

    def distribution_t(self, state):
        latent_state = self._mu(state, **self._predict_params, output_tensor=True)
        mu = self.get_mean(latent_state)
        sigma = self.get_covariance_matrix(latent_state)
        return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma, validate_args=False)

    def get_mean(self, latent_state):
        mu = self._latent_layer(latent_state, output_tensor=True)
        return mu

    def entropy_t(self, state):
        analytic_entropy = self.distribution_t(state).entropy().mean()
        return analytic_entropy

    def set_weights(self, weights):
        self._mu.set_weights(weights[:self._mu_w_size])
        self._latent_layer.set_weights(weights[self._mu_w_size:self._mu_w_size + self._latent_layer_w_size])

        log_sigma_a_data = torch.from_numpy(weights[self._mu_w_size + self._latent_layer_w_size:
                                                    self._mu_w_size + self._latent_layer_w_size + self._sigma_a_size])
        if self.use_cuda:
            log_sigma_a_data = log_sigma_a_data.cuda()
        self._log_sigma_a.data = log_sigma_a_data

        log_sigma_x_data = torch.from_numpy(weights[-self._sigma_x_size:])
        if self.use_cuda:
            log_sigma_x_data = log_sigma_x_data.cuda()
        self._log_sigma_x_flat.data = log_sigma_x_data

    def get_weights(self):
        mu_weights = self._mu.get_weights()
        latent_weights = self._latent_layer.get_weights()
        sigma_a_weights = self._log_sigma_a.data.detach().cpu().numpy()
        sigma_x_weights = self._log_sigma_x_flat.data.detach().cpu().numpy()

        return np.concatenate([mu_weights, latent_weights, sigma_a_weights, sigma_x_weights])

    def parameters(self):
        return chain(self._mu.model.network.parameters(), self._latent_layer.model.network.parameters(),
                     [self._log_sigma_a], [self._log_sigma_x_flat])