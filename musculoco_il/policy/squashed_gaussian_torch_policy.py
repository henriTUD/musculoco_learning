from mushroom_rl.policy import GaussianTorchPolicy
import torch
from mushroom_rl.utils.torch import to_float_tensor


class SquashedGaussianTorchPolicy(GaussianTorchPolicy):

    def __init__(self, a_dim, min_a, max_a, n_entropy_samples=1, **kwargs):
        super().__init__(**kwargs)

        self.a_dim = a_dim
        self._delta_a = to_float_tensor(.5 * (max_a - min_a), self.use_cuda)
        self._central_a = to_float_tensor(.5 * (max_a + min_a), self.use_cuda)

        self.n_entropy_samples = n_entropy_samples

    def draw_action_t(self, state):
        a_raw = self.distribution_t(state).sample().detach()
        a = torch.tanh(a_raw)
        a_true = a * self._delta_a + self._central_a

        return a_true

    def get_mean_and_chol(self, state):
        return self._mu(state, **self._predict_params, output_tensor=True), \
               torch.diag(torch.exp(torch.clamp(self._log_sigma, min=-8., max=8.)))

    def log_prob_t(self, state, action, a_raw=None):
        a_squashed = torch.clamp((action - self._central_a) / self._delta_a, -1+5e-7, 1-5e-7)

        if a_raw is None:
            a_raw = torch.atanh(a_squashed)

        log_prob = self.distribution_t(state).log_prob(a_raw.squeeze())
        log_prob -= torch.log(1. - a_squashed.pow(2) + 1e-6).sum(dim=1)

        return log_prob.unsqueeze(dim=1)

    def entropy_t(self, state=None):
        a_raw = self.distribution_t(state).rsample((self.n_entropy_samples,))  # RSAMPLE !!
        a = torch.tanh(a_raw)
        sampled_action = a * self._delta_a + self._central_a
        state_repeated = state.repeat((self.n_entropy_samples, 1))
        entropy = -1 * torch.mean(self.log_prob_t(state_repeated,
                                                  sampled_action.view(-1, self.a_dim),
                                                  a_raw=a_raw.view(-1, self.a_dim)))

        return entropy