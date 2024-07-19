from imitation_lib.imitation import GAIL_TRPO

import torch
import numpy as np


class ActionDivergenceGAIL(GAIL_TRPO):

    def __init__(self, action_divergence_coef, **kwargs):
        self.action_divergence_coef = action_divergence_coef

        super().__init__(**kwargs)
        print(f'Action Divergence Coef: {self.action_divergence_coef}')
        print(f'Entropy Coef: {self._ent_coeff()}')

    def _compute_loss(self, obs, act, adv, old_log_prob):
        ratio = torch.exp(self.policy.log_prob_t(obs, act) - old_log_prob)
        J = torch.mean(ratio * adv)

        return J + self._ent_coeff() * self.policy.entropy_t(obs) + \
               -1 * self.action_divergence_coef * self._action_divergence(obs)

    def _action_divergence(self, obs):
        raise NotImplementedError


class Uniform2NormalKLDGAIL(ActionDivergenceGAIL):
    def __init__(self, a_dim, lower_action_bound=-1.0, upper_action_bound=1.0, **kwargs):
        self.a_dim = a_dim
        self.lower_action_bound = lower_action_bound
        self.upper_action_bound = upper_action_bound
        print('Action Bounds:')
        print(lower_action_bound)
        print(upper_action_bound)
        super().__init__(**kwargs)

    def _action_divergence(self, obs):
        lower_bound = torch.ones(self.a_dim) * self.lower_action_bound
        upper_bound = torch.ones(self.a_dim) * self.upper_action_bound

        dist_uniform = torch.distributions.uniform.Uniform(lower_bound, upper_bound)
        kl_div = torch.distributions.kl.kl_divergence(dist_uniform, self.policy.get_as_normal_dist(obs))
        kl_div = kl_div.sum(dim=1)
        return kl_div.mean()


class TargetEntropyUniform2NormalKLDGAIL(Uniform2NormalKLDGAIL):
    def __init__(self, target_entropy, **kwargs):
        self.target_entropy = target_entropy
        super().__init__(**kwargs)

    def _compute_loss(self, obs, act, adv, old_log_prob):
        ratio = torch.exp(self.policy.log_prob_t(obs, act) - old_log_prob)
        J = torch.mean(ratio * adv)

        return J + -1 * self._ent_coeff() * (self.policy.entropy_t(obs) - self.target_entropy) ** 2 + \
               -1 * self.action_divergence_coef * self._action_divergence(obs)


class Uniform2MultivariateGaussianKLDGAIL(ActionDivergenceGAIL):

    def __init__(self, a_dim, lower_action_bound=-1.0, upper_action_bound=1.0, **kwargs):
        self.a_dim = a_dim
        self.lower_action_bound = lower_action_bound
        self.upper_action_bound = upper_action_bound

        self.log_uniform_density = self._calc_uniform_density()
        print(f'Uniform Log Density: {self.log_uniform_density}')
        super().__init__(**kwargs)

    def _calc_uniform_density(self):
        return -1 * np.log((self.upper_action_bound-self.lower_action_bound)**self.a_dim)

    def _calc_batched_kl(self, obs):
        pol_dist = self.policy.distribution_t(obs)

        scalar_part = ((self.a_dim/2) * np.log(2*np.pi) + self.log_uniform_density) * torch.ones(obs.shape[0])
        cov_det = 0.5 * torch.log(torch.linalg.det(pol_dist.covariance_matrix))

        shifted_upper_bound = torch.ones_like(pol_dist.loc) * self.upper_action_bound - pol_dist.loc
        shifted_lower_bound = torch.ones_like(pol_dist.loc) * self.lower_action_bound - pol_dist.loc
        inv_cov = torch.linalg.inv(pol_dist.covariance_matrix)

        batched_inv_diag = inv_cov[:, torch.arange(self.a_dim), torch.arange(self.a_dim)]
        diag_integral = (self.upper_action_bound-self.lower_action_bound)**(self.a_dim-1)*batched_inv_diag
        diag_integral = (diag_integral*(shifted_upper_bound**3-shifted_lower_bound**3))/3.0
        diag_integral = diag_integral.sum(dim=1)

        bound_mat = torch.einsum('bi,bj->bij', ((shifted_upper_bound**2-shifted_lower_bound**2),
                                                (shifted_upper_bound**2-shifted_lower_bound**2)))
        off_diag_integral = (self.upper_action_bound-self.lower_action_bound)**(self.a_dim-2) * inv_cov
        off_diag_integral = (off_diag_integral * bound_mat)/4.0
        off_diag_integral[:, torch.arange(self.a_dim), torch.arange(self.a_dim)] = 0.0
        off_diag_integral = off_diag_integral.sum(dim=1)
        off_diag_integral = off_diag_integral.sum(dim=1)

        return scalar_part + cov_det + 0.5 * np.exp(self.log_uniform_density) * (diag_integral + off_diag_integral)

    def _action_divergence(self, obs):
        kl_div = self._calc_batched_kl(obs)
        return kl_div.mean()


class TargetEntropyUniform2MultivariateGaussianKLDGAIL(Uniform2MultivariateGaussianKLDGAIL):

    def __init__(self, target_entropy, **kwargs):
        self.target_entropy = target_entropy
        super().__init__(**kwargs)

    def _compute_loss(self, obs, act, adv, old_log_prob):
        ratio = torch.exp(self.policy.log_prob_t(obs, act) - old_log_prob)
        J = torch.mean(ratio * adv)

        return J + -1 * self._ent_coeff() * (self.policy.entropy_t(obs) - self.target_entropy) ** 2 + \
                    -1 * self.action_divergence_coef * self._action_divergence(obs)