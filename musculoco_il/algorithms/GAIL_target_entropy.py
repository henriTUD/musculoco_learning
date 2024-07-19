
from imitation_lib.imitation import GAIL_TRPO

import torch


class TargetEntropyGAIL(GAIL_TRPO):

    def __init__(self, target_entropy, **kwargs):
        self.target_entropy = target_entropy
        super().__init__(**kwargs)

    def _compute_loss(self, obs, act, adv, old_log_prob):
        ratio = torch.exp(self.policy.log_prob_t(obs, act) - old_log_prob)
        J = torch.mean(ratio * adv)
        return J + -1 * self._ent_coeff() * (self.policy.entropy_t(obs) - self.target_entropy) ** 2