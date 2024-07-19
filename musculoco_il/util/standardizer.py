import torch

import torch.nn as nn
import numpy as np


class Standardizer(nn.Module):

    def __init__(self, use_cuda=False):
        # call base constructor
        super(Standardizer, self).__init__()

        self._sum = 0.0
        self._sumsq = 0.0
        self._count = 0
        self._use_cuda = use_cuda

        self.mean = 0.0
        self.std = 1.0

        self.frozen = False

    def forward(self, inputs):
        if not self.frozen:
            self.update_mean_std(inputs.detach().cpu().numpy())
        mean = torch.tensor(self.mean).cuda() if self._use_cuda else torch.tensor(self.mean)
        std = torch.tensor(self.std).cuda() if self._use_cuda else torch.tensor(self.std)
        return (inputs - mean) / std

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def update_mean_std(self, x):
        self._sum += x.sum(axis=0).ravel()
        self._sumsq += np.square(x).sum(axis=0).ravel()
        self._count += np.array([len(x)])
        self.mean = self._sum / self._count
        self.std = np.sqrt(np.maximum((self._sumsq / self._count) - np.square(self.mean), 1e-3))