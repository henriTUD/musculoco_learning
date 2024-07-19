import torch
import torch.nn as nn

from imitation_lib.utils import FullyConnectedNetwork


class AutoEncoder(nn.Module):

    def __init__(self, input_shape, latent_shape, params_encoder, params_decoder):
        super(AutoEncoder, self).__init__()

        print(f"Latent Dim: {latent_shape}")
        self.encoder = FullyConnectedNetwork(input_shape, latent_shape, **params_encoder)
        self.decoder = FullyConnectedNetwork(latent_shape, input_shape, **params_decoder)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class VariationalAutoEncoder(nn.Module):

    def __init__(self, input_shape, latent_shape, params_encoder, params_decoder):
        super(VariationalAutoEncoder, self).__init__()

        print(f"Latent Dim: {latent_shape}")
        self.latent_shape = latent_shape
        self.encoder = FullyConnectedNetwork(input_shape, (2*latent_shape[0],), **params_encoder)
        self.decoder = FullyConnectedNetwork(latent_shape, input_shape, **params_decoder)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        z = self.encoder(x)
        mu = z[:, :self.latent_shape[0]]
        sig = z[:, self.latent_shape[0]:]
        sig = torch.exp(sig)
        z_sample = mu + sig * self.N.sample(mu.shape)
        self.kl = (sig ** 2 + mu ** 2 - torch.log(sig) - 1 / 2).sum()  # TODO: Validate this
        return self.decoder(z_sample)


class LinearLayerWrapper(nn.Module):

    def __init__(self, input_shape, output_shape,
                 initializer=None, squeeze_out=False, standardizer=None, **kwargs):
        # call base constructor
        super().__init__()

        assert len(input_shape) == len(output_shape) == 1

        self.input_shape = input_shape[0]
        self.output_shape = output_shape[0]

        # construct the linear layers
        self._linear_layer = nn.Linear(input_shape[0], output_shape[0])

        self._stand = standardizer
        self._squeeze_out = squeeze_out

        # make initialization
        if initializer is None:
            nn.init.xavier_uniform_(self._linear_layer.weight)
        else:
            initializer(self._linear_layer.weight)

    def forward(self, *inputs, dim=1):
        inputs = torch.squeeze(torch.cat(inputs, dim=dim), 1)
        if self._stand is not None:
            inputs = self._stand(inputs)
        # define forward pass
        z = inputs.float()
        z = self._linear_layer(z)

        if self._squeeze_out:
            out = torch.squeeze(z)
        else:
            out = z

        return out
