
from mushroom_rl.core.serialization import *

from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from musculoco_il.util.torch_models import AutoEncoder, VariationalAutoEncoder
from musculoco_il.util.util_functions import batch_iterable


class SynergisticActionRepresentation(Serializable):

    def __init__(self, dim_action_space, dim_synergistic_space):
        self.dim_action_space = dim_action_space
        self.dim_synergistic_space = dim_synergistic_space

        self._add_save_attr(
            dim_action_space='primitive',
            dim_synergistic_space='primitive',
        )

    def fit(self, act):
        raise NotImplementedError

    def action_to_synergistic(self, actions):
        raise NotImplementedError

    def synergistic_to_action(self, synergistic_action):
        raise NotImplementedError


class SAR_PCAICA(SynergisticActionRepresentation):

    def __init__(self, dim_action_space, dim_synergistic_space, ica_max_iter=1500):
        super().__init__(dim_action_space, dim_synergistic_space)
        self._ica_max_iter = ica_max_iter

        self.pca = None
        self.ica = None
        self.action_normalizer = None

        self._add_save_attr(
            _ica_max_iter='primitive',
            pca='pickle',
            ica='pickle',
            action_normalizer='pickle'
        )

    @staticmethod
    def find_synergies(acts, plot=True):
        syn_dict = {}
        for i in range(acts.shape[1]):
            pca = PCA(n_components=i + 1)
            _ = pca.fit_transform(acts)
            syn_dict[i + 1] = round(sum(pca.explained_variance_ratio_), 4)

        if plot:
            plt.plot(list(syn_dict.keys()), list(syn_dict.values()))
            plt.title('VAF by N synergies')
            plt.xlabel('# synergies')
            plt.ylabel('VAF')
            plt.grid()
            #plt.show()
        return syn_dict

    def fit(self, acts):
        self.pca = PCA(n_components=self.dim_synergistic_space)
        pca_act = self.pca.fit_transform(acts)

        self.ica = FastICA(max_iter=self._ica_max_iter)
        pcaica_act = self.ica.fit_transform(pca_act)

        self.action_normalizer = MinMaxScaler((-1, 1))
        self.action_normalizer.fit(pcaica_act)

    def action_to_synergistic(self, actions):
        return self.action_normalizer.transform(
                    self.ica.transform(
                        self.pca.transform(actions)))

    def synergistic_to_action(self, synergistic_action):
        action = self.pca.inverse_transform(
                        self.ica.inverse_transform(
                            self.action_normalizer.inverse_transform(synergistic_action)))
        return action


class SAR_AutoEncoder(SynergisticActionRepresentation):

    def __init__(self, dim_action_space, dim_synergistic_space,
                 encoder_params, decoder_params, epochs_fit=500, variational_mode=True):
        super().__init__(dim_action_space, dim_synergistic_space)

        self.variational_mode = variational_mode

        if self.variational_mode:
            self.auto_encoder = VariationalAutoEncoder((dim_action_space,), (dim_synergistic_space,), encoder_params, decoder_params)
        else:
            self.auto_encoder = AutoEncoder((dim_action_space,), (dim_synergistic_space,), encoder_params, decoder_params)
        self.epochs_fit = epochs_fit

        self._add_save_attr(
            variational_mode='primitive',
            auto_encoder='torch',
        )

    def fit(self, act):
        print("--- fitting SAR Auto Encoder...")
        opt = torch.optim.Adam(self.auto_encoder.parameters(), lr=6e-4, weight_decay=1e-4)
        batches = [torch.tensor(batch) for batch in batch_iterable(act.copy(), n=500)]
        for epoch in range(self.epochs_fit):
            for batch in batches:
                opt.zero_grad()
                x_hat = self.auto_encoder(batch)
                loss = ((batch - x_hat) ** 2).mean()
                if self.variational_mode:
                    loss += self.auto_encoder.kl
                loss.backward()
                opt.step()
        print(f'Final Loss: {loss}')

    def action_to_synergistic(self, actions):
        if self.variational_mode:
            return self.auto_encoder.encoder(torch.tensor(actions.copy())
                                             .unsqueeze(0)).detach().numpy()[:, :self.dim_synergistic_space]
        else:
            return self.auto_encoder.encoder(torch.tensor(actions.copy()).unsqueeze(0)).detach().numpy()

    def synergistic_to_action(self, synergistic_action):
        return self.auto_encoder.decoder(torch.tensor(synergistic_action.copy())).detach().numpy()
