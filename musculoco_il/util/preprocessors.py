from mushroom_rl.core.serialization import Serializable


class StateSelectionPreprocessor(Serializable):

    def __init__(self, first_n):
        self.first_n = first_n

        self._add_save_attr(
            first_n='primitive',
        )

    def __call__(self, state):
        return state[:self.first_n]