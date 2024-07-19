
import numpy as np


class OutOfBoundsActionCost(object):

    def __init__(self, lower_bound, upper_bound, reward_scale=1.0, const_cost=0.0, func_type='abs'):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.reward_scale = reward_scale
        self.const_cost = const_cost

        if func_type == 'abs':
            self.func = np.abs
        elif func_type == 'squared':
            self.func = np.square
        else:
            raise Exception(f'{func_type} is not a valid function type!')

    def __call__(self, state, action, next_state):
        lower_cost = (self.lower_bound - action + self.const_cost) * (action < self.lower_bound)
        upper_cost = (action - self.upper_bound + self.const_cost) * (action > self.upper_bound)
        return -1 * self.reward_scale * np.sum(self.func(lower_cost + upper_cost))


if __name__ == "__main__":
    rew = OutOfBoundsActionCost(0.0, 1.0, reward_scale=1.0, const_cost=0.0, func_type='squared')
    u_test = np.array([1, 0.0, 1.5, 0.1, 0.999, -2, -0.5, 2.0, -1.1, 0.5])
    print(rew(np.zeros(1), u_test, np.zeros(1)))