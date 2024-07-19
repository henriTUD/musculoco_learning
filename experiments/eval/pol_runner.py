import matplotlib.pyplot as plt
from mushroom_rl.core.serialization import *

from argparse import ArgumentParser

from mushroom_rl.policy import GaussianTorchPolicy

from musculoco_il.util.action_specs import HAMNER_HUMANOID_FIXED_ARMS_ACTION_SPEC
from musculoco_il.util.rewards import OutOfBoundsActionCost
from musculoco_il.policy.latent_exploration_torch_policy import *
from musculoco_il.policy.random_action_policy import RandomGaussianPolicy

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J, parse_dataset, compute_episodes_length

from loco_mujoco import LocoEnv

import time

np.random.seed(5011)
torch.manual_seed(5011)

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename",
                    help="Agent pickle", metavar="FILE")
parser.add_argument('--deterministic', action='store_true')
parser.set_defaults(deterministic=False)

args = parser.parse_args()

print(args)

env_freq = 1000  # hz, added here as a reminder
traj_data_freq = 500  # hz, added here as a reminder
desired_contr_freq = 50

n_substeps = env_freq // desired_contr_freq

mdp = LocoEnv.make('HumanoidMuscle.walk',
                    gamma=0.99,
                    horizon=1000,
                    n_substeps=n_substeps,
                    timestep=1 / env_freq,
                    use_box_feet=True,
                    use_foot_forces=False,
                    obs_mujoco_act=False,
                    muscle_force_scaling=1.25,
                    alpha_box_feet=0.5,
                    )

#mdp.play_trajectory(n_steps_per_episode=700, n_episodes=1, render=True, record=True)

action_dim = mdp.info.action_space.shape[0]

print('Action Dim: ' + str(action_dim))

print('State Dim: ' + str(mdp.info.observation_space._shape))

print(mdp.get_all_observation_keys())
print(len(mdp.get_all_observation_keys()))

AGENT_PATH = args.filename
DETERMINISTIC_POLICY = args.deterministic

agent = Serializable.load(AGENT_PATH)

#agent.policy = RandomGaussianPolicy(a_shape=(92,))

agent.policy.deterministic = False
if DETERMINISTIC_POLICY:
    agent.policy.deterministic = True

core = Core(agent, mdp)

start = time.time()

dataset = core.evaluate(render=True, n_episodes=2, record=True)

done = time.time()
elapsed = done - start
print("ELAPSED: " + str(elapsed))

s, a, r, ns, *_ = parse_dataset(dataset)

#np.save('latent_oob_data_run.npy', s)

J = np.mean(compute_J(dataset, mdp.info.gamma))
R = np.mean(compute_J(dataset))
L = np.mean(compute_episodes_length(dataset))
