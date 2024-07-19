from mushroom_rl.core.serialization import *

from argparse import ArgumentParser

from musculoco_il.environment.SAR_env_wrapper import SARWrappedMuscleHumanoid
from musculoco_il.policy.latent_exploration_torch_policy import *

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J, parse_dataset, compute_episodes_length

import time

np.random.seed(2011)
torch.manual_seed(2011)

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

sar_module = Serializable.load("PATH_TO_SAR_MODULE/sar_module.msh")

print('--- Building SAR env wrapper...')
mdp = SARWrappedMuscleHumanoid.generate(
    task='run',
    gamma=0.99,
    horizon=1000,
    n_substeps=n_substeps,
    timestep=1 / 1000,
    muscle_force_scaling=1.25,
    sar_module=sar_module
)

action_dim = mdp.info.action_space.shape[0]

print('Action Dim: ' + str(action_dim))

print('State Dim: ' + str(mdp.info.observation_space._shape))

print(mdp.get_all_observation_keys())
print(len(mdp.get_all_observation_keys()))

AGENT_PATH = args.filename
DETERMINISTIC_POLICY = args.deterministic

agent = Serializable.load(AGENT_PATH)

agent.policy.deterministic = False
if DETERMINISTIC_POLICY:
    agent.policy.deterministic = True

core = Core(agent, mdp)

dataset = core.evaluate(render=True, n_episodes=5, record=True)
