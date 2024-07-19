import os

from time import perf_counter
from contextlib import contextmanager

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length, parse_dataset
from mushroom_rl.core.logger.logger import Logger

from imitation_lib.imitation import GAIL_TRPO
from imitation_lib.utils import FullyConnectedNetwork, DiscriminatorNetwork, NormcInitializer, \
                         GailDiscriminatorLoss
from imitation_lib.utils import BestAgentSaver
from experiment_launcher import run_experiment

from mushroom_rl.core.serialization import *

from loco_mujoco import LocoEnv

import time

from musculoco_il.environment.unnormalized_action_space import UnnormalizedActionSpaceHumanoid
from musculoco_il.policy.beta_distribution_torch_policy import BetaDistributionTorchPolicy
from musculoco_il.policy.gaussian_torch_policy import OptionalGaussianTorchPolicy
from musculoco_il.util.preprocessors import StateSelectionPreprocessor
from musculoco_il.util.rewards import OutOfBoundsActionCost
from musculoco_il.util.standardizer import Standardizer


def initial_log(core, tb_writer, logger_stoch, logger_deter, n_eval_episodes, gamma):
    epoch = 0
    dataset = core.evaluate(n_episodes=n_eval_episodes)
    J_mean = np.mean(compute_J(dataset))
    tb_writer.add_scalar("Eval_J", J_mean, epoch)
    with catchtime() as t:
        print('Epoch %d | Time %fs ' % (epoch + 1, float(t())))

    # evaluate with deterministic policy
    core.agent.policy.deterministic = True
    dataset = core.evaluate(n_episodes=n_eval_episodes)
    R_mean = np.mean(compute_J(dataset))
    J_mean = np.mean(compute_J(dataset, gamma=gamma))
    L = np.mean(compute_episodes_length(dataset))
    logger_deter.log_numpy(Epoch=epoch, R_mean=R_mean, J_mean=J_mean, L=L)
    tb_writer.add_scalar("Eval_R-deterministic", R_mean, epoch)
    tb_writer.add_scalar("Eval_J-deterministic", J_mean, epoch)
    tb_writer.add_scalar("Eval_L-deterministic", L, epoch)

    # evaluate with stochastic policy
    core.agent.policy.deterministic = False
    dataset = core.evaluate(n_episodes=n_eval_episodes)
    s, *_ = parse_dataset(dataset)
    R_mean = np.mean(compute_J(dataset))
    J_mean = np.mean(compute_J(dataset, gamma=gamma))
    L = np.mean(compute_episodes_length(dataset))
    E = core.agent.policy.entropy(s)

    logger_stoch.log_numpy(Epoch=epoch, R_mean=R_mean, J_mean=J_mean, L=L, E=E)
    tb_writer.add_scalar("Eval_R-stochastic", R_mean, epoch)
    tb_writer.add_scalar("Eval_J-stochastic", J_mean, epoch)
    tb_writer.add_scalar("Eval_L-stochastic", L, epoch)


def build_agent(mdp, expert_data, use_cuda, discrim_obs_mask, train_D_n_th_epoch=3,
                lrc=1e-3, lrD=0.0003, sw=None, policy_entr_coef=0.0,
                use_noisy_targets=False, last_policy_activation="identity",
                use_next_states=True, max_kl=5e-3, d_entr_coef=1e-3,
                env_reward_frac=0.0, standardize_obs=True, softplus_offset=0.6,
                ab_offset=1.):
    mdp_info = deepcopy(mdp.info)

    trpo_standardizer = Standardizer(use_cuda=use_cuda) if standardize_obs else None

    print("Action DIM:")
    print(mdp_info.action_space.shape)
    print("OBS DIM:")
    print(mdp_info.observation_space.shape)
    print("LAST ACTIVATION:")
    print(last_policy_activation)

    feature_dims = [512, 256]

    policy_params = dict(network=FullyConnectedNetwork,
                         input_shape=(len(discrim_obs_mask),),
                         output_shape=mdp_info.action_space.shape,
                         n_features=feature_dims,
                         initializers=[NormcInitializer(1.0), NormcInitializer(1.0), NormcInitializer(0.001)],
                         activations=['relu', 'relu', last_policy_activation],
                         standardizer=trpo_standardizer,
                         softplus_offset=softplus_offset,
                         ab_offset=ab_offset,
                         use_cuda=use_cuda)

    critic_params = dict(network=FullyConnectedNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': lrc,
                                               'weight_decay': 0.0}},
                         loss=F.mse_loss,
                         batch_size=256,
                         input_shape=(len(discrim_obs_mask),),
                         activations=['relu', 'relu', 'identity'],
                         standardizer=trpo_standardizer,
                         squeeze_out=False,
                         output_shape=(1,),
                         initializers=[NormcInitializer(1.0), NormcInitializer(1.0), NormcInitializer(0.001)],
                         n_features=[512, 256],
                         use_cuda=use_cuda)

    # remove hip rotations
    discrim_act_mask = []  # if disc_only_state else np.arange(mdp_info.action_space.shape[0])
    discrim_input_shape = (2 * len(discrim_obs_mask),) if use_next_states else (len(discrim_obs_mask),)
    discrim_standardizer = Standardizer() if standardize_obs else None

    discriminator_params = dict(optimizer={'class': optim.Adam,
                                           'params': {'lr': lrD,
                                                      'weight_decay': 0.0}},
                                batch_size=2000,
                                network=DiscriminatorNetwork,
                                use_next_states=use_next_states,
                                input_shape=discrim_input_shape,
                                output_shape=(1,),
                                squeeze_out=False,
                                n_features=[512, 256],
                                initializers=None,
                                activations=['tanh', 'tanh', 'identity'],
                                standardizer=discrim_standardizer,
                                use_actions=False,
                                use_cuda=use_cuda)

    alg_params = dict(train_D_n_th_epoch=train_D_n_th_epoch,
                      state_mask=discrim_obs_mask,
                      act_mask=discrim_act_mask,
                      n_epochs_cg=25,
                      trpo_standardizer=trpo_standardizer,
                      D_standardizer=discrim_standardizer,
                      loss=GailDiscriminatorLoss(entcoeff=d_entr_coef),
                      ent_coeff=policy_entr_coef,
                      use_noisy_targets=use_noisy_targets,
                      max_kl=max_kl,
                      use_next_states=use_next_states,
                      env_reward_frac=env_reward_frac)

    print(f'USE_NEXT_STATES: {use_next_states}')
    print(f'ENV_REWARD_FRAC: {env_reward_frac}')

    agent = GAIL_TRPO(mdp_info=mdp_info, policy_class=BetaDistributionTorchPolicy, policy_params=policy_params, sw=sw,
                      discriminator_params=discriminator_params, critic_params=critic_params,
                      demonstrations=expert_data, **alg_params)
    return agent


def experiment(n_epochs: int = 500,
               n_steps_per_epoch: int = 10000,
               n_steps_per_fit: int = 1024,
               n_eval_episodes: int = 50,
               n_epochs_save: int = 500,
               horizon: int = 1000,
               gamma: float = 0.99,
               policy_entr_coef: float = 1e-3,
               train_D_n_th_epoch: int = 3,
               lrc: float = 1e-3,
               lrD: float = 0.0003,
               last_policy_activation: str = "identity",
               use_noisy_targets: bool = False,
               use_next_states: bool = False,
               use_cuda: bool = False,
               results_dir: str = './logs',
               max_kl: float = 5e-3,
               d_entr_coef: float = 1e-3,
               env_freq: int = 1000,
               ctrl_freq: int = 100,
               reward_type: str = 'target_velocity',
               env_reward_frac: float = 0.0,
               env_reward_scale: float = 1.0,
               env_reward_func_type: str = 'abs',
               reward_action_mean: float = 0.5,
               reward_const_cost: float = 0.0,
               standardize_obs: bool = True,
               env_id: str = 'Atlas.walk',
               softplus_offset: float = 0.6,
               ab_offset: float = 1.,
               seed: int = 0):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    results_dir = os.path.join(results_dir, str(seed))

    # logging stuff
    logger_stoch = Logger(results_dir=results_dir, log_name="stochastic_logging", seed=seed, append=True)
    logger_deter = Logger(results_dir=results_dir, log_name="deterministic_logging", seed=seed, append=True)

    tb_writer = SummaryWriter(log_dir=results_dir)
    agent_saver = BestAgentSaver(save_path=results_dir, n_epochs_save=n_epochs_save)

    # define env and data frequencies
    n_substeps = env_freq // ctrl_freq

    if reward_type == 'target_velocity':
        print(f'Using custom Reward: {reward_type}')
        if '4Ages' not in env_id:
            reward_type_loco = 'target_velocity'
        else:
            raise NotImplementedError
        if 'walk' in env_id:
            reward_params = dict(target_velocity=1.25)
        elif 'run' in env_id:
            reward_params = dict(target_velocity=2.5)
    elif reward_type == 'out_of_bounds_action_cost':
        print(f'Using custom Reward: {reward_type}')
        reward_type_loco = 'custom'
        reward_callback = OutOfBoundsActionCost(0.0, 1.0, reward_scale=env_reward_scale,
                                                const_cost=reward_const_cost,
                                                func_type=env_reward_func_type,)
        reward_params = dict(reward_callback=reward_callback)
    elif reward_type == 'action_cost':
        print(f'Using custom Reward: {reward_type}')
        reward_type_loco = 'custom'
        raise NotImplementedError
    else:
        raise Exception(f'{reward_type} is not a valid reward type!')

    mdp = UnnormalizedActionSpaceHumanoid.generate(#env_id,
                                               task=env_id.split(".")[1],
                                               gamma=gamma,
                                               horizon=horizon,
                                               n_substeps=n_substeps,
                                               timestep=1/env_freq,
                                               reward_type=reward_type_loco,
                                               reward_params=reward_params,
                                               muscle_force_scaling=1.25,
                                               )

    print('Env Action Scaling:')
    print(mdp.norm_act_mean)
    print(mdp.norm_act_delta)

    test_mdp = UnnormalizedActionSpaceHumanoid.generate(#env_id,
                                                    task=env_id.split(".")[1],
                                                    gamma=gamma,
                                                    horizon=horizon,
                                                    n_substeps=n_substeps,
                                                    timestep=1 / env_freq,
                                                    muscle_force_scaling=1.25
                                                    )

    print(f'DT: {mdp.dt}')
    print(f'ENV_DT: {mdp._timestep}')

    # create a dataset
    unavailable_keys = ["q_pelvis_tx", "q_pelvis_tz"]
    expert_data = mdp.create_dataset(ignore_keys=unavailable_keys)

    discrim_obs_mask = mdp.get_kinematic_obs_mask()

    print("Discrim Obs Mask:")
    print(len(discrim_obs_mask))

    # create agent and core
    agent = build_agent(mdp=mdp, expert_data=expert_data, use_cuda=use_cuda,
                        train_D_n_th_epoch=train_D_n_th_epoch, lrc=lrc,
                        lrD=lrD, sw=tb_writer, policy_entr_coef=policy_entr_coef,
                        use_noisy_targets=use_noisy_targets, use_next_states=use_next_states,
                        last_policy_activation=last_policy_activation, discrim_obs_mask=discrim_obs_mask,
                        max_kl=max_kl, d_entr_coef=d_entr_coef, env_reward_frac=env_reward_frac,
                        standardize_obs=standardize_obs, softplus_offset=softplus_offset, ab_offset=ab_offset)

    core = Core(agent, mdp)
    test_core = Core(agent, test_mdp)

    if '4Ages' in env_id:
        agent.add_preprocessor(StateSelectionPreprocessor(first_n=len(discrim_obs_mask)))

    #test_core.evaluate(n_episodes=50, render=True)

    assert agent.policy._network.model.network._stand is not None

    if agent.policy._network.model.network._stand is not None:
        agent.policy._network.model.network._stand.freeze()

    initial_log(test_core, tb_writer, logger_stoch, logger_deter, n_eval_episodes, gamma)

    if agent.policy._network.model.network._stand is not None:
        agent.policy._network.model.network._stand.unfreeze()

    # gail train loop
    for epoch in range(1, n_epochs):
        with catchtime() as t:
            start = time.time()
            core.agent.policy.deterministic = False
            core.learn(n_steps=n_steps_per_epoch, n_steps_per_fit=n_steps_per_fit, quiet=True, render=False)

            done = time.time()
            elapsed = done - start
            tb_writer.add_scalar("time_in_learn", elapsed, epoch)

            dataset = test_core.evaluate(n_episodes=n_eval_episodes)
            J_mean = np.mean(compute_J(dataset))
            tb_writer.add_scalar("Eval_J", J_mean, epoch)
            agent_saver.save(core.agent, J_mean)
            print('Epoch %d | Time %fs ' % (epoch + 1, float(t())))

            #### evaluate with deterministic policy
            core.agent.policy.deterministic = True

            if agent.policy._network.model.network._stand is not None:
                agent.policy._network.model.network._stand.freeze()

            dataset = test_core.evaluate(n_episodes=n_eval_episodes)
            R_mean = np.mean(compute_J(dataset))
            J_mean = np.mean(compute_J(dataset, gamma=gamma))
            L = np.mean(compute_episodes_length(dataset))
            logger_deter.log_numpy(Epoch=epoch, R_mean=R_mean, J_mean=J_mean, L=L)
            tb_writer.add_scalar("Eval_R-deterministic", R_mean, epoch)
            tb_writer.add_scalar("Eval_J-deterministic", J_mean, epoch)
            tb_writer.add_scalar("Eval_L-deterministic", L, epoch)

            #### evaluate with stochastic policy
            core.agent.policy.deterministic = False
            dataset = test_core.evaluate(n_episodes=n_eval_episodes)
            s, *_ = parse_dataset(dataset)
            R_mean = np.mean(compute_J(dataset))
            J_mean = np.mean(compute_J(dataset, gamma=gamma))
            L = np.mean(compute_episodes_length(dataset))
            E = agent.policy.entropy(s)

            tb_writer.add_scalar("Eval_R-stochastic", R_mean, epoch)
            tb_writer.add_scalar("Eval_J-stochastic", J_mean, epoch)
            tb_writer.add_scalar("Eval_L-stochastic", L, epoch)

            ### evaluate on shaped reward
            dataset = core.evaluate(n_episodes=n_eval_episodes)
            shaped_R_mean = np.mean(compute_J(dataset))
            tb_writer.add_scalar("Shaped_R-stochastic", shaped_R_mean, epoch)
            logger_stoch.log_numpy(Epoch=epoch, R_mean=R_mean, J_mean=J_mean, L=L, E=E,
                                   Shaped_R=shaped_R_mean)

            if agent.policy._network.model.network._stand is not None:
                agent.policy._network.model.network._stand.unfreeze()

    agent_saver.save_curr_best_agent()
    print("Finished.")


@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


if __name__ == "__main__":
    run_experiment(experiment)