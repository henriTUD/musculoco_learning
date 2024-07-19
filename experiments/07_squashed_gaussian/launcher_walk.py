
from itertools import product
from experiment_launcher import Launcher, is_local


if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = False

    JOBLIB_PARALLEL_JOBS = 1  # or os.cpu_count() to use all cores
    N_SEEDS = 15

    if LOCAL:
        n_steps_per_epoch = 1000
    else:
        n_steps_per_epoch = 100000

    launcher = Launcher(exp_name='07_squashed_gaussian_walk',
                        exp_file='experiment',
                        n_seeds=N_SEEDS,
                        n_cores=1,
                        memory_per_core=2500,
                        days=2,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=True,
                        project_name='PROJECT_NAME',
                        partition='PARTITION'
                        )

    default_params = dict(n_epochs=250,
                          n_steps_per_epoch=n_steps_per_epoch,
                          n_epochs_save=50,
                          n_eval_episodes=10,
                          n_steps_per_fit=1000,
                          use_cuda=USE_CUDA,
                          )

    lrs = [(5e-5, 1e-5)]
    std_0s = [1.2]
    ctrl_freqs = [50]
    max_kls = [4e-2, 5e-3]
    reward_types = ['target_velocity']
    ent_coeffs = [1e-3]
    grfs = [False]
    envs = ['HumanoidMuscle.walk']

    for lr, std_0, ctrl_hz, max_kl, r_t, ent_c, grf, env in product(lrs, std_0s,
                                                                    ctrl_freqs,
                                                                    max_kls,
                                                                    reward_types,
                                                                    ent_coeffs, grfs, envs):

        if r_t == 'target_velocity':
            env_r_frac = 0.0
        else:
            env_r_frac = 0.5

        lrc, lrD = lr

        launcher.add_experiment(lrc__=lrc,
                                lrD__=lrD,
                                last_policy_activation__='identity',
                                std_0__=std_0,
                                max_kl__=max_kl,
                                ctrl_freq__=ctrl_hz,
                                reward_type__=r_t,
                                env_reward_frac__=env_r_frac,
                                env_reward_func_type__='squared',
                                env_reward_scale__=0.05,
                                standardize_obs=True,
                                policy_entr_coef__=ent_c,
                                env_id__=env,
                                **default_params)

    launcher.run(LOCAL, TEST)
