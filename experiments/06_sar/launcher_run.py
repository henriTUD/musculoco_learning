
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

    launcher = Launcher(exp_name='06_sar_run',
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

    lrs = [(2e-5, 1e-6)]
    std_0s = [0.8]
    ctrl_freqs = [50]
    max_kls = [1e-2]
    reward_types = ['out_of_bounds_action_cost']
    ent_coeffs = [1e-3]
    grfs = [False]
    envs = ['HumanoidMuscle.run']

    sar_init_stds = [0.8]
    sar_max_kls = [5e-3, 4e-2]
    sar_pol_entr_coefs = [1e-3]
    sar_modes = ['pcaica']
    sar_start_epochs = [15, 25]

    for lr, std_0, ctrl_hz, max_kl, r_t, ent_c, grf, env, sar_init_std, sar_max_kl, sar_entr_coef, sar_mode, sar_start in product(lrs, std_0s,
                                                                ctrl_freqs,
                                                                max_kls,
                                                                reward_types,
                                                                ent_coeffs, grfs, envs,
                                                                sar_init_stds,
                                                                sar_max_kls,
                                                                sar_pol_entr_coefs,
                                                                sar_modes,
                                                                sar_start_epochs,
                                                                ):

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
                                sar_pol_std_0__=sar_init_std,
                                sar_pol_max_kl__=sar_max_kl,
                                sar_pol_entr_coef__=sar_entr_coef,
                                sar_mode__=sar_mode,
                                n_sar_acquisition_epochs__=sar_start,
                                **default_params)

    launcher.run(LOCAL, TEST)
