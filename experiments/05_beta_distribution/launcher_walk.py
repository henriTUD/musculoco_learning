
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

    launcher = Launcher(exp_name='05_beta_dist_walk',
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

    # TODO: All hypers (add beta specific ones)
    lrs = [(5e-5, 1e-5)]
    ctrl_freqs = [50]
    max_kls = [1e-3, 4e-2]
    softplus_offsets = [-2.5]
    reward_types = ['target_velocity']
    ent_coeffs = [1e-3]
    grfs = [False]
    envs = ['HumanoidMuscle.walk']
    ab_offs = [1.0, 0.0]

    for lr, ctrl_hz, max_kl, r_t, ent_c, grf, env, so, ab_off in product(lrs,
                                                                ctrl_freqs,
                                                                max_kls,
                                                                reward_types,
                                                                ent_coeffs, grfs, envs,
                                                                softplus_offsets,
                                                                ab_offs):

        if r_t == 'target_velocity':
            env_r_frac = 0.0
        else:
            env_r_frac = 0.5

        lrc, lrD = lr

        launcher.add_experiment(lrc__=lrc,
                                lrD__=lrD,
                                last_policy_activation__='identity',
                                max_kl__=max_kl,
                                ctrl_freq__=ctrl_hz,
                                reward_type__=r_t,
                                env_reward_frac__=env_r_frac,
                                env_reward_func_type__='squared',
                                env_reward_scale__=0.05,
                                standardize_obs=True,
                                policy_entr_coef__=ent_c,
                                env_id__=env,
                                softplus_offset__=so,
                                ab_offset__=ab_off,
                                **default_params)

    launcher.run(LOCAL, TEST)
