# Exciting Action: Investigating Efficient Exploration for Learning Musculoskeletal Humanoid Locomotion

Repository for the paper ["Exciting Action: Investigating Efficient Exploration for Learning Musculoskeletal Humanoid Locomotion"](https://arxiv.org/pdf/2407.11658).

__Abstract:__ Learning a locomotion controller for a muscu- loskeletal system is challenging due to over-actuation and high- dimensional action space. While many reinforcement learning methods attempt to address this issue, they often struggle to learn human-like gaits because of the complexity involved in engineering an effective reward function. In this paper, we demonstrate that adversarial imitation learning can address this issue by analyzing key problems and providing solutions using both current literature and novel techniques. We vali- date our methodology by learning walking and running gaits on a simulated humanoid model with 16 degrees of free- dom and 92 Muscle-Tendon Units, achieving natural-looking gaits with only a few demonstrations.

<p align="center">
  <img src="https://github.com/henriTUD/musculoco_learning/blob/main/assets/exciting_action_supp_gif.gif" width=500>
</p>

---
## Project Structure

In this repository we strictly separate between the code for the policies and objectives
described in the paper and the experiments that work with them:

- The code for all experiments can be found [here](experiments/).
- Implementation of the [squashed gaussian policy](musculoco_il/policy/squashed_gaussian_torch_policy.py).
- Implementation of the [beta distribution policy](musculoco_il/policy/beta_distribution_torch_policy.py) with different types of parameterization.
- Implementation of [Latent Exploration](musculoco_il/policy/latent_exploration_torch_policy.py).
- Implementation of [SAR](musculoco_il/algorithms/SAR.py) as adapted from [here](https://github.com/MyoHub/myosuite/blob/main/docs/source/tutorials/SAR/SAR_tutorial.ipynb).
- Implementation of the [Uniform KL Divergence Objective](musculoco_il/algorithms/GAIL_KL_objective.py).



---
## Installation

1. Create a new conda environment and activate it:

```bash
conda create --name musculoco_paper python=3.8
conda activate musculoco_paper
```

2. Install the [Experiment Launcher](https://git.ias.informatik.tu-darmstadt.de/common/experiment_launcher) from source. 

3. Install the [imitation_lib](https://github.com/robfiras/ls-iq) from source. It provides the GAIL implementation used throughout this work.

4. Clone the [LocoMuJoCo repository](https://github.com/robfiras/loco-mujoco) and install the **muscle_act_obs** branch from source. Using this specific branch will not be necessary in later versions.

5. Download the motion capture datasets with the following command. This will take up about 4.5 GB of disc space.

```bash
loco-mujoco-download-real
```

6. Install torch: 

```bash
pip install torch
```

7. Install the code as a python package by running the following in the top directory of this repository.

```bash
pip install -e .
```


---
## Running Experiments

Now with your conda environment activated, you can run any experiment by cd-ing into its directory and running for instance:

```bash
python launcher_walk.py
```

Note: The amount of __steps_per_epoch__ currently set for local running is just for testing the code. All experiments were run with 100.000 __steps_per_epoch__.


---
## Trained Policies

- TBD
