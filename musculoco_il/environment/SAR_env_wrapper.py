from loco_mujoco import LocoEnv
from loco_mujoco.environments import HumanoidMuscle
from loco_mujoco.environments.humanoids.base_humanoid import BaseHumanoid
from loco_mujoco.utils import check_validity_task_mode_dataset
from mushroom_rl.core import Serializable, Core, Agent
from mushroom_rl.utils.dataset import parse_dataset

from musculoco_il.algorithms.SAR import SynergisticActionRepresentation, SAR_PCAICA, SAR_AutoEncoder
from musculoco_il.policy.random_action_policy import RandomGaussianPolicy
import matplotlib.pyplot as plt
import numpy as np


class SARWrappedMuscleHumanoid(HumanoidMuscle):

    def __init__(self, sar_module, **kwargs):
        super().__init__(**kwargs)
        self.sar_module = sar_module

    def _preprocess_action(self, action):
        if len(action.shape) == 1:
            action_in = np.expand_dims(action.copy(), axis=0)
        else:
            assert len(action.shape) == 2
            action_in = action.copy()

        muscle_space_action = self.sar_module.synergistic_to_action(action_in)

        unnormalized_action = ((muscle_space_action * self.norm_act_delta) + self.norm_act_mean)
        return unnormalized_action

    @staticmethod
    def generate(task="walk", dataset_type="real", **kwargs):

        check_validity_task_mode_dataset(HumanoidMuscle.__name__, task, None, dataset_type,
                                         *HumanoidMuscle.valid_task_confs.get_all())

        if dataset_type == "real":
            if task == "walk":
                path = "datasets/humanoids/real/02-constspeed_reduced_humanoid.npz"
            elif task == "run":
                path = "datasets/humanoids/real/05-run_reduced_humanoid.npz"
        elif dataset_type == "perfect":
            if "use_foot_forces" in kwargs.keys():
                assert kwargs["use_foot_forces"] is False
            if "disable_arms" in kwargs.keys():
                assert kwargs["disable_arms"] is True
            if "use_box_feet" in kwargs.keys():
                assert kwargs["use_box_feet"] is True

            if task == "walk":
                path = "datasets/humanoids/perfect/humanoid_muscle_walk/perfect_expert_dataset_det.npz"

        return BaseHumanoid.generate(SARWrappedMuscleHumanoid, path, task, dataset_type, **kwargs)
