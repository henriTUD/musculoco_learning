from loco_mujoco import LocoEnv
from loco_mujoco.environments import HumanoidMuscle
from loco_mujoco.environments.humanoids.base_humanoid import BaseHumanoid
from loco_mujoco.utils import check_validity_task_mode_dataset


class UnnormalizedActionSpaceHumanoid(HumanoidMuscle):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _preprocess_action(self, action):
        return action

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

        return BaseHumanoid.generate(UnnormalizedActionSpaceHumanoid, path, task, dataset_type, **kwargs)