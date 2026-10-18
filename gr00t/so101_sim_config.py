# GR00T N1.7 modality config for the Isaac Lab SO-101 pick-and-place dataset.
# Identical to Isaac-GR00T/examples/SO100/so100_config.py (front + wrist cameras, 5 arm joints +
# gripper, 16-step action horizon). Kept in this project so the dataset, the fine-tune and the
# policy server all point at one file. NOTE: the action horizon (delta_indices) is baked into the
# diffusion head at training time and cannot be changed at inference (NVIDIA e2e workflow docs).
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

ACTION_HORIZON = 16

so101_sim_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["front", "wrist"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["single_arm", "gripper"]),
    "action": ModalityConfig(
        delta_indices=list(range(0, ACTION_HORIZON)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            # arm joints: RELATIVE (delta from current state) generalises better
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            # gripper: ABSOLUTE target works better for open/close
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(so101_sim_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
