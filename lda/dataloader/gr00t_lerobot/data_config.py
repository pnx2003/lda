# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#


from abc import ABC, abstractmethod

from lda.dataloader.gr00t_lerobot.datasets import ModalityConfig
from lda.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform, ModalityTransform
from lda.dataloader.gr00t_lerobot.transform.concat import ConcatTransform
from lda.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from lda.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
# from gr00t.model.transforms import GR00TTransform


class BaseDataConfig(ABC):
    video_backend = "torchvision_av"
    video_keys = ["video.top_head"]
    future_video_keys = [
        "future_video.top_head"
    ]
    state_keys = [
        "state.left_eef_position",
        "state.left_eef_rotation",
        "state.left_gripper",
        "state.right_eef_position",
        "state.right_eef_rotation",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_eef_position",
        "action.left_eef_rotation",
        "action.left_gripper",
        "action.right_eef_position",
        "action.right_eef_rotation",
        "action.right_gripper",
    ]
    language_keys = ["annotation.language.action_text"]
    observation_indices = [-5, 0]
    future_observation_indices = [5]
    history_action_indices = list(range(-5, 0))
    action_indices = list(range(-5, 17))
    img_interval = 3


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        future_video_modality = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
            "future_video": future_video_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            # StateActionSinCosTransform(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "q99" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_eef_position": "q99", 
                    "action.right_eef_position": "q99",
                    "action.left_eef_rotation": "q99",
                    "action.right_eef_rotation": "q99",
                    "action.left_gripper": "q99",
                    "action.right_gripper": "q99",
                    },
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class HumanBaseDataConfig(ABC):
    video_backend = "decord"
    video_keys = ["video.top_head"]
    future_video_keys = [
        "future_video.top_head"
    ]
    state_keys = [
        "state.left_eef_position",
        "state.left_eef_rotation",
        "state.right_eef_position",
        "state.right_eef_rotation",
        "state.left_mano_hand",
        "state.right_mano_hand",
    ]
    action_keys = [
        "action.left_eef_position",
        "action.left_eef_rotation",
        "action.left_mano_hand",
        "action.right_eef_position",
        "action.right_eef_rotation",
        "action.right_mano_hand",
    ]
    language_keys = ["annotation.language.action_text"]
    observation_indices = [-5, 0]
    future_observation_indices = [5]
    history_action_indices = list(range(-5, 0)) # indicate which part is history action
    action_indices = list(range(-5, 17))
    img_interval = 3


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        future_video_modality = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
            "future_video": future_video_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "q99" for key in self.action_keys},
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)
###########################################################################################

class FourierGr1ArmsWaist_twohistoryDataConfig:
    video_keys = ["video.ego_view"]
    future_video_keys = [
        "future_video.ego_view"
    ]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [-5, 0]
    future_observation_indices = [16]
    action_indices = list(range(16))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        future_video_modality = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
            "future_video": future_video_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class FourierGr1ArmsWaistDataConfig:
    video_keys = ["video.ego_view"]
    future_video_keys = [
        "future_video.ego_view"
    ]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    future_observation_indices = [16]
    action_indices = list(range(16))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        future_video_modality = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
            "future_video": future_video_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)

###########################################################################################
class FourierGr1EEFDataConfig:
    video_keys = ["video.ego_view"]
    future_video_keys = [
        "future_video.ego_view"
    ]
    state_keys = [
        "state.left_eef_position",
        "state.left_eef_rotation",
        "state.right_eef_position",
        "state.right_eef_rotation",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = [
        "action.left_eef_position",
        "action.left_eef_rotation",
        "action.right_eef_position",
        "action.right_eef_rotation",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [-5, 0]
    history_action_indices = list(range(-5, 0))
    future_observation_indices = [5]
    action_indices = list(range(-5, 17))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        future_video_modality = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
            "future_video": future_video_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

# RobotDataset

class AgibotWorldDataConfig(BaseDataConfig):
    pass

class AgibotDexDataConfig(BaseDataConfig):
    video_backend = "torchvision_av"

class GalaxeaDataConfig(BaseDataConfig):
    img_interval = 5
    video_backend = "torchvision_av"
    pass

class DroidDataConfig(BaseDataConfig):
    pass

class HumanoidEverydayDataConfig(BaseDataConfig):
    pass

class InternDataConfig(BaseDataConfig):
    pass

class FrankaDataConfig(BaseDataConfig):
    state_keys = [
        "state.left_eef_position",
        "state.left_eef_rotation",
        "state.left_gripper",
    ]
    action_keys = [
        "action.left_eef_position",
        "action.left_eef_rotation",
        "action.left_gripper",
    ]
    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            # StateActionSinCosTransform(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "q99" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_eef_position": "q99", 
                    "action.left_eef_rotation": "q99",
                    "action.left_gripper": "binary",
                    },
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class OxeDataConfig(BaseDataConfig):
    pass

class RoboCoin_g1eduDataConfig(BaseDataConfig):
    video_backend = "torchvision_av"
    pass

class RoboCoin_lejuDataConfig(BaseDataConfig):
    pass

class RoboCoin_r1liteDataConfig(BaseDataConfig):
    pass

class RobomindDataConfig(BaseDataConfig):
    pass

class Challange2025DataConfig(BaseDataConfig):
    pass

class RH20TDataConfig(BaseDataConfig):
    pass

# Human Data Config
class VitraDataConfig(HumanBaseDataConfig):
    pass

class EgodexDataConfig(HumanBaseDataConfig):
    video_backend = "torchvision_av"
    state_keys = [
        "state.left_eef_position",
        "state.left_eef_rotation",
        "state.right_eef_position",
        "state.right_eef_rotation",
    ]
    action_keys = [
        "action.left_eef_position",
        "action.left_eef_rotation",
        "action.right_eef_position",
        "action.right_eef_rotation",
    ]

class Hoi4dDataConfig(HumanBaseDataConfig):
    pass

class HoloAssitDataConfig(HumanBaseDataConfig):
    pass

class hot3dDataConfig(HumanBaseDataConfig):
    pass

class oakinkDataConfig(HumanBaseDataConfig):
    video_backend = "torchvision_av"

class seasmallDataConfig(HumanBaseDataConfig):
    video_backend = "torchvision_av"

class TacoDataConfig(HumanBaseDataConfig):
    video_backend = "torchvision_av"
    state_keys = [
        "state.left_eef_position",
        "state.left_eef_rotation",
        "state.right_eef_position",
        "state.right_eef_rotation",
    ]
    action_keys = [
        "action.left_eef_position",
        "action.left_eef_rotation",
        "action.right_eef_position",
        "action.right_eef_rotation",
    ]

class TASTE_robDataConfig(HumanBaseDataConfig):
    video_backend = "decord"
    video_keys = ["video.top_head"]
    future_video_keys = [
        "future_video.top_head"
    ]
    language_keys = ["annotation.language.action_text"]


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        future_video_modality = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "language": language_modality,
            "future_video": future_video_modality,
        }
        return modality_configs
    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class EgoCentric10KDataConfig(HumanBaseDataConfig):
    video_backend = "decord"
    target_fps = 10
    video_keys = ["video.top_head"]
    future_video_keys = [
        "future_video.top_head"
    ]
    state_keys = [
        "state.qpos",
    ]
    action_keys = [
        "action.qpos",
    ]
    language_keys = ["annotation.language.action_text"]


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        future_video_modality = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "language": language_modality,
            "future_video": future_video_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class DemoDataConfig:
    video_backend = "torchvision_av"
    video_keys = ["video.ego_view"]
    future_video_keys = [
        "future_video.ego_view"
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_width",
    ]
    action_keys = [
        "action.eef_position",
        "action.eef_rotation",
        "action.gripper_width",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [-1, 0]
    future_observation_indices = [5]
    action_indices = list(range(0, 16))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        future_video_modality = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
            "future_video": future_video_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            # StateActionSinCosTransform(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "q99" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "q99" for key in self.action_keys}
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)

# Video Prompt DataConfig classes
class FrankaVideoPromptDataConfig(FrankaDataConfig):
    use_video_prompt = True
    num_support_demos = 2
    num_support_frames = 4
    wrong_prompt_prob = 0.0
    prompt_mode = "correct"


class InternFrankaVideoPromptDataConfig(FrankaDataConfig):
    use_video_prompt = True
    num_support_demos = 2
    num_support_frames = 4
    wrong_prompt_prob = 0.0
    prompt_mode = "correct"


class FourierGr1VideoPromptDataConfig(FourierGr1ArmsWaistDataConfig):
    use_video_prompt = True
    num_support_demos = 2
    num_support_frames = 4
    wrong_prompt_prob = 0.0
    prompt_mode = "correct"


class AgibotVideoPromptDataConfig(AgibotWorldDataConfig):
    use_video_prompt = True
    num_support_demos = 2
    num_support_frames = 4
    wrong_prompt_prob = 0.0
    prompt_mode = "correct"


class DroidVideoPromptDataConfig(DroidDataConfig):
    use_video_prompt = True
    num_support_demos = 2
    num_support_frames = 4
    wrong_prompt_prob = 0.0
    prompt_mode = "correct"


class DroidFrankaDataConfig:
    """DataConfig matching the actual droid_dataset modality.json: single-arm Franka, state=8, action=7."""
    video_backend = "torchvision_av"
    video_keys = ["video.exterior_1"]
    future_video_keys = [
        "future_video.exterior_1"
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper",
    ]
    action_keys = [
        "action.eef_position",
        "action.eef_rotation",
        "action.gripper",
    ]
    language_keys = ["annotation.language"]
    observation_indices = [-5, 0]
    future_observation_indices = [5]
    history_action_indices = list(range(-5, 0))
    action_indices = list(range(-5, 17))
    img_interval = 3

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        future_video_modality = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
            "future_video": future_video_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "q99" for key in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "q99" for key in self.action_keys},
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class DroidFrankaVideoPromptDataConfig(DroidFrankaDataConfig):
    use_video_prompt = True
    num_support_demos = 2
    num_support_frames = 4
    wrong_prompt_prob = 0.0
    prompt_mode = "correct"


class DemoVideoPromptDataConfig(DemoDataConfig):
    use_video_prompt = True
    num_support_demos = 2
    num_support_frames = 4
    wrong_prompt_prob = 0.0
    prompt_mode = "correct"


class LiberoFrankaDataConfig:
    """DataConfig for libero mujoco lerobot datasets: single-arm Franka, state=8, action=7.

    The libero modality.json uses subkeys x/y/z/roll/pitch/yaw/pad/gripper for state
    and x/y/z/roll/pitch/yaw/gripper for action. original_key defaults to
    'observation.state' / 'action' via the pydantic schema, matching the parquet
    columns. The `pad` dim (state index 6) is dropped as it is unused padding,
    giving state_dim=7.
    """
    video_backend = "torchvision_av"
    video_keys = ["video.primary_image", "video.wrist_image"]
    future_video_keys = ["future_video.primary_image", "future_video.wrist_image"]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [-2, 0]
    future_observation_indices = [16]
    history_action_indices = list(range(-2, 0))
    # action_indices must cover history + future_action_window_size+1 steps.
    # The mixture __getitem__ slices off the first `len(history_action_indices)` (=2)
    # steps as history_action, leaving the rest as the action target. The model then
    # takes the last (future_action_window_size+1)=16 steps. So we need >= 18 raw
    # steps -> range(-2, 16) = 18 steps, sliced to 16, matching action_horizon=16.
    action_indices = list(range(-2, 16))
    img_interval = 1

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        future_video_modality = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
            "future_video": future_video_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "q99" for key in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "q99" for key in self.action_keys},
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class LiberoFrankaVideoPromptDataConfig(LiberoFrankaDataConfig):
    use_video_prompt = True
    num_support_demos = 2
    num_support_frames = 4
    wrong_prompt_prob = 0.0
    prompt_mode = "correct"


ROBOT_TYPE_CONFIG_MAP = {
    "fourier_gr1_arms_waist": FourierGr1ArmsWaistDataConfig(),
    "fourier_gr1_eef": FourierGr1EEFDataConfig(),
    "fourier_gr1_arms_waist_twohistory": FourierGr1ArmsWaist_twohistoryDataConfig(),

    "agibot_gripper": AgibotWorldDataConfig(),
    "agibot_dex": AgibotDexDataConfig(),
    "galaxea": GalaxeaDataConfig(),
    "droid": DroidDataConfig(),
    "unitree": HumanoidEverydayDataConfig(),
    "intern_franka": FrankaDataConfig(),
    "intern_piper": InternDataConfig(),
    "intern_genie1": InternDataConfig(),
    "oxe": OxeDataConfig(),
    "robocoin_g1edu": RoboCoin_g1eduDataConfig(),
    "robocoin_leju": RoboCoin_lejuDataConfig(),
    "robocoin_r1lite": RoboCoin_r1liteDataConfig(),
    "ur": FrankaDataConfig(),
    "agilex":RobomindDataConfig(),
    "robomind_franka": FrankaDataConfig(),
    "robomind_franka_640": FrankaDataConfig(),
    "robomind_franka_dual": RobomindDataConfig(),

    "tienkung_gello": RobomindDataConfig(),
    "tienkung_xsens": RobomindDataConfig(),
    "r1pro": Challange2025DataConfig(),

    "egodex": EgodexDataConfig(),
    "hoi4d": Hoi4dDataConfig(),
    "holo_assist": HoloAssitDataConfig(),
    "hot3d": hot3dDataConfig(),
    "oakink": oakinkDataConfig(),
    "seasmall": seasmallDataConfig(),
    "taco": TacoDataConfig(),
    "taste_rob": TASTE_robDataConfig(),
    "egocentric_10k": EgoCentric10KDataConfig(),
    "vitra": VitraDataConfig(),
    "rh20t": RH20TDataConfig(),

    "demo_data": DemoDataConfig(),

    # Video prompt configs
    "intern_franka_video_prompt": InternFrankaVideoPromptDataConfig(),
    "franka_video_prompt": FrankaVideoPromptDataConfig(),
    "fourier_gr1_video_prompt": FourierGr1VideoPromptDataConfig(),
    "agibot_video_prompt": AgibotVideoPromptDataConfig(),
    "droid_video_prompt": DroidVideoPromptDataConfig(),
    "droid_franka": DroidFrankaDataConfig(),
    "droid_franka_video_prompt": DroidFrankaVideoPromptDataConfig(),
    "demo_video_prompt": DemoVideoPromptDataConfig(),
    "libero_franka": LiberoFrankaDataConfig(),
    "libero_franka_video_prompt": LiberoFrankaVideoPromptDataConfig(),
}
