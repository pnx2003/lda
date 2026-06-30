"""
mixtures.py

Defines a registry of dataset mixtures and weights for Datasets. Each dataset is associated with
a float "sampling weight"
"""
import os
from typing import Dict, List, Tuple

def get_dataset_mixtures(data_root_dir: str, data_mix:str):
    """
    Get the dataset mixtures for the given data mix.
    """
    dataset_mixture = []
    for data_name, weight, robot_type in DATASET_NAMED_MIXTURES[data_mix]:
        dataset_path = os.path.join(data_root_dir, data_name)
        if robot_type == "egocentric_10k":
            dataset_mixture.append((data_name, weight, robot_type))
        elif os.path.exists(f'{dataset_path}/data'):
            dataset_mixture.append((data_name, weight, robot_type))
        else:
            task_list = os.listdir(dataset_path)    
            for task in task_list:
                if 'test' in task:
                    continue
                # if data_name == "RobotData/Galaxea":
                #     dataset_mixture.append((f'{data_name}/{task}/{task}', weight, robot_type))
                # else:
                if os.path.exists(f'{dataset_path}/{task}/data'):
                    dataset_mixture.append((f'{data_name}/{task}', weight, robot_type))
                else:
                    sub_task_list = os.listdir(f'{dataset_path}/{task}')
                    for sub_task in sub_task_list:
                        if os.path.exists(f'{dataset_path}/{task}/{sub_task}/data'):
                            dataset_mixture.append((f'{data_name}/{task}/{sub_task}', weight, robot_type))
    return dataset_mixture

# Dataset mixture name mapped to a list of tuples containing:
## {nakename: [(data_name, sampling_weight, robot_type)] }
DATASET_NAMED_MIXTURES = {

    "fourier_gr1_unified_1000": [
        ("gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
    ],
    "my_video_prompt_train": [
        ("your_dataset_name", 1.0, "intern_franka_video_prompt"),
    ],
    "BEHAVIOR_challenge": [
        ("BEHAVIOR_challenge", 1.0, "R1Pro"),
    ],


    "fourier_gr1_unified_1000_two_history": [
        ("gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist_twohistory"),
    ],

    "demo_data": [
        ("sim_pick_place", 1.0, "demo_data"),
    ],

    # Video prompt mixtures - 数据路径不变，只换 robot type
    "demo_data_video_prompt": [
        ("sim_pick_place", 1.0, "demo_video_prompt"),
    ],

    # Droid dataset - single-arm Franka
    "droid_video_prompt": [
        ("droid_dataset", 1.0, "droid_franka_video_prompt"),
    ],

    # Droid tiny subset for testing (10 episodes)
    "droid_tiny_video_prompt": [
        ("droid_dataset_tiny", 1.0, "droid_franka_video_prompt"),
    ],

    # Libero mujoco - single-arm Franka, 4 suites, video prompt context
    "libero_video_prompt": [
        ("libero_mujoco3.3.2/libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka_video_prompt"),
        ("libero_mujoco3.3.2/libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka_video_prompt"),
        ("libero_mujoco3.3.2/libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_video_prompt"),
        ("libero_mujoco3.3.2/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka_video_prompt"),
    ],

    # Libero mujoco - single-arm Franka, 4 suites, normal training (no video prompt).
    # Uses the plain "libero_franka" data config -> LeRobotSingleDataset.
    "libero": [
        ("libero_mujoco3.3.2/libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_mujoco3.3.2/libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_mujoco3.3.2/libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_mujoco3.3.2/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],

    # Libero single suite variants (normal, no video prompt)
    "libero_spatial": [
        ("libero_mujoco3.3.2/libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_object": [
        ("libero_mujoco3.3.2/libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_goal": [
        ("libero_mujoco3.3.2/libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_10": [
        ("libero_mujoco3.3.2/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],

    # Libero single suite variants (video prompt)
    "libero_spatial_video_prompt": [
        ("libero_mujoco3.3.2/libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka_video_prompt"),
    ],
    "libero_object_video_prompt": [
        ("libero_mujoco3.3.2/libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka_video_prompt"),
    ],
    "libero_goal_video_prompt": [
        ("libero_mujoco3.3.2/libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_video_prompt"),
    ],
    "libero_10_video_prompt": [
        ("libero_mujoco3.3.2/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka_video_prompt"),
    ],

    "intern_franka_video_prompt": [
        ("InternData-A1/sim/articulation_tasks/franka", 1.0, "intern_franka_video_prompt"),
        ("InternData-A1/sim/basic_tasks/franka", 1.0, "intern_franka_video_prompt"),
        ("InternData-A1/sim/long_horizon_tasks/franka", 1.0, "intern_franka_video_prompt"),
        ("InternData-A1/sim/pick_and_place_tasks/franka", 1.0, "intern_franka_video_prompt"),
        ("InternData-A1/sim_updated/basic_tasks/franka", 1.0, "intern_franka_video_prompt"),
        ("InternData-A1/sim_updated/articulation_tasks/franka", 1.0, "intern_franka_video_prompt"),
    ],

    "all_dataset":[
        ("world_model/data/RobotData/2025-challenge-demos", 1.0, "r1pro"),
        ("world_model/data/RobotData/AgiBotWorld-Beta2.1", 1.0, "agibot_gripper"),
        ("world_model/data/RobotData/AgiBotWorld-Beta2.1_2", 1.0, "agibot_dex"),
        ("world_model/data/RobotData/AgiBotWorld-Beta2.1_3", 1.0, "agibot_gripper"),

        ("world_model/data/RobotData/Galaxea", 1.0, "galaxea"),
        ("world_model/data/RobotData/droid", 1.0, "droid"), 
        ("world_model/data/RobotData/droid_2", 1.0, "droid"), 
        ("world_model/data/RobotData/Humanoid_everyday_10hz", 1.0, "unitree"),
        ("world_model/data/RobotData/InternData-A1/sim/articulation_tasks/franka", 1.0, "intern_franka"),
        ("world_model/data/RobotData/InternData-A1/sim/articulation_tasks/lift2", 1.0, "intern_piper"),
        ("world_model/data/RobotData/InternData-A1/sim/articulation_tasks/split_aloha", 1.0, "intern_piper"),
        ("world_model/data/RobotData/InternData-A1/real/genie1", 1.0, "intern_genie1"),

        ("world_model/data/RobotData/InternData-A1/sim/basic_tasks/franka", 1.0, "intern_franka"),
        ("world_model/data/RobotData/InternData-A1/sim/basic_tasks/genie1", 1.0, "intern_genie1"),
        ("world_model/data/RobotData/InternData-A1/sim/basic_tasks/lift2", 1.0, "intern_piper"),
        ("world_model/data/RobotData/InternData-A1/sim/basic_tasks/split_aloha", 1.0, "intern_piper"),

        ("world_model/data/RobotData/InternData-A1/sim/long_horizon_tasks/franka", 1.0, "intern_franka"),
        ("world_model/data/RobotData/InternData-A1/sim/long_horizon_tasks/lift2", 1.0, "intern_piper"),
        ("world_model/data/RobotData/InternData-A1/sim/long_horizon_tasks/split_aloha", 1.0, "intern_piper"),


        ("world_model/data/RobotData/InternData-A1/sim/pick_and_place_tasks/franka", 1.0, "intern_franka"),
        ("world_model/data/RobotData/InternData-A1/sim/pick_and_place_tasks/genie1", 1.0, "intern_genie1"),
        ("world_model/data/RobotData/InternData-A1/sim/pick_and_place_tasks/lift2", 1.0, "intern_piper"),
        ("world_model/data/RobotData/InternData-A1/sim/pick_and_place_tasks/split_aloha", 1.0, "intern_piper"),

        ("world_model/data/RobotData/InternData-A1/sim_updated/basic_tasks/franka", 1.0, "intern_franka"),
        ("world_model/data/RobotData/InternData-A1/sim_updated/basic_tasks/genie1", 1.0, "intern_genie1"),

        ("world_model/data/RobotData/InternData-A1/sim_updated/articulation_tasks/franka", 1.0, "intern_franka"),
        ("world_model/data/RobotData/InternData-A1/sim_updated/articulation_tasks/lift2", 1.0, "intern_piper"),
        ("world_model/data/RobotData/InternData-A1/sim_updated/articulation_tasks/split_aloha", 1.0, "intern_piper"),

        ("world_model/data/RobotData/open-x-embodiment", 1.0, "oxe"),

        ("world_model/data/RobotData/RoboCOIN_g1edu", 1.0, "robocoin_g1edu"),
        ("world_model/data/RobotData/RoboCOIN_leju", 1.0, "robocoin_leju"),
        ("world_model/data/RobotData/RoboCOIN_r1lite", 1.0, "robocoin_r1lite"),
        ("world_model/data/RobotData/RoboMIND_postprocessed/benchmark1_0_compressed/agilex_3rgb", 1.0, "agilex"),
        ("world_model/data/RobotData/RoboMIND_postprocessed/benchmark1_0_compressed/franka_1rgb", 1.0, "robomind_franka"),
        ("world_model/data/RobotData/RoboMIND_postprocessed/benchmark1_0_compressed/franka_3rgb", 1.0, "robomind_franka"),
        ("world_model/data/RobotData/RoboMIND_postprocessed/benchmark1_0_compressed/simulation", 1.0, "robomind_franka"),
        ("world_model/data/RobotData/RoboMIND_postprocessed/benchmark1_0_compressed/tienkung_gello_1rgb", 1.0, "tienkung_gello"),
        ("world_model/data/RobotData/RoboMIND_postprocessed/benchmark1_0_compressed/tienkung_xsens_1rgb", 1.0, "tienkung_xsens"),
        ("world_model/data/RobotData/RoboMIND_postprocessed/benchmark1_0_compressed/ur_1rgb",  1.0, "ur"),

        ("world_model/data/RobotData/RoboMIND_postprocessed/benchmark1_1_compressed/agilex_3rgb", 1.0, "agilex"),
        ("world_model/data/RobotData/RoboMIND_postprocessed/benchmark1_1_compressed/franka_3rgb", 1.0, "robomind_franka"),
        ("world_model/data/RobotData/RoboMIND_postprocessed/benchmark1_1_compressed/franka_fr3_dual", 1.0, "robomind_franka_dual"),
        ("world_model/data/RobotData/RoboMIND_postprocessed/benchmark1_1_compressed/sim_franka_3rgb", 1.0, "robomind_franka"),
        ("world_model/data/RobotData/RoboMIND_postprocessed/benchmark1_1_compressed/sim_tienkung_1rgb", 1.0, "tienkung_xsens"),
        ("world_model/data/RobotData/RoboMIND_postprocessed/benchmark1_1_compressed/tienkung_gello_1rgb", 1.0, "tienkung_gello"),
        ("world_model/data/RobotData/RoboMIND_postprocessed/benchmark1_1_compressed/tienkung_prod1_gello_1rgb", 1.0, "tienkung_gello"),
        ("world_model/data/RobotData/RoboMIND_postprocessed/benchmark1_1_compressed/tienkung_xsens_1rgb", 1.0, "tienkung_xsens"),
        ("world_model/data/RobotData/RoboMIND_postprocessed/benchmark1_1_compressed/ur_1rgb",  1.0, "ur"),

        ("world_model/data/RobotData/RoboMIND_postprocessed/benchmark1_2_compressed/franka_3rgb", 1.0, "robomind_franka"),
        
        ("world_model/data/HumanData/EgoDex", 1.0, "egodex"),
        ("world_model/data/HumanData/HOI4D", 1.0, "hoi4d"),
        ("world_model/data/HumanData/HoloAssist", 1.0, "holo_assist"),
        ("world_model/data/HumanData/hot3d", 1.0, "hot3d"),
        ("world_model/data/HumanData/oakink2", 1.0, "oakink"),
        ("world_model/data/HumanData/sea-small", 1.0, "seasmall"),
        ("world_model/data/HumanData/TACO", 1.0, "taco"),
        ("world_model/data/HumanData/TASTE-Rob", 1.0, "taste_rob"),

        ("world_model/data/HumanData/egoexo4d", 1.0, "vitra"),
        ("world_model/data/epic_kitchen", 1.0, "vitra"),
        ("world_model/data/ego4d/ego4d_cooking_and_cleaning", 1.0, "vitra"),
        ("world_model/data/ego4d/ego4d_cooking_and_cleaning_part2", 1.0, "vitra"),
        ("world_model/data/ego4d/ego4d_cooking_and_cleaning_part3", 1.0, "vitra"),
        ("world_model/data/ego4d/ego4d_cooking_and_cleaning_part4", 1.0, "vitra"),
        ("world_model/data/ego4d/ego4d_cooking_and_cleaning_part5", 1.0, "vitra"),
        ("world_model/data/ego4d/ego4d_cooking_and_cleaning_part6", 1.0, "vitra"),
        ("world_model/data/ego4d/ego4d_cooking_and_cleaning_part7", 1.0, "vitra"),
        ("world_model/data/ego4d/ego4d_cooking_and_cleaning_part8", 1.0, "vitra"),
        ("world_model/data/ego4d/ego4d_cooking_and_cleaning_part9", 1.0, "vitra"),
        ("world_model/data/ego4d/ego4d_cooking_and_cleaning_part10", 1.0, "vitra"),
        ("world_model/data/ego4d/ego4d_cooking_and_cleaning_part11", 1.0, "vitra"),
        ("world_model/data/ego4d/ego4d_other", 1.0, "vitra"),
        ("world_model/data/ego4d/ego4d_other_part2", 1.0, "vitra"),
        ("world_model/data/ego4d/ego4d_other_part3", 1.0, "vitra"),
        ("world_model/data/ego4d/ego4d_other_part4", 1.0, "vitra"),
        ("world_model/data/ego4d/ego4d_other_part5", 1.0, "vitra"),
        ("world_model/data/ego4d/ego4d_other_part6", 1.0, "vitra"),

        ("world_model/data/RobotData/RH20T", 1.0, "rh20t"),

        ("public/world_model/RawData/egocentric-10k/egocentric-10k_extracted", 1.0, "egocentric_10k")
    ]

}


# ---------------------------------------------------------------------------
# Libero video-prompt subset selection via environment variable.
#
# Lets the training shell freely choose which libero suites to use WITHOUT
# editing this file. Set env var LIBERO_VIDEO_PROMPT_EXCLUDE to a
# comma-separated list of suite short-names to exclude, e.g.
#     LIBERO_VIDEO_PROMPT_EXCLUDE=libero_10            # run spatial+object+goal
#     LIBERO_VIDEO_PROMPT_EXCLUDE=libero_goal          # run spatial+object+10
#     LIBERO_VIDEO_PROMPT_EXCLUDE=libero_spatial,libero_10   # run object+goal
# When unset/empty, all 4 suites are used (default).
#
# This overrides the "libero_video_prompt" entry above at import time, so every
# accelerate worker process picks up the same selection.
# ---------------------------------------------------------------------------
_LIBERO_VIDEO_PROMPT_ALL_SUBSETS = [
    ("libero_spatial", "libero_mujoco3.3.2/libero_spatial_no_noops_1.0.0_lerobot"),
    ("libero_object",  "libero_mujoco3.3.2/libero_object_no_noops_1.0.0_lerobot"),
    ("libero_goal",    "libero_mujoco3.3.2/libero_goal_no_noops_1.0.0_lerobot"),
    ("libero_10",      "libero_mujoco3.3.2/libero_10_no_noops_1.0.0_lerobot"),
]

_libero_exclude_raw = os.environ.get("LIBERO_VIDEO_PROMPT_EXCLUDE", "").strip()
if _libero_exclude_raw:
    _libero_excluded = {
        s.strip() for s in _libero_exclude_raw.split(",") if s.strip()
    }
else:
    _libero_excluded = set()

_libero_selected = [
    (path, 1.0, "libero_franka_video_prompt")
    for short, path in _LIBERO_VIDEO_PROMPT_ALL_SUBSETS
    if short not in _libero_excluded
]
if not _libero_selected:
    # Fallback: if everything was excluded, keep all 4 (safer than empty training).
    _libero_selected = [
        (path, 1.0, "libero_franka_video_prompt")
        for short, path in _LIBERO_VIDEO_PROMPT_ALL_SUBSETS
    ]

DATASET_NAMED_MIXTURES["libero_video_prompt"] = _libero_selected


# ---------------------------------------------------------------------------
# Libero normal (no video prompt) subset selection via environment variable.
#
# Same mechanism as the video-prompt selection above, but for the plain
# "libero" mixture (robot_type="libero_franka" -> LeRobotSingleDataset).
# Set env var LIBERO_EXCLUDE to a comma-separated list of suite short-names to
# exclude, e.g.
#     LIBERO_EXCLUDE=libero_10            # run spatial+object+goal
#     LIBERO_EXCLUDE=libero_spatial,libero_10   # run object+goal
# When unset/empty, all 4 suites are used (default).
# ---------------------------------------------------------------------------
_LIBERO_ALL_SUBSETS = _LIBERO_VIDEO_PROMPT_ALL_SUBSETS  # same 4 (short, path) pairs

_libero_plain_exclude_raw = os.environ.get("LIBERO_EXCLUDE", "").strip()
if _libero_plain_exclude_raw:
    _libero_plain_excluded = {
        s.strip() for s in _libero_plain_exclude_raw.split(",") if s.strip()
    }
else:
    _libero_plain_excluded = set()

_libero_plain_selected = [
    (path, 1.0, "libero_franka")
    for short, path in _LIBERO_ALL_SUBSETS
    if short not in _libero_plain_excluded
]
if not _libero_plain_selected:
    # Fallback: if everything was excluded, keep all 4 (safer than empty training).
    _libero_plain_selected = [
        (path, 1.0, "libero_franka")
        for short, path in _LIBERO_ALL_SUBSETS
    ]

DATASET_NAMED_MIXTURES["libero"] = _libero_plain_selected

