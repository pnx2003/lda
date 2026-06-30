#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Offline evaluation for video-prompt ablation.

Runs open-loop action prediction on held-out trajectories under different
prompt modes (none / correct / wrong / shuffled / final_frame) and records
per-trajectory action L1 losses.

Usage:
  python lda/eval/eval_LDA_video_prompt.py \
    --config_yaml lda/config/training/LDA_pretrain.yaml \
    --model_path /path/to/checkpoint.pt \
    --prompt_mode correct \
    --data_root_dir /path/to/data \
    --data_mix droid_video_prompt \
    --output_dir ./eval_outputs
"""

import argparse
import json
import os
import random
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from lda.training.trainer_utils.trainer_tools import normalize_dotlist_args
from lda.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
from lda.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from lda.dataloader.gr00t_lerobot.datasets import VideoPromptLeRobotSingleDataset
from lda.dataloader.gr00t_lerobot.embodiment_tags import EMBODIMENT_TAG_MAPPING
from lda.model.framework.base_framework import baseframework

warnings.simplefilter("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Action L1 computation
# ---------------------------------------------------------------------------

def angular_diff(a, b):
    """Smallest signed angle difference, result in [-pi, pi)."""
    diff = a - b
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff


def compute_action_l1(gt_actions, pred_actions, action_dim_info=None):
    """Compute per-dimension-group L1 losses.

    Args:
        gt_actions: np.ndarray [T, D]
        pred_actions: np.ndarray [T, D]
        action_dim_info: dict mapping group_name -> (start, end) slices.

    Returns:
        dict with per-group L1 and total L1/MSE.
    """
    T, D = gt_actions.shape
    sq_error = gt_actions - pred_actions

    if action_dim_info is None:
        if D == 7:
            # DroidFranka: eef_pos(3) + eef_rot(3) + gripper(1)
            action_dim_info = {
                "ee_position": (0, 3),
                "ee_rotation": (3, 6),
                "gripper": (6, 7),
            }
        elif D == 14:
            action_dim_info = {
                "left_ee_position": (0, 3),
                "left_ee_rotation": (3, 6),
                "left_gripper": (6, 7),
                "right_ee_position": (7, 10),
                "right_ee_rotation": (10, 13),
                "right_gripper": (13, 14),
            }
        elif D >= 24:
            action_dim_info = {
                "left_ee_position": (0, 3),
                "left_ee_rotation": (3, 6),
                "right_ee_position": (6, 9),
                "right_ee_rotation": (9, 12),
                "hand": (12, D),
            }
        else:
            action_dim_info = {
                "full_action": (0, D),
            }

    result = {}
    result["total_action_l1"] = float(np.mean(np.abs(sq_error)))
    result["total_action_mse"] = float(np.mean(sq_error ** 2))

    for name, (start, end) in action_dim_info.items():
        if "rotation" in name:
            diff = angular_diff(gt_actions[:, start:end], pred_actions[:, start:end])
            result[f"{name}_l1"] = float(np.mean(np.abs(diff)))
        else:
            result[f"{name}_l1"] = float(np.mean(np.abs(sq_error[:, start:end])))

    return result


# ---------------------------------------------------------------------------
# Dataset-type detection for unapply
# ---------------------------------------------------------------------------

WO_GRIPPER_DATASET = ['human', 'egovla']
ROBOCASA_DATASET = ["robocasa", "gr1"]


def unnormalize_action(action_chunk_tensor, dataset):
    """Un-normalize a predicted action chunk back to original scale.

    Handles multiple dataset types (DroidFranka, RoboCasa, bimanual, etc.)
    following the same pattern as eval_wo_postprocess.py.

    Args:
        action_chunk_tensor: torch.Tensor [T, D] normalized actions
        dataset: the single-embodiment dataset with .transforms.unapply()

    Returns:
        np.ndarray [T, D] unnormalized actions
    """
    tag = dataset._metadata.embodiment_tag.value

    if tag in ROBOCASA_DATASET:
        modality_keys = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
        unapplied = dataset.transforms.unapply({
            "action.left_arm": action_chunk_tensor[:, :7],
            "action.right_arm": action_chunk_tensor[:, 7:14],
            "action.left_hand": action_chunk_tensor[:, 14:20],
            "action.right_hand": action_chunk_tensor[:, 20:26],
            "action.waist": action_chunk_tensor[:, 26:29],
        })
        return np.concatenate([unapplied[f"action.{k}"] for k in modality_keys], axis=-1)

    elif tag in WO_GRIPPER_DATASET:
        modality_keys = ["left_eef_position", "left_eef_rotation",
                         "right_eef_position", "right_eef_rotation",
                         "left_mano_hand_param", "right_mano_hand_param"]
        unapplied = dataset.transforms.unapply({
            "action.left_eef_position": action_chunk_tensor[:, :3],
            "action.left_eef_rotation": action_chunk_tensor[:, 3:6],
            "action.right_eef_position": action_chunk_tensor[:, 6:9],
            "action.right_eef_rotation": action_chunk_tensor[:, 9:12],
            "action.left_mano_hand_param": action_chunk_tensor[:, 12:18],
            "action.right_mano_hand_param": action_chunk_tensor[:, 18:24],
        })
        return np.concatenate([unapplied[f"action.{k}"] for k in modality_keys], axis=-1)

    else:
        # DroidFranka (single-arm, action_dim=7) or bimanual Franka (action_dim=14)
        action_dim = action_chunk_tensor.shape[-1]
        if action_dim == 7:
            # DroidFranka: eef_pos(3) + eef_rot(3) + gripper(1)
            unapplied = dataset.transforms.unapply({
                "action.eef_position": action_chunk_tensor[:, :3],
                "action.eef_rotation": action_chunk_tensor[:, 3:6],
                "action.gripper": action_chunk_tensor[:, 6:7],
            })
            return np.concatenate([
                unapplied["action.eef_position"],
                unapplied["action.eef_rotation"],
                unapplied["action.gripper"],
            ], axis=-1)
        elif action_dim >= 14:
            # Bimanual Franka
            unapplied = dataset.transforms.unapply({
                "action.left_eef_position": action_chunk_tensor[:, :3],
                "action.left_eef_rotation": action_chunk_tensor[:, 3:6],
                "action.left_gripper": action_chunk_tensor[:, 6:7],
                "action.right_eef_position": action_chunk_tensor[:, 69:72],
                "action.right_eef_rotation": action_chunk_tensor[:, 72:75],
                "action.right_gripper": action_chunk_tensor[:, 75:76],
            })
            return np.concatenate([
                unapplied["action.left_eef_position"],
                unapplied["action.left_eef_rotation"],
                unapplied["action.left_gripper"],
                unapplied["action.right_eef_position"],
                unapplied["action.right_eef_rotation"],
                unapplied["action.right_gripper"],
            ], axis=-1)
        else:
            # Fallback: no unapply
            return action_chunk_tensor.numpy()


# ---------------------------------------------------------------------------
# Per-trajectory eval loop
# ---------------------------------------------------------------------------

def eval_single_trajectory(
    policy,
    dataset,
    traj_id,
    prompt_mode,
    action_horizon=16,
    max_steps=300,
):
    """Evaluate a single trajectory under a given prompt_mode.

    Returns dict of L1 losses.
    """
    return_state = False if policy.config.framework.action_model.state_dim is None else True
    tag = dataset._metadata.embodiment_tag.value
    is_robocasa = tag in ROBOCASA_DATASET
    wo_gripper = tag in WO_GRIPPER_DATASET

    gt_actions_all = []
    pred_actions_all = []

    eval_steps = min(max_steps, dataset.trajectory_lengths[traj_id])
    is_video_prompt_dataset = isinstance(dataset, VideoPromptLeRobotSingleDataset)

    for step_count in range(eval_steps):
        if step_count % action_horizon == 0:
            data_point = dataset.get_step_data_with_transform(
                traj_id, step_count, return_state=return_state
            )

            # Add support videos for video-prompt datasets
            if is_video_prompt_dataset and prompt_mode != "none":
                query_ep = traj_id
                # Get language from the data_point
                query_lang = str(data_point.get("lang", data_point.get("language", "")))
                support_eps, prompt_is_correct = dataset._sample_support_episodes(
                    query_ep=query_ep, query_lang=query_lang,
                )
                support_videos = [dataset._load_support_video(ep) for ep in support_eps]

                if prompt_mode == "shuffled":
                    support_videos = dataset._shuffle_video_frames(support_videos)
                elif prompt_mode == "final_frame":
                    support_videos = dataset._make_final_frame_prompt(support_videos)

                data_point["support_videos"] = support_videos
                data_point["support_episode_ids"] = support_eps
                data_point["query_episode_id"] = query_ep
                data_point["prompt_is_correct"] = prompt_is_correct
                data_point["prompt_mode"] = prompt_mode
            else:
                # None or not a video prompt dataset
                data_point["support_videos"] = None
                data_point["prompt_mode"] = prompt_mode

            batch = collate_fn([data_point])
            action_chunk = policy.predict_action(batch)
            action_chunk = torch.from_numpy(action_chunk["normalized_actions"][0])

            # Unnormalize
            concat_pred_action = unnormalize_action(action_chunk, dataset)

            for j in range(action_horizon):
                if step_count + j < eval_steps:
                    pred_actions_all.append(concat_pred_action[j])
                    # GT action: different slicing based on dataset type
                    if is_robocasa:
                        gt_actions_all.append(data_point["action"][j])
                    elif wo_gripper:
                        gt_actions_all.append(data_point["action"][j][:24])
                    else:
                        # DroidFranka or bimanual
                        if data_point["action"].shape[-1] >= 14:
                            gt_actions_all.append(data_point["action"][j + 1][:14])
                        else:
                            gt_actions_all.append(data_point["action"][j + 1][:7])

    gt_actions_all = np.array(gt_actions_all)
    pred_actions_all = np.array(pred_actions_all)

    # Trim to match dimensions
    min_dim = min(gt_actions_all.shape[-1], pred_actions_all.shape[-1])
    gt_actions_all = gt_actions_all[:, :min_dim]
    pred_actions_all = pred_actions_all[:, :min_dim]

    if len(gt_actions_all) == 0:
        return {"total_action_l1": float("nan")}

    return compute_action_l1(gt_actions_all, pred_actions_all)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Video prompt offline eval")
    parser.add_argument("--config_yaml", type=str, required=True,
                        help="Path to training YAML config")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint .pt file")
    parser.add_argument("--prompt_mode", type=str, default="correct",
                        choices=["none", "correct", "wrong", "shuffled", "final_frame"],
                        help="Video prompt mode for evaluation")
    parser.add_argument("--data_root_dir", type=str, required=True,
                        help="Root directory of LeRobot datasets")
    parser.add_argument("--data_mix", type=str, required=True,
                        help="Dataset mixture name (registered in mixtures.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for JSON results")
    parser.add_argument("--max_eval_trajs", type=int, default=20,
                        help="Max number of trajectories to evaluate")
    parser.add_argument("--start_traj", type=int, default=0,
                        help="Starting trajectory index")
    parser.add_argument("--action_horizon", type=int, default=16,
                        help="Action horizon for inference")
    parser.add_argument("--max_steps", type=int, default=300,
                        help="Max steps per trajectory")
    parser.add_argument("--num_support_demos", type=int, default=2,
                        help="K support demos per query (overrides DataConfig)")
    parser.add_argument("--num_support_frames", type=int, default=4,
                        help="T frames per support demo (overrides DataConfig)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args, clipargs = parser.parse_known_args()

    # Load base config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    if dotlist:
        cli_cfg = OmegaConf.from_dotlist(dotlist)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Override prompt_mode on DataConfig instances
    for robot_type, data_config in ROBOT_TYPE_CONFIG_MAP.items():
        if hasattr(data_config, "use_video_prompt") and data_config.use_video_prompt:
            data_config.prompt_mode = args.prompt_mode
            data_config.num_support_demos = args.num_support_demos
            data_config.num_support_frames = args.num_support_frames

    # Update data config for dataset creation
    cfg.datasets.vla_data.data_root_dir = args.data_root_dir
    cfg.datasets.vla_data.data_mix = args.data_mix

    # Create dataset
    model_cfg = cfg.framework.action_model
    model_id = cfg.framework.qwenvl.base_vlm
    dataset = get_vla_dataset(cfg.datasets.vla_data, model_cfg=model_cfg, model_id=model_id)

    # Load model
    print(f"Loading model from {args.model_path} ...")
    policy = baseframework.from_pretrained(pretrained_checkpoint=args.model_path)
    policy.eval()
    policy.to("cuda")
    print("Model loaded successfully.")

    # Select trajectories to evaluate
    # Use the first sub-dataset that has video prompt support
    eval_dataset = None
    for ds in dataset.datasets:
        if isinstance(ds, VideoPromptLeRobotSingleDataset):
            eval_dataset = ds
            break
    if eval_dataset is None:
        # Fall back to any sub-dataset
        eval_dataset = dataset.datasets[0]
        print(f"WARNING: No VideoPromptLeRobotSingleDataset found, using {type(eval_dataset).__name__}")

    max_traj_id = len(eval_dataset.trajectory_ids)
    n_trajs = min(args.max_eval_trajs, max_traj_id - args.start_traj)
    traj_ids = list(range(args.start_traj, args.start_traj + n_trajs))
    print(f"Evaluating {len(traj_ids)} trajectories with prompt_mode={args.prompt_mode}")

    # Run evaluation
    all_results = []
    for i, traj_id in enumerate(traj_ids):
        print(f"[{i+1}/{len(traj_ids)}] Trajectory {traj_id} ...")
        try:
            l1_dict = eval_single_trajectory(
                policy=policy,
                dataset=eval_dataset,
                traj_id=traj_id,
                prompt_mode=args.prompt_mode,
                action_horizon=args.action_horizon,
                max_steps=args.max_steps,
            )
            result = {
                "traj_id": traj_id,
                "prompt_mode": args.prompt_mode,
                "num_support_demos": args.num_support_demos,
                "num_support_frames": args.num_support_frames,
                **l1_dict,
            }
            all_results.append(result)
            print(f"  -> total_action_l1={l1_dict.get('total_action_l1', 'N/A'):.6f}")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            all_results.append({
                "traj_id": traj_id,
                "prompt_mode": args.prompt_mode,
                "num_support_demos": args.num_support_demos,
                "num_support_frames": args.num_support_frames,
                "error": str(e),
            })

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"eval_{args.prompt_mode}_k{args.num_support_demos}_t{args.num_support_frames}.json"
    )
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_file}")

    # Print summary
    valid = [r for r in all_results if "error" not in r]
    if valid:
        print("\n" + "=" * 60)
        print(f"Summary for prompt_mode={args.prompt_mode}, K={args.num_support_demos}")
        print("=" * 60)
        for key in valid[0]:
            if key in ("traj_id", "prompt_mode", "num_support_demos",
                       "num_support_frames", "error"):
                continue
            vals = [r[key] for r in valid if key in r]
            if vals:
                print(f"  {key}: {np.mean(vals):.6f} +/- {np.std(vals):.6f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
