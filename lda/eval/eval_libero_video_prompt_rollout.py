#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Closed-loop LIBERO rollout evaluation for the LDA video-prompt ablation.

This script loads a video-prompt-trained LDA checkpoint and runs *closed-loop*
rollouts in the LIBERO MuJoCo simulation (libero_spatial / object / goal / 10),
measuring task success rate under different video-prompt modes:

  - "none"     : no support video fed to the model (the baseline)
  - "correct"  : K support demos sampled from OTHER episodes with the SAME
                 task instruction, encoded into prompt tokens
  - "wrong"    : support demos from a DIFFERENT task instruction
                 (control: a useful prompt should beat this)
  - "shuffled" : correct demos but with frame order shuffled
  - "final_frame": correct demos collapsed to the last frame repeated T times

By comparing success rates across modes (especially `none` vs `correct`, and
`correct` vs `wrong`/`shuffled`), you can tell whether the video prompt is
actually helping the policy.

The simulation side reuses the conventions of vla-evaluation-harness
(`vla_eval/benchmarks/libero/`) and gr00t's `libero_env.py`:

  - state fed to the model = [eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos[1]]
            (matches the LiberoFrankaDataConfig, which drops the `pad` dim
             = gripper_qpos[0] and keeps gripper_qpos[1])
  - state/action are q99-normalized using the checkpoint's `norm_stats['franka']`.
  - action gripper ∈ [0,1] (dataset convention, 0=close, 1=open) is converted to
    robosuite's [-1,+1] and binarized (sign) before env.step().
  - agentview + wrist images are expand2square-padded with ImageNet mean,
    resized to 224×224, and stacked as [view0_t0, view0_t1, view1_t0, view1_t1]
    (obs_horizon=2, num_views=2). Each sim image is flip180'd (``[::-1,::-1]``)
    before preprocessing: the LDA libero lerobot dataset stores flip180 of the
    raw MuJoCo offscreen render, and the model was trained on those — so the
    eval must feed flip180 too (feeding raw makes the scene upside-down vs
    training → 0% success). For closed-loop inference we have no history, so
    t-1 is filled with the current frame (same as duplicating the current
    observation).

Usage (run inside the LDA conda env, which has libero + mujoco + the LDA deps):

  MUJOCO_GL=egl python lda/eval/eval_libero_video_prompt_rollout.py \\
      --model_path runs/lda_libero_video_prompt_k2_t4/checkpoints/steps_20000_pytorch_model.pt \\
      --data_root_dir /cpfs01/Embodied/datasets \\
      --suites libero_spatial \\
      --prompt_modes none correct \\
      --episodes_per_task 20 \\
      --output_dir eval_outputs/libero_video_prompt_rollout
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
import sys
import warnings
from collections import deque
from pathlib import Path

# ---- headless rendering (must be set before libero/mujoco import) ----------
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "device")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
from PIL import Image

# LDA lives one dir up from this file's package -> repo root on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lda.model.framework.base_framework import baseframework
from lda.dataloader.lerobot_datasets import make_LeRobotSingleDataset

warnings.simplefilter("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# LIBERO env / image / state conventions  (mirrors gr00t libero_env.py +
# vla-evaluation-harness benchmarks/libero)
# ---------------------------------------------------------------------------

# MuJoCo offscreen renders are upside-down vs the dataset/training convention;
# preprocess_libero_image / _obs_agentview_uint8 apply flip180 to compensate.
LIBERO_ENV_RESOLUTION = 256
# Dummy open-gripper action used to settle the sim at episode start.
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

MAX_STEP_MAPPING = {
    "libero_spatial": 220,
    "libero_goal": 300,
    "libero_object": 280,
    "libero_10": 520,
    "libero_90": 400,
}

# ImageNet mean (used for expand2square padding, matches LDA dataset pipeline).
IMG_MEAN = [0.485, 0.456, 0.406]


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """robosuite-style quat [x,y,z,w] -> axis-angle. Matches gr00t libero_env."""
    q = np.asarray(quat, dtype=np.float64).copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * math.acos(q[3]) / den).astype(np.float32)


def expand2square(pil_img: Image.Image, background_color) -> Image.Image:
    w, h = pil_img.size
    if w == h:
        return pil_img
    if w > h:
        result = Image.new(pil_img.mode, (w, w), background_color)
        result.paste(pil_img, (0, (w - h) // 2))
        return result
    result = Image.new(pil_img.mode, (h, h), background_color)
    result.paste(pil_img, ((h - w) // 2, 0))
    return result


def preprocess_libero_image(img: np.ndarray) -> Image.Image:
    """uint8 -> 224x224 PIL, matching the LDA dataset pipeline.

    The MuJoCo offscreen render (``obs["agentview_image"]`` /
    ``obs["robot0_eye_in_hand_image"]``) is upside-down relative to the frames
    the model was trained on: the LDA libero lerobot dataset stores frames that
    are flip180 (``[::-1, ::-1]``) of the raw sim render. Verified empirically
    (2026-06-26, libero_spatial task0 init_state[0], mujoco 3.2.3):
        corr(stored_frame0, sim_raw)            = -0.24
        corr(stored_frame0, sim_raw[::-1,::-1]) = +0.91
    So we MUST apply the same flip180 before feeding the image to the model.
    Feeding raw (unflipped) frames makes the scene upside-down vs training and
    drives closed-loop success to 0%.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    img = np.ascontiguousarray(img[::-1, ::-1])  # match dataset/training orientation
    pil = Image.fromarray(img)
    bg = tuple(int(x * 255) for x in IMG_MEAN)
    pil = expand2square(pil, bg).resize((224, 224))
    return pil


# ---------------------------------------------------------------------------
# Model wrapper: builds the predict_action example from a sim observation
# ---------------------------------------------------------------------------

class LDARolloutPolicy:
    """Loads an LDA checkpoint and exposes a step-level action predictor."""

    def __init__(
        self,
        model_path: str,
        use_bf16: bool = True,
        num_inference_steps: int | None = None,
    ):
        print(f"[policy] Loading LDA checkpoint: {model_path}")
        self.model = baseframework.from_pretrained(pretrained_checkpoint=model_path)
        if use_bf16:
            self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to("cuda").eval()

        # Override the flow-matching denoising step count at inference time.
        # The model reads `num_inference_timesteps` from its config (yaml default
        # 4) inside predict_action; the official LDA Robocasa eval also uses 4
        # (the num_ddim_steps=10 it passes is a legacy kwarg the framework ignores).
        # We patch the attribute on action_model so the eval can sweep it without
        # touching model code.
        self.num_inference_steps = num_inference_steps
        if num_inference_steps is not None:
            self.model.action_model.num_inference_timesteps = int(num_inference_steps)
        cur_steps = self.model.action_model.num_inference_timesteps
        print(f"[policy] num_inference_timesteps = {cur_steps}")

        am = self.model.config.framework.action_model
        self.obs_horizon = am.obs_horizon
        self.num_views = am.num_views
        self.state_dim = am.state_dim
        self.action_dim = am.action_dim
        self.use_video_prompt = getattr(am, "use_video_prompt", False)

        stats = self.model.norm_stats
        assert "franka" in stats, f"Expected 'franka' norm_stats, got {list(stats.keys())}"
        self.norm_stats = stats["franka"]
        s = self.norm_stats["state"]
        self.state_q01 = np.asarray(s["q01"], dtype=np.float32)
        self.state_q99 = np.asarray(s["q99"], dtype=np.float32)
        self.state_mask = np.asarray(s.get("mask", [True] * len(self.state_q01)), dtype=bool)
        a = self.norm_stats["action"]
        self.act_q01 = np.asarray(a["q01"], dtype=np.float32)
        self.act_q99 = np.asarray(a["q99"], dtype=np.float32)
        self.act_mask = np.asarray(a.get("mask", [True] * len(self.act_q01)), dtype=bool)
        self.real_action_dim = len(self.act_q01)  # 7 for franka

        # FRANKA embodiment id (see lda/dataloader/gr00t_lerobot/embodiment_tags.py)
        self.embodiment_id = 4

        # Per-view image history, mirroring the official Robocasa MultiStepWrapper:
        # a deque seeded with the first frame at episode reset, then appended with
        # each new observation. build_images reads the last `obs_horizon` REAL
        # frames instead of duplicating the current frame (which is a train/eval
        # distribution mismatch — training sees true history frames).
        self.image_history: list[deque] = [
            deque(maxlen=self.obs_horizon) for _ in range(self.num_views)
        ]

        print(
            f"[policy] obs_horizon={self.obs_horizon} num_views={self.num_views} "
            f"state_dim={self.state_dim} action_dim={self.action_dim} "
            f"use_video_prompt={self.use_video_prompt}"
        )

    def reset_history(self, obs: dict) -> None:
        """Seed the per-view history with the first frame (obs_horizon copies).

        Matches MultiStepWrapper.reset: `deque([obs]*(max_steps_needed+1))`, so at
        episode start the history is full of the initial frame and the policy sees
        a valid (if repeated) context, then real frames accumulate as the loop runs.
        """
        first = [preprocess_libero_image(obs["agentview_image"]),
                 preprocess_libero_image(obs["robot0_eye_in_hand_image"])]
        for v in range(self.num_views):
            self.image_history[v].clear()
            for _ in range(self.obs_horizon):
                self.image_history[v].append(first[v])

    def _normalize_state(self, raw_state_7: np.ndarray) -> np.ndarray:
        """q99-normalize the 7-dim franka state: 2*(x-q01)/(q99-q01)-1 (masked dims)."""
        x = raw_state_7.astype(np.float32)
        out = x.copy()
        mask = self.state_mask
        denom = self.state_q99[mask] - self.state_q01[mask]
        denom = np.where(np.abs(denom) < 1e-6, 1.0, denom)
        out[mask] = 2.0 * (x[mask] - self.state_q01[mask]) / denom - 1.0
        # unmasked dims stay at raw value (matches StateActionTransform q99 path)
        return out

    def _unnormalize_action(self, norm_action_7: np.ndarray) -> np.ndarray:
        """q99-unnormalize the 7-dim action. Gripper (mask=False) passes through."""
        x = np.clip(norm_action_7.astype(np.float32), -1.0, 1.0)
        out = x.copy()
        mask = self.act_mask
        out[mask] = (x[mask] + 1.0) / 2.0 * (self.act_q99[mask] - self.act_q01[mask]) + self.act_q01[mask]
        return out

    def build_state(self, obs: dict) -> np.ndarray:
        """[eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos[1]] -> 7-dim."""
        xyz = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
        raw = np.concatenate([xyz, rpy, [gripper[1]]])  # drop pad=gripper[0]
        return self._normalize_state(raw)

    def build_images(self, obs: dict) -> list[Image.Image]:
        """Return [view0_t0, view0_t1, view1_t0, view1_t1] of 224x224 PIL.

        Uses the real observation history (seeded at episode reset, appended each
        call), matching the official Robocasa MultiStepWrapper. Layout is
        view-major / time-minor (oldest -> newest), consistent with the model's
        `rearrange("b (v t) c h w -> b v t c h w")`.
        """
        new_frames = [preprocess_libero_image(obs["agentview_image"]),
                      preprocess_libero_image(obs["robot0_eye_in_hand_image"])]
        for v in range(self.num_views):
            self.image_history[v].append(new_frames[v])

        images = []
        for v in range(self.num_views):
            # deque holds the last `obs_horizon` frames, oldest-first.
            for t in range(self.obs_horizon):
                images.append(self.image_history[v][t])
        return images

    def predict(
        self,
        obs: dict,
        task_lang: str,
        support_videos: list | None,
    ) -> np.ndarray:
        """Predict a 16-step action chunk in *normalized* (q99) action space (7-dim).

        The model is trained on q99-normalized actions and this returns the raw
        q99-normalized output. The caller (run_episode) must denormalize via
        _unnormalize_action before env.step, because the LIBERO dataset actions
        are already in OSC_POSE delta controller space and LDA's q99 normalization
        on top is non-identity (feeding normalized actions raw overscales the
        small-range rotation/xyz deltas). Only the gripper dim is masked
        (passthrough) and binarized in env.step.
        """
        example = {
            "image": self.build_images(obs),
            "lang": task_lang,
            "embodiment_id": self.embodiment_id,
            "state": self.build_state(obs)[None, :],  # (1, 7)
        }
        if support_videos is not None:
            example["support_videos"] = support_videos

        with torch.inference_mode():
            out = self.model.predict_action(examples=[example])
        norm_actions = out["normalized_actions"][0]  # (chunk_len, action_dim)
        # take only the real action dims (model may pad to max action_dim)
        norm_actions = norm_actions[:, : self.real_action_dim].astype(np.float32)
        return norm_actions  # (chunk_len, 7) normalized


# ---------------------------------------------------------------------------
# Support-video sampler: one VideoPromptLeRobotSingleDataset per suite
# ---------------------------------------------------------------------------

class SupportVideoSampler:
    """Samples K support demos (T frames each) for a given task language.

    Uses the same LeRobot dataset the model was trained on, via the LDA
    VideoPromptLeRobotSingleDataset helpers (_sample_support_episodes /
    _load_support_video), so the support frames are decoded and resized
    exactly as during training.
    """

    def __init__(
        self,
        data_root_dir: str,
        suite: str,
        num_support_demos: int,
        num_support_frames: int,
        seed: int = 0,
    ):
        suite_to_ds = {
            "libero_spatial": "libero_mujoco3.3.2/libero_spatial_no_noops_1.0.0_lerobot",
            "libero_object": "libero_mujoco3.3.2/libero_object_no_noops_1.0.0_lerobot",
            "libero_goal": "libero_mujoco3.3.2/libero_goal_no_noops_1.0.0_lerobot",
            "libero_10": "libero_mujoco3.3.2/libero_10_no_noops_1.0.0_lerobot",
        }
        assert suite in suite_to_ds, f"Unsupported suite: {suite}"
        print(f"[support] building dataset for {suite} ...")
        self.dataset = make_LeRobotSingleDataset(
            Path(data_root_dir),
            suite_to_ds[suite],
            "libero_franka_video_prompt",
            data_cfg={
                "use_video_prompt": True,
                "num_support_demos": num_support_demos,
                "num_support_frames": num_support_frames,
                "video_backend": "torchvision_av",
                "use_delta_action": False,
            },
        )
        self.dataset.num_support_demos = num_support_demos
        self.dataset.num_support_frames = num_support_frames
        self.K = num_support_demos
        self._rng = random.Random(seed)

    def _sample_correct(self, query_lang: str):
        eps, correct = self.dataset._sample_support_episodes(
            query_ep=-1, query_lang=query_lang
        )
        return eps, correct

    def _sample_wrong(self, query_lang: str):
        other_langs = [l for l in self.dataset.language_to_episodes if l != query_lang]
        if not other_langs:
            return self._sample_correct(query_lang)
        wrong_lang = self._rng.choice(other_langs)
        candidates = [ep for ep in self.dataset.language_to_episodes[wrong_lang]]
        if len(candidates) >= self.K:
            eps = self._rng.sample(candidates, self.K)
        else:
            eps = self._rng.choices(candidates, k=self.K)
        return eps, False

    def sample(self, query_lang: str, prompt_mode: str):
        """Return support_videos in List[K][T][V] PIL layout, or None.

        - none / final_frame(collapsed) handled here.
        - shuffled: correct episodes, frames shuffled per demo.
        """
        if prompt_mode == "none":
            return None

        if prompt_mode == "wrong":
            eps, _ = self._sample_wrong(query_lang)
        else:  # correct / shuffled / final_frame all use same-instruction eps
            eps, _ = self._sample_correct(query_lang)

        support_videos = [self.dataset._load_support_video(ep) for ep in eps]

        if prompt_mode == "shuffled":
            support_videos = self.dataset._shuffle_video_frames(support_videos)
        elif prompt_mode == "final_frame":
            support_videos = self.dataset._make_final_frame_prompt(support_videos)

        return support_videos


# ---------------------------------------------------------------------------
# LIBERO environment wrapper (mirrors vla-eval-harness LIBEROBenchmark)
# ---------------------------------------------------------------------------

def _patch_torch_load_for_libero():
    """LIBERO init-state files use torch.save with numpy globals; PyTorch>=2.6
    defaults weights_only=True which blocks them. Patch once, lazily."""
    original = torch.load

    @functools.wraps(original)
    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = patched


class LiberoEnv:
    def __init__(self, suite: str, seed: int = 7, num_steps_wait: int = 10):
        _patch_torch_load_for_libero()
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv

        self._OffScreenRenderEnv = OffScreenRenderEnv
        self.suite = suite
        self.seed = seed
        self.num_steps_wait = num_steps_wait
        bd = benchmark.get_benchmark_dict()
        self.task_suite = bd[suite]()
        self._env = None
        self._current_task_id = None

    def num_tasks(self) -> int:
        return self.task_suite.n_tasks

    def task_language(self, task_id: int) -> str:
        return self.task_suite.get_task(task_id).language

    def num_init_states(self, task_id: int) -> int:
        return len(self.task_suite.get_task_init_states(task_id))

    def reset(self, task_id: int, episode_idx: int):
        from libero.libero import get_libero_path

        if self._env is None or self._current_task_id != task_id:
            if self._env is not None:
                self._env.close()
            task = self.task_suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            self._env = self._OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=LIBERO_ENV_RESOLUTION,
                camera_widths=LIBERO_ENV_RESOLUTION,
            )
            self._env.seed(self.seed)
            self._current_task_id = task_id

        self._env.reset()
        init_states = self.task_suite.get_task_init_states(task_id)
        obs = self._env.set_init_state(init_states[episode_idx])
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self._env.step(LIBERO_DUMMY_ACTION)
        return obs

    def step(self, action_7: np.ndarray):
        a = np.asarray(action_7, dtype=np.float64).tolist()
        assert len(a) == 7
        # Gripper sign convention:
        #   dataset  : 0 = close, 1 = open
        #   robosuite: -1 = open, +1 = close   (gr00t libero_env inverts the sign)
        # So dataset 0 (close) -> +1 (close), dataset 1 (open) -> -1 (open).
        gripper = 1.0 if a[-1] < 0.5 else -1.0
        processed = a[:-1] + [gripper]
        obs, reward, done, info = self._env.step(processed)
        return obs, reward, done, info

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_episode(
    policy: LDARolloutPolicy,
    env: LiberoEnv,
    task_id: int,
    episode_idx: int,
    task_lang: str,
    prompt_mode: str,
    support_sampler: SupportVideoSampler | None,
    chunk_size: int,
    max_steps: int,
    record_frames: list | None = None,
) -> dict:
    obs = env.reset(task_id, episode_idx)
    policy.reset_history(obs)
    if record_frames is not None:
        record_frames.append(_obs_agentview_uint8(obs))

    # Sample support videos ONCE per episode (the prompt guides the whole traj).
    support_videos = None
    if support_sampler is not None and prompt_mode != "none":
        support_videos = support_sampler.sample(task_lang, prompt_mode)

    success = False
    success_step = -1
    step = 0
    action_chunk = None
    chunk_idx = 0
    # Receding-horizon execution (mirrors official Robocasa eval): the model
    # predicts a full action_horizon chunk (16 steps), but we only execute the
    # first `chunk_size` steps before re-planning with the latest observation
    # (official n_action_steps=12). This closes the loop more frequently than
    # executing the whole chunk, reducing open-loop error accumulation.
    # chunk_size <= 0 means execute the whole chunk (legacy open-loop behavior).
    exec_horizon = chunk_size if chunk_size and chunk_size > 0 else len(action_chunk) if action_chunk is not None else 1
    while step < max_steps:
        if action_chunk is None or chunk_idx >= exec_horizon or chunk_idx >= len(action_chunk):
            action_chunk = policy.predict(obs, task_lang, support_videos)
            chunk_idx = 0
            exec_horizon = chunk_size if chunk_size and chunk_size > 0 else len(action_chunk)
        a = action_chunk[chunk_idx]
        chunk_idx += 1
        # Denormalize the q99-normalized action back into the raw LIBERO OSC_POSE
        # delta controller space the env expects. Training applies q99 normalization
        # (state_action.Normalizer.forward) on top of the already-controller-space
        # dataset actions, so the model output is q99-normalized; feeding it raw
        # would overscale the small-range rotation deltas ~6-9x and xyz ~1.1x,
        # driving closed-loop success to 0%. The masked gripper dim passes through
        # unchanged and is still binarized in env.step.
        a = policy._unnormalize_action(a)
        obs, reward, done, info = env.step(a)
        step += 1
        if record_frames is not None:
            record_frames.append(_obs_agentview_uint8(obs))
        if env._env.check_success():
            success = True
            success_step = step
            break
        if done:
            break

    return {
        "task_id": task_id,
        "episode_idx": episode_idx,
        "task": task_lang,
        "prompt_mode": prompt_mode,
        "success": success,
        "success_step": success_step,
        "steps": step,
    }


def _obs_agentview_uint8(obs: dict) -> np.ndarray:
    """Return the agentview image as an upright uint8 HxWx3 array (for video).

    Mirrors preprocess_libero_image: the raw MuJoCo offscreen render is
    upside-down, so we flip180 to store an upright frame that matches what the
    model (and the dataset) sees. Without this the saved rollout videos look
    upside-down.
    """
    img = obs["agentview_image"]
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return np.ascontiguousarray(img[::-1, ::-1])


def save_frames_as_video(frames: list, out_path: str, fps: int = 20):
    """Write a list of HxWx3 uint8 frames to an mp4 via imageio/ffmpeg."""
    import imageio.v2 as imageio

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                quality=8, macro_block_size=1)
    for f in frames:
        writer.append_data(f)
    writer.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="LDA video-prompt closed-loop LIBERO eval")
    p.add_argument("--model_path", type=str, required=True, help="LDA checkpoint .pt")
    p.add_argument("--data_root_dir", type=str, default="/cpfs01/Embodied/datasets")
    p.add_argument(
        "--suites", type=str, nargs="+",
        default=["libero_spatial"],
        help="libero_spatial / libero_object / libero_goal / libero_10",
    )
    p.add_argument(
        "--prompt_modes", type=str, nargs="+",
        default=["none", "correct"],
        choices=["none", "correct", "wrong", "shuffled", "final_frame"],
        help="Video-prompt modes to compare. none = no prompt (baseline).",
    )
    p.add_argument("--episodes_per_task", type=int, default=20)
    p.add_argument(
        "--chunk_size", type=int, default=12,
        help="Receding-horizon: actions executed per inference call before "
             "re-planning. The model predicts a full action_horizon chunk (16); "
             "only the first chunk_size steps are executed (official Robocasa eval "
             "uses n_action_steps=12). Set <= 0 to execute the whole chunk "
             "(legacy open-loop behavior).",
    )
    p.add_argument(
        "--num_inference_steps", type=int, default=4,
        help="Flow-matching denoising steps at inference. Official LDA eval uses "
             "4 (model yaml default). Override to sweep, e.g. 10.",
    )
    p.add_argument("--num_support_demos", type=int, default=2, help="K support demos")
    p.add_argument("--num_support_frames", type=int, default=4, help="T frames per support demo")
    p.add_argument("--max_steps", type=int, default=None, help="override suite max steps")
    p.add_argument("--seed", type=int, default=7, help="env seed")
    p.add_argument("--use_bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", dest="use_bf16", action="store_false")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--record_videos", action="store_true",
        help="Save an mp4 of the agentview for selected rollouts.",
    )
    p.add_argument(
        "--videos_per_task", type=int, default=1,
        help="How many episodes per (suite, mode, task) to record (the first N).",
    )
    p.add_argument("--video_fps", type=int, default=20)
    p.add_argument(
        "--task_ids", type=int, nargs="*", default=None,
        help="Only run these task_ids (for multi-GPU sharding). "
             "Default None = all tasks in the suite.",
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    policy = LDARolloutPolicy(
        args.model_path,
        use_bf16=args.use_bf16,
        num_inference_steps=args.num_inference_steps,
    )

    all_results = []
    summary = {}  # {(suite, mode): [success_bool, ...]}

    for suite in args.suites:
        env = LiberoEnv(suite=suite, seed=args.seed)
        max_steps = args.max_steps or MAX_STEP_MAPPING.get(suite, 300)
        n_tasks = env.num_tasks()
        # task_ids sharding (multi-GPU). Default = all tasks.
        task_ids = list(args.task_ids) if args.task_ids is not None else list(range(n_tasks))
        print(f"\n{'='*70}\nSuite {suite}: {n_tasks} tasks (running {len(task_ids)}: {task_ids}), max_steps={max_steps}\n{'='*70}")

        # Build one support sampler per suite (needed only if any mode != none)
        support_sampler = None
        if any(m != "none" for m in args.prompt_modes):
            support_sampler = SupportVideoSampler(
                data_root_dir=args.data_root_dir,
                suite=suite,
                num_support_demos=args.num_support_demos,
                num_support_frames=args.num_support_frames,
                seed=args.seed,
            )

        for mode in args.prompt_modes:
            if mode != "none" and not policy.use_video_prompt:
                print(f"[skip] model has use_video_prompt=False, cannot run mode={mode}")
                continue
            print(f"\n--- {suite} | prompt_mode={mode} ---")
            succ = 0
            total = 0
            for task_id in task_ids:
                task_lang = env.task_language(task_id)
                n_eps = min(args.episodes_per_task, env.num_init_states(task_id))
                for ep_idx in range(n_eps):
                    record = (
                        args.record_videos and ep_idx < args.videos_per_task
                    )
                    frames = [] if record else None
                    try:
                        res = run_episode(
                            policy=policy,
                            env=env,
                            task_id=task_id,
                            episode_idx=ep_idx,
                            task_lang=task_lang,
                            prompt_mode=mode,
                            support_sampler=support_sampler,
                            chunk_size=args.chunk_size,
                            max_steps=max_steps,
                            record_frames=frames,
                        )
                    except Exception as e:
                        print(f"  [error] task{task_id} ep{ep_idx}: {e}")
                        res = {
                            "task_id": task_id, "episode_idx": ep_idx, "task": task_lang,
                            "prompt_mode": mode, "success": False, "steps": 0,
                            "error": str(e),
                        }
                    all_results.append({**res, "suite": suite})
                    total += 1
                    if res["success"]:
                        succ += 1
                    print(
                        f"  [{total}/{len(task_ids)*n_eps}] task{task_id} ep{ep_idx} "
                        f"{'OK' if res['success'] else 'x'} (steps={res['steps']})"
                    )
                    if frames is not None and len(frames) > 0:
                        vid_dir = os.path.join(args.output_dir, "videos", suite, mode)
                        os.makedirs(vid_dir, exist_ok=True)
                        safe_lang = "".join(
                            c if c.isalnum() else "_" for c in task_lang
                        )[:60]
                        vid_path = os.path.join(
                            vid_dir,
                            f"task{task_id:02d}_ep{ep_idx:02d}_{safe_lang}"
                            f"_{'OK' if res['success'] else 'x'}.mp4",
                        )
                        try:
                            save_frames_as_video(frames, vid_path, fps=args.video_fps)
                            res["video_path"] = vid_path
                            print(f"      video -> {vid_path}")
                        except Exception as ve:
                            print(f"      [video error] {ve}")
            rate = succ / total if total else 0.0
            summary[(suite, mode)] = {"success": succ, "total": total, "rate": rate}
            print(f"=> {suite} | {mode}: {succ}/{total} = {rate*100:.1f}%")
        env.close()

    # ---- save ----
    out_file = os.path.join(args.output_dir, "rollout_results.json")
    with open(out_file, "w") as f:
        json.dump({"args": vars(args), "summary": {f"{s}|{m}": v for (s, m), v in summary.items()},
                   "episodes": all_results}, f, indent=2)
    print(f"\nSaved per-episode results -> {out_file}")

    # ---- summary table ----
    print("\n" + "=" * 70)
    print(f"{'suite':<16}{'mode':<14}{'success':>10}{'total':>8}{'rate':>10}")
    print("-" * 70)
    for (s, m), v in summary.items():
        print(f"{s:<16}{m:<14}{v['success']:>10}{v['total']:>8}{v['rate']*100:>9.1f}%")
    print("=" * 70)

    # ---- video-prompt effect ----
    print("\nVideo-prompt effect (success-rate gain from `correct` over `none`):")
    for suite in args.suites:
        none = summary.get((suite, "none"), {}).get("rate")
        corr = summary.get((suite, "correct"), {}).get("rate")
        if none is not None and corr is not None:
            delta = (corr - none) * 100
            arrow = "↑ HELPS" if delta > 0.5 else ("↓ hurts" if delta < -0.5 else "≈ no effect")
            print(f"  {suite:<16} none={none*100:5.1f}%  correct={corr*100:5.1f}%  Δ={delta:+.1f}pp  {arrow}")
    print("=" * 70)


if __name__ == "__main__":
    main()
