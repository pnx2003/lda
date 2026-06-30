#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Unset proxy to avoid wandb connection issues
# ============================================================
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

# ============================================================
# Debug mode: set to true for easier debugging
# ============================================================
DEBUG_MODE=${DEBUG_MODE:-false}

if [ "$DEBUG_MODE" = "true" ]; then
    export CUDA_LAUNCH_BLOCKING=1
    export TORCH_USE_CUDA_DSA=1
    export NCCL_ASYNC_ERROR_HANDLING=1
    echo "=== DEBUG MODE ENABLED ==="
    echo "CUDA_LAUNCH_BLOCKING=1"
    echo "TORCH_USE_CUDA_DSA=1"
    echo "NCCL_ASYNC_ERROR_HANDLING=1"
fi

# ============================================================
# LDA-1B Video-Prompt Training Script
# Version: DINO / world encoder prompt tokens
#
# Core idea:
#   support videos -> DINO/world encoder -> prompt tokens
#   concat(prompt tokens, vl_embs)
#   loss only on query action / query future latent
# ============================================================

# -------------------------
# Distributed config
# -------------------------
NNODES=${SENSECORE_PYTORCH_NNODES:-1}
NPROC_PER_NODE=${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}
NODE_RANK=${SENSECORE_PYTORCH_NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE - 1)))

echo "=== Distributed Config ==="
echo "NNODES: ${NNODES}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "NODE_RANK: ${NODE_RANK}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# -------------------------
# Optional: W&B
# -------------------------
export WANDB_MODE=${WANDB_MODE:-online}

wandb_project=lda-video-prompt
wandb_entity="2100017816"

# -------------------------
# Model paths
# -------------------------
Framework_name=QwenMMDiT

# Replace with your local Qwen3-VL checkpoint path
base_vlm=/cpfs01/Embodied/checkpoints/LDA-1B/Qwen3-VL-4B-Instruct

# Replace with your local DINOv3 checkpoint parent path
vision_encoder_path=/cpfs01/pnx/models/

# Load LDA pretrained checkpoint.
# Set to null if training from scratch.
pretrained_checkpoint=/cpfs01/Embodied/checkpoints/LDA-1B/LDA-pretrain/checkpoints/LDA-pretrain.pt 

# -------------------------
# Dataset config
# -------------------------
# Root directory of LeRobot-format datasets
# data_root_dir=/cpfs01/fc.liu/world-model/LDA-1B/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
data_root_dir=/cpfs01/Embodied/datasets
# This must be registered in:
#   lda/dataloader/gr00t_lerobot/mixtures.py
#
# Example:
#   "my_video_prompt_train": [
#       ("your_dataset_name", 1.0, "intern_franka_video_prompt"),
#   ]
data_mix=droid_video_prompt

# Your video-prompt robot type must be registered in:
#   lda/dataloader/gr00t_lerobot/data_config.py
#
# Example:
#   "intern_franka_video_prompt": FrankaVideoPromptDataConfig()
#
# The script does not pass robot_type directly; it is read from mixtures.py.

# -------------------------
# Video prompt settings
# -------------------------
use_video_prompt=true

# K-shot support demos per query sample
num_support_demos=2

# Frames sampled from each support demo
num_support_frames=4

# Max number of prompt tokens after DINO/world encoder.
# Keep this small at first to avoid OOM.
max_prompt_tokens=512

# -------------------------
# Debug mode overrides
# -------------------------
if [ "$DEBUG_MODE" = "true" ]; then
    NPROC_PER_NODE=1
    num_support_demos=1
    num_support_frames=2
    max_prompt_tokens=128
    echo "=== Debug Parameters ==="
    echo "NPROC_PER_NODE=1"
    echo "num_support_demos=1"
    echo "num_support_frames=2"
    echo "max_prompt_tokens=128"
fi

# -------------------------
# Memory-saving overrides for video prompt training
# -------------------------
# Video prompt adds significant memory:
#   - support_imgs pixels: [B, K, T, V, 3, 224, 224]
#   - dinov3 forward on K*T frames per sample
#   - repeated_diffusion_steps multiplies effective batch
# Use only_policy=true to train only the policy task (reduces task count 4→1,
# allowing smaller batch_size). repeated_diffusion_steps=1 for minimal memory.
only_policy=true
repeated_diffusion_steps=1
per_device_batch_size=4

# Debug mode: reduce for single-GPU
if [ "$DEBUG_MODE" = "true" ]; then
    per_device_batch_size=1
    repeated_diffusion_steps=1
    echo "per_device_batch_size=1"
    echo "repeated_diffusion_steps=1"
fi

# Probability of sampling wrong-task prompt during training.
# Start with 0.0. Later use 0.1 or 0.2 for robustness / ablation.
wrong_prompt_prob=0.0

# -------------------------
# LDA action model settings
# -------------------------
DIT_TYPE="DiT-L"

obs_horizon=2

# Change these according to your robot.
# For RoboCasa GR1 official script: state_dim=58, action_dim=138.
# For demo_data (Franka): state_dim=12 (eef_position=3 + eef_rotation=3 + gripper_width=6), action_dim=12.
# For droid (Franka single-arm): state_dim=8 (eef_position=3 + eef_rotation_quat=4 + gripper=1), action_dim=7 (eef_position=3 + eef_rotation_euler=3 + gripper=1).
state_dim=8
action_dim=7

max_num_embodiments=32
use_delta_action=false

# null, sinusoidal, rope
positional_embeddings=null

num_layers=16
# repeated_diffusion_steps is set in the video prompt memory override section below
future_obs_index=16

# Whether to train only policy / with video generation.
# only_policy is set in the video prompt memory override section above
policy_and_video_gen=false
only_wo_video_gen=false

# LDA default 4 task weights.
# When only_policy=true, the dataloader automatically uses ["policy"] only.
TRAINING_TASK_WEIGHTS="[1,0,0,0]"

# -------------------------
# Freeze settings
# -------------------------
# Since support prompt uses DINO/world encoder, start by freezing vision encoder.
# You train MM-DiT / action head to use prompt tokens.
freeze_module_list='action_model.vision_encoder'

# If you also want to freeze Qwen-VL:
# freeze_module_list='qwen_vl_interface,action_model.vision_encoder'

# -------------------------
# Training hyperparameters
# -------------------------
# per_device_batch_size is set in the video prompt memory override section above
# repeated_diffusion_steps is set in the video prompt memory override section above

# Gradient accumulation to compensate for reduced batch/repeat.
# Original: 8 GPUs × per_device=4 × repeat=4 × grad_accum=1 = 128 effective samples/step
# Now:      8 GPUs × per_device=2 × repeat=1 × grad_accum=8 = 128 effective samples/step
gradient_accumulation_steps=8

max_train_steps=300000
save_interval=10000
logging_frequency=100
eval_interval=1000

base_lr=4e-5
action_model_lr=1e-4

# -------------------------
# Output
# -------------------------
run_root_dir=/cpfs01/pnx/wordmodels/incontext/LDA-1B/runs
run_id=lda_video_prompt_k${num_support_demos}_t${num_support_frames}

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"

# Save a copy of this launch script
cp "$0" "${output_dir}/launch_script.sh"

echo "=== Training Config ==="
echo "data_root_dir: ${data_root_dir}"
echo "data_mix: ${data_mix}"
echo "run_id: ${run_id}"
echo "use_video_prompt: ${use_video_prompt}"
echo "num_support_demos: ${num_support_demos}"
echo "num_support_frames: ${num_support_frames}"
echo "max_prompt_tokens: ${max_prompt_tokens}"
echo "wrong_prompt_prob: ${wrong_prompt_prob}"
echo "output_dir: ${output_dir}"

# -------------------------
# Launch training
# -------------------------
accelerate launch \
  --config_file lda/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines ${NNODES} \
  --num_processes $((NNODES * NPROC_PER_NODE)) \
  --machine_rank ${NODE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  lda/training/train_LDA.py \
  --config_yaml lda/config/training/LDA_robocasa.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.vision_encoder_path ${vision_encoder_path} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --framework.action_model.max_num_embodiments ${max_num_embodiments} \
  --framework.action_model.state_dim ${state_dim} \
  --framework.action_model.action_dim ${action_dim} \
  --framework.action_model.obs_horizon ${obs_horizon} \
  --framework.action_model.future_obs_index ${future_obs_index} \
  --framework.action_model.only_policy ${only_policy} \
  --framework.action_model.policy_and_video_gen ${policy_and_video_gen} \
  --framework.action_model.only_wo_video_gen ${only_wo_video_gen} \
  --framework.action_model.diffusion_model_cfg.num_layers ${num_layers} \
  --framework.action_model.diffusion_model_cfg.positional_embeddings ${positional_embeddings} \
  \
  --framework.action_model.use_video_prompt ${use_video_prompt} \
  --framework.action_model.max_prompt_tokens ${max_prompt_tokens} \
  \
  --datasets.vla_data.use_delta_action ${use_delta_action} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.training_task_weights ${TRAINING_TASK_WEIGHTS} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${per_device_batch_size} \
  \
  --datasets.vla_data.use_video_prompt ${use_video_prompt} \
  --datasets.vla_data.num_support_demos ${num_support_demos} \
  --datasets.vla_data.num_support_frames ${num_support_frames} \
  --datasets.vla_data.wrong_prompt_prob ${wrong_prompt_prob} \
  \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps ${max_train_steps} \
  --trainer.save_interval ${save_interval} \
  --trainer.logging_frequency ${logging_frequency} \
  --trainer.eval_interval ${eval_interval} \
  --trainer.repeated_diffusion_steps ${repeated_diffusion_steps} \
  --trainer.gradient_accumulation_steps ${gradient_accumulation_steps} \
  --trainer.learning_rate.base ${base_lr} \
  --trainer.learning_rate.action_model ${action_model_lr} \
  --trainer.pretrained_checkpoint ${pretrained_checkpoint} \
  \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project ${wandb_project} \
  --wandb_entity "'${wandb_entity}'" \
  --is_debug False \
  2>&1 | tee "${output_dir}/train.log"
