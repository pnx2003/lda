#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Unset proxy to avoid wandb connection issues
# ============================================================
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

# ============================================================
# Debug mode: set to true for easier debugging
#   DEBUG_MODE=true bash run_lerobot_datasets_LDA_libero.sh
# ============================================================
DEBUG_MODE=${DEBUG_MODE:-false}

if [ "$DEBUG_MODE" = "true" ]; then
    export CUDA_LAUNCH_BLOCKING=1
    export TORCH_USE_CUDA_DSA=1
    export NCCL_ASYNC_ERROR_HANDLING=1
    echo "=== DEBUG MODE ENABLED ==="
fi

# Reduce CUDA memory fragmentation (recommended by PyTorch OOM message).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# LDA-1B Libero Normal Post-Training Script (no video prompt)
#
# Normal libero training logic:
#   primary/wrist images + state -> vl_embs
#   loss on query action / query future latent
#   No support-demo video prompt, no DINO/world prompt tokens.
#
# Dataset: libero mujoco (libero_mujoco3.3.2), single-arm Franka.
#   state=8 (x,y,z,roll,pitch,yaw,pad,gripper), action=7.
#   The `pad` dim (state index 6) is dropped -> state_dim=7, action_dim=7.
#   Two cameras: primary_image + wrist_image -> num_views=2.
#
# Requires the libero dataloader support registered in:
#   - lda/dataloader/gr00t_lerobot/data_config.py        (LiberoFrankaDataConfig)
#   - lda/dataloader/gr00t_lerobot/embodiment_tags.py    (libero_franka -> FRANKA)
#   - lda/dataloader/gr00t_lerobot/mixtures.py           (libero / libero_spatial / ...)
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

wandb_project=lda-libero
wandb_entity="2100017816"

# -------------------------
# Model paths
# -------------------------
Framework_name=QwenMMDiT

# Local Qwen3-VL checkpoint path
base_vlm=/cpfs01/Embodied/checkpoints/LDA-1B/Qwen3-VL-4B-Instruct

# Parent path of the DINOv3 checkpoint
vision_encoder_path=/cpfs01/pnx/models/

# LDA pretrained checkpoint (set to null if training from scratch)
pretrained_checkpoint=/cpfs01/Embodied/checkpoints/LDA-1B/LDA-pretrain/checkpoints/LDA-pretrain.pt

# -------------------------
# Dataset config
# -------------------------
# Root directory of LeRobot-format datasets.
# The mixture paths are relative to this root, e.g.
#   ${data_root_dir}/libero_mujoco3.3.2/libero_spatial_no_noops_1.0.0_lerobot
data_root_dir=/cpfs01/Embodied/datasets

# Registered in lda/dataloader/gr00t_lerobot/mixtures.py.
# Options (normal, no video prompt):
#   libero            (all 4 suites: spatial+object+goal+10)
#   libero_spatial    (single suite)
#   libero_object
#   libero_goal
#   libero_10
data_mix=libero

# -------------------------
# Libero subset selection (only used when data_mix=libero)
# -------------------------
# By default all 4 suites are used. Set any of these to false to exclude that
# suite. The selection is passed to the dataloader via the LIBERO_EXCLUDE env
# var (read in mixtures.py at import time, so every accelerate worker picks up
# the same selection).
#
# Examples:
#   RUN_LIBERO_SPATIAL=false   -> run object + goal + 10
#   RUN_LIBERO_10=false        -> run spatial + object + goal
#   RUN_LIBERO_GOAL=false RUN_LIBERO_OBJECT=false  -> run spatial + 10
RUN_LIBERO_SPATIAL=${RUN_LIBERO_SPATIAL:-true}
RUN_LIBERO_OBJECT=${RUN_LIBERO_OBJECT:-true}
RUN_LIBERO_GOAL=${RUN_LIBERO_GOAL:-true}
RUN_LIBERO_10=${RUN_LIBERO_10:-true}

_LIBERO_EXCLUDE=""
[ "$RUN_LIBERO_SPATIAL" = "false" ] && _LIBERO_EXCLUDE="${_LIBERO_EXCLUDE},libero_spatial"
[ "$RUN_LIBERO_OBJECT"  = "false" ] && _LIBERO_EXCLUDE="${_LIBERO_EXCLUDE},libero_object"
[ "$RUN_LIBERO_GOAL"    = "false" ] && _LIBERO_EXCLUDE="${_LIBERO_EXCLUDE},libero_goal"
[ "$RUN_LIBERO_10"      = "false" ] && _LIBERO_EXCLUDE="${_LIBERO_EXCLUDE},libero_10"
# strip leading comma
_LIBERO_EXCLUDE="${_LIBERO_EXCLUDE#,}"
export LIBERO_EXCLUDE="${_LIBERO_EXCLUDE}"

echo "=== Libero subset selection ==="
echo "RUN_LIBERO_SPATIAL=${RUN_LIBERO_SPATIAL}  RUN_LIBERO_OBJECT=${RUN_LIBERO_OBJECT}  RUN_LIBERO_GOAL=${RUN_LIBERO_GOAL}  RUN_LIBERO_10=${RUN_LIBERO_10}"
echo "LIBERO_EXCLUDE='${LIBERO_EXCLUDE}' (empty = all 4 suites)"

# -------------------------
# Normal training settings (no video prompt)
# -------------------------
use_video_prompt=false

# -------------------------
# Debug mode overrides
# -------------------------
if [ "$DEBUG_MODE" = "true" ]; then
    NPROC_PER_NODE=1
    echo "=== Debug Parameters ==="
    echo "NPROC_PER_NODE=1"
fi

# -------------------------
# Memory-saving overrides
# -------------------------
# Aligned with the official Robocasa post-train script: only_policy=false with
# 4-task joint training (policy + forward_dynamics + inverse_dynamics + video_gen)
# and repeated_diffusion_steps=4 for stable flow-matching gradients. The auxiliary
# video/dynamics tasks supervise the DiT backbone and help the policy head.
#
# per_device_batch_size: 4-task joint training + repeated_diffusion_steps=4 is
# memory-heavy (the MMDiT joint-attention mask is the OOM hotspot). Default 4
# keeps the joint-attention within 80G; gradient_accumulation_steps is raised
# to 16 below so the effective per-GPU batch (4*16=64) is unchanged from the old
# 8*8. Override via env if your GPUs have headroom:
#   PER_DEVICE_BATCH_SIZE=8 bash run_lerobot_datasets_LDA_libero.sh
# (remember to lower gradient_accumulation_steps to match if you raise batch)
only_policy=false
repeated_diffusion_steps=4
per_device_batch_size=${PER_DEVICE_BATCH_SIZE:-4}

if [ "$DEBUG_MODE" = "true" ]; then
    per_device_batch_size=1
    repeated_diffusion_steps=1
    echo "per_device_batch_size=1"
    echo "repeated_diffusion_steps=1"
fi

# -------------------------
# LDA action model settings
# -------------------------
DIT_TYPE="DiT-L"

obs_horizon=2

# Libero single-arm Franka:
#   state_dim=7 (x,y,z,roll,pitch,yaw,gripper; the `pad` dim is dropped)
#   action_dim=7 (x,y,z,roll,pitch,yaw,gripper)
state_dim=7
action_dim=7

# Libero has two cameras (primary_image + wrist_image)
num_views=2

max_num_embodiments=32
use_delta_action=false

# null, sinusoidal, rope
positional_embeddings=null

num_layers=16
future_obs_index=16

policy_and_video_gen=false
only_wo_video_gen=false

# LDA default 4 task weights (aligned with official Robocasa post-train).
# only_policy=false now, so all 4 tasks are trained jointly:
#   [policy, forward_dynamics, inverse_dynamics, video_gen].
TRAINING_TASK_WEIGHTS="[1,1,1,1]"

# -------------------------
# Freeze settings
# -------------------------
# Freeze vision encoder (DINO/world encoder).
freeze_module_list='action_model.vision_encoder'

# -------------------------
# Training hyperparameters
# -------------------------
# Raised from 8 to 16 to compensate for the lower per_device_batch_size (4),
# keeping the effective per-GPU batch = 4*16 = 64 (= old 8*8) while halving the
# joint-attention memory. If you raise PER_DEVICE_BATCH_SIZE, lower this to match.
gradient_accumulation_steps=16

max_train_steps=20000
save_interval=4000
logging_frequency=100
eval_interval=1000

base_lr=4e-5
action_model_lr=1e-4

# -------------------------
# Output
# -------------------------
run_root_dir=/cpfs01/pnx/wordmodels/incontext/LDA-1B/runs
run_id=lda_libero

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"

# Save a copy of this launch script
cp "$0" "${output_dir}/launch_script.sh"

echo "=== Training Config ==="
echo "data_root_dir: ${data_root_dir}"
echo "data_mix: ${data_mix}"
echo "run_id: ${run_id}"
echo "use_video_prompt: ${use_video_prompt}"
echo "state_dim: ${state_dim}  action_dim: ${action_dim}  num_views: ${num_views}"
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
  --framework.action_model.num_views ${num_views} \
  --framework.action_model.obs_horizon ${obs_horizon} \
  --framework.action_model.future_obs_index ${future_obs_index} \
  --framework.action_model.only_policy ${only_policy} \
  --framework.action_model.policy_and_video_gen ${policy_and_video_gen} \
  --framework.action_model.only_wo_video_gen ${only_wo_video_gen} \
  --framework.action_model.diffusion_model_cfg.num_layers ${num_layers} \
  --framework.action_model.diffusion_model_cfg.positional_embeddings ${positional_embeddings} \
  \
  --framework.action_model.use_video_prompt ${use_video_prompt} \
  \
  --datasets.vla_data.use_delta_action ${use_delta_action} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.training_task_weights ${TRAINING_TASK_WEIGHTS} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${per_device_batch_size} \
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
