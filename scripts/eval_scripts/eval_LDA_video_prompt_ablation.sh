#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Video Prompt Ablation Evaluation
#
# Runs open-loop action prediction under 5 prompt modes:
#   none / correct / wrong / shuffled / final_frame
#
# After all modes finish, runs the summarizer to compute
# gap metrics and produce the final table.
# ============================================================

# -------------------------
# Environment setup
# -------------------------
# Use the LDA conda env (python 3.10, torch 2.6+cu124) that the model was
# trained with. Activate if not already in it.
LDA_ENV_PY=/cpfs01/pnx/miniconda3/envs/LDA/bin/python
if [ -x "${LDA_ENV_PY}" ]; then
  export PATH="/cpfs01/pnx/miniconda3/envs/LDA/bin:${PATH}"
fi
# Single GPU for offline eval (model fits on one L20Y).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Repo root on PYTHONPATH so `import lda` works.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

# -------------------------
# Config (EDIT THESE)
# -------------------------
# Run-specific config from the training run directory (matches the checkpoint).
RUN_DIR=/cpfs01/pnx/wordmodels/incontext/LDA-1B/runs/lda_video_prompt_k2_t4
CHECKPOINT=${RUN_DIR}/checkpoints/steps_190000_pytorch_model.pt
CONFIG_YAML=${RUN_DIR}/config.yaml

# Data: trained on droid_video_prompt (in-distribution eval).
DATA_ROOT=/cpfs01/Embodied/datasets
DATA_MIX=droid_video_prompt

OUT_ROOT=./eval_outputs/video_prompt_ablation
mkdir -p "${OUT_ROOT}"

# Eval parameters
MAX_EVAL_TRAJS=20
START_TRAJ=0
ACTION_HORIZON=16
MAX_STEPS=300
SEED=42

# Video prompt settings
NUM_SUPPORT_DEMOS=2
NUM_SUPPORT_FRAMES=4

# -------------------------
# Run each prompt mode
# -------------------------
PROMPT_MODES=(
  none
  correct
  wrong
  shuffled
  final_frame
)

for MODE in "${PROMPT_MODES[@]}"; do
  echo "============================================================"
  echo "Running eval: prompt_mode=${MODE}"
  echo "============================================================"

  python lda/eval/eval_LDA_video_prompt.py \
    --config_yaml ${CONFIG_YAML} \
    --model_path ${CHECKPOINT} \
    --prompt_mode ${MODE} \
    --data_root_dir ${DATA_ROOT} \
    --data_mix ${DATA_MIX} \
    --output_dir ${OUT_ROOT} \
    --max_eval_trajs ${MAX_EVAL_TRAJS} \
    --start_traj ${START_TRAJ} \
    --action_horizon ${ACTION_HORIZON} \
    --max_steps ${MAX_STEPS} \
    --num_support_demos ${NUM_SUPPORT_DEMOS} \
    --num_support_frames ${NUM_SUPPORT_FRAMES} \
    --seed ${SEED}

  echo ""
done

# -------------------------
# Summarize
# -------------------------
echo "============================================================"
echo "Summarizing results..."
echo "============================================================"

python lda/eval/summarize_video_prompt_eval.py \
  --eval_dir ${OUT_ROOT} \
  --k ${NUM_SUPPORT_DEMOS}

echo "Done! Check ${OUT_ROOT}/video_prompt_ablation_results.md"
