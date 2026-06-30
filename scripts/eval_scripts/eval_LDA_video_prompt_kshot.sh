#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# K-Shot Scaling Evaluation —— 多卡并行版
#
# Runs open-loop action prediction with K = 0, 1, 2, 4, 8
# support demos under prompt_mode=correct.
#   K=0 ≡ prompt_mode=none.
#
# 多卡并行: 5 个 K 值之间无依赖, 按 K 分到多张卡上同时跑, 共用 OUT_ROOT
# (各 K 写独立文件 eval_{mode}_k{K}_t{T}.json, 互不冲突), 全部跑完再 summarize。
# 本脚本是纯 open-loop 推理 (droid 数据集), 不含 mujoco/libero/EGL 渲染,
# 所以选卡只需 CUDA_VISIBLE_DEVICES=$g, 不需要 MUJOCO_EGL_DEVICE_ID 那套。
# ============================================================

# -------------------------
# Environment setup
# -------------------------
LDA_ENV_PY=/cpfs01/pnx/miniconda3/envs/LDA/bin/python
if [ -x "${LDA_ENV_PY}" ]; then
  export PATH="/cpfs01/pnx/miniconda3/envs/LDA/bin:${PATH}"
fi
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

OUT_ROOT=./eval_outputs/video_prompt_kshot
mkdir -p "${OUT_ROOT}"

# Eval parameters
MAX_EVAL_TRAJS=20
START_TRAJ=0
ACTION_HORIZON=16
MAX_STEPS=300
SEED=42

# Video prompt settings
NUM_SUPPORT_FRAMES=4

# -------------------------
# GPU 并行参数
# -------------------------
# 用哪几张卡(空格分隔)。改成 "0 1" 就是双卡。并发上限 = 卡数。
GPUS=(0 1 2 3)
NGPU=${#GPUS[@]}
# 要跑的 K 值。K=0 走 prompt_mode=none; 其余走 prompt_mode=correct。
K_VALUES=(0 1 2 4 8)

echo "============================================================"
echo "K-Shot eval (multi-GPU): K=${K_VALUES[*]} on GPUs ${GPUS[*]}"
echo "  output_dir: ${OUT_ROOT}"
echo "============================================================"

# -------------------------
# 单个 K 的评测函数: 绑定指定物理卡, 写独立 log。
# 用法: run_k <phys_gpu> <K> <logfile>
# -------------------------
run_k() {
  local g=$1
  local K=$2
  local logfile=$3

  local mode="correct"
  if [ "$K" -eq 0 ]; then
    mode="none"
  fi

  echo "[K=${K}] GPU ${g} | mode=${mode} -> ${logfile}"
  CUDA_VISIBLE_DEVICES="$g" \
  python lda/eval/eval_LDA_video_prompt.py \
    --config_yaml ${CONFIG_YAML} \
    --model_path ${CHECKPOINT} \
    --prompt_mode ${mode} \
    --data_root_dir ${DATA_ROOT} \
    --data_mix ${DATA_MIX} \
    --output_dir ${OUT_ROOT} \
    --max_eval_trajs ${MAX_EVAL_TRAJS} \
    --start_traj ${START_TRAJ} \
    --action_horizon ${ACTION_HORIZON} \
    --max_steps ${MAX_STEPS} \
    --num_support_demos ${K} \
    --num_support_frames ${NUM_SUPPORT_FRAMES} \
    --seed ${SEED} \
    > "${logfile}" 2>&1
}

# -------------------------
# Job 池: 并发 = NGPU, 每个任务绑一张卡。
# 空闲卡池用数组 free_gpus 维护; 没有空闲卡就 wait -n 等一个任务结束再回收其卡。
# -------------------------
declare -a free_gpus=("${GPUS[@]}")
declare -A job_gpu=()   # pid -> gpu
fail=0
logdir="${OUT_ROOT}/logs"
mkdir -p "${logdir}"

for K in "${K_VALUES[@]}"; do
  # 没有空闲卡, 等任意一个在跑的任务结束并回收它的卡
  while [ ${#free_gpus[@]} -eq 0 ]; do
    wait -n || true
    # 找出已结束的 job, 回收其 GPU
    for pid in "${!job_gpu[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        if ! wait "${pid}" 2>/dev/null; then
          fail=1
          echo "[warn] K-job pid ${pid} exited non-zero"
        fi
        free_gpus+=("${job_gpu[${pid}]}")
        unset 'job_gpu[${pid}]'
      fi
    done
  done

  # 取一张空闲卡
  g=${free_gpus[0]}
  free_gpus=("${free_gpus[@]:1}")

  logfile="${logdir}/k${K}.log"
  run_k "${g}" "${K}" "${logfile}" &
  pid=$!
  job_gpu[${pid}]=${g}
  echo "  -> started pid ${pid} on GPU ${g} for K=${K}"
done

# -------------------------
# 等所有剩余任务结束
# -------------------------
for pid in "${!job_gpu[@]}"; do
  if ! wait "${pid}" 2>/dev/null; then
    fail=1
    echo "[warn] K-job pid ${pid} exited non-zero"
  fi
done

echo ""
if [ ${fail} -ne 0 ]; then
  echo "============================================================"
  echo "WARNING: 部分 K-job 异常退出, 请检查 ${logdir}/k*.log"
  echo "============================================================"
fi

# -------------------------
# Summarize
# -------------------------
echo "============================================================"
echo "Summarizing K-shot results..."
echo "============================================================"

python lda/eval/summarize_video_prompt_eval.py \
  --eval_dir ${OUT_ROOT} \
  --k 2

echo "Done! Check ${OUT_ROOT}/video_prompt_ablation_results.md"
echo "per-K logs -> ${logdir}/k*.log"
