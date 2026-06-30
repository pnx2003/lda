#!/usr/bin/env bash
# ==============================================================================
# LDA libero 闭环 rollout 评测 —— 4 卡并行版(按 task 分片)
#
# 把 libero_spatial 的 10 个 task 按 task_id % 4 分到 4 张卡, 每卡一个 worker
# 独立跑各自的 task 子集, 各写 worker{0..3}/ 子目录, 互不覆盖; 跑完自动合并
# 全局成功率。相对单卡串行约 4x 加速。
#
# 运行: bash scripts/eval_scripts/eval_libero_LDA_4gpu.sh
# 前置: 已给 lda/eval/eval_libero_video_prompt_rollout.py 加 --task_ids 参数。
# ==============================================================================
set -euo pipefail

# ------------------------------------------------------------------------------
# 0. 环境
# ------------------------------------------------------------------------------
source /cpfs01/pnx/miniconda3/bin/activate LDA
cd /cpfs01/pnx/wordmodels/incontext/LDA-1B
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

# ------------------------------------------------------------------------------
# 1. 用户可配置参数(按需修改)
# ------------------------------------------------------------------------------
# 要评测的 checkpoint(.pt 文件)。
MODEL_PATH="/cpfs01/pnx/wordmodels/incontext/LDA-1B/runs/lda_libero/checkpoints/steps_16000_pytorch_model.pt"

# libero 数据集根目录。一般不用改。
DATA_ROOT_DIR="/cpfs01/Embodied/datasets"

# 评测哪些 benchmark suite。4 卡并行按 task 分片, 仅支持单 suite(多 suite 顺序跑)。
SUITES="libero_spatial"

# video-prompt 模式。none = 纯策略 baseline。
PROMPT_MODES="none"

# 每个 task 跑多少个 episode。标准评测用 20。
EPISODES_PER_TASK=20

# 模型推理输出的 action chunk 长度(需与训练 future_obs_index=16 对齐)。
CHUNK_SIZE=16

# 评测用随机种子(影响初始状态采样)。
SEED=7

# ------------------------------------------------------------------------------
# 2. 录像参数
# ------------------------------------------------------------------------------
RECORD_VIDEOS=true
VIDEOS_PER_TASK=3
VIDEO_FPS=20

# ------------------------------------------------------------------------------
# 3. GPU 与分片参数
# ------------------------------------------------------------------------------
# 用哪几张卡(空格分隔)。改成 "0 1" 就是双卡。
GPUS=(0 1 2 3)
NGPU=${#GPUS[@]}

# libero_spatial/object/goal/10 都是 10 个 task。
NTASKS=10

# ------------------------------------------------------------------------------
# 4. 输出目录
# ------------------------------------------------------------------------------
STEPS=$(basename "$MODEL_PATH" | sed -n 's/steps_\([0-9]*\)_pytorch_model.pt/\1/p')
OUTPUT_DIR="runs/lda_libero/eval_libero/steps_${STEPS}_$(echo $SUITES | tr ' ' '_')_4gpu"
mkdir -p "$OUTPUT_DIR"
cp "$0" "$OUTPUT_DIR/launch_script.sh" 2>/dev/null || true

VIDEO_ARGS=""
if [ "$RECORD_VIDEOS" = "true" ]; then
    VIDEO_ARGS="--record_videos --videos_per_task ${VIDEOS_PER_TASK} --video_fps ${VIDEO_FPS}"
fi

echo "========================================================================"
echo "LDA libero eval (4-GPU sharded)"
echo "  model_path       : ${MODEL_PATH}"
echo "  suites           : ${SUITES}"
echo "  prompt_modes     : ${PROMPT_MODES}"
echo "  episodes_per_task: ${EPISODES_PER_TASK}"
echo "  gpus             : ${GPUS[*]} (${NGPU} workers)"
echo "  output_dir       : ${OUTPUT_DIR}"
echo "========================================================================"

# ------------------------------------------------------------------------------
# 5. 启动各 worker (后台并行)
# ------------------------------------------------------------------------------
pids=()
for i in "${!GPUS[@]}"; do
    g=${GPUS[$i]}
    # task_id % NGPU == i 的 task 分给该 worker
    ids=()
    for t in $(seq 0 $((NTASKS-1))); do
        if [ $((t % NGPU)) -eq $i ]; then
            ids+=("$t")
        fi
    done
    echo "[worker${i}] GPU ${g} -> tasks ${ids[*]}"

    # GPU 路由说明(两道 assert 都要过):
    #   1) robosuite binding_utils.py: 要求 MUJOCO_EGL_DEVICE_ID 是
    #      CUDA_VISIBLE_DEVICES 字符串的子串(如 "0" in "1,0"). 故 CVD 必须含字符 '0'.
    #   2) mujoco egl: 设了 CVD 后只看到 1 个 EGL 设备, MUJOCO_EGL_DEVICE_ID 必须=0.
    #   3) torch cuda:0 = CVD 列表第一张卡 = 物理卡 $g.
    # 所以: CUDA_VISIBLE_DEVICES="$g,0"(首元素=目标物理卡, 且含字符'0'),
    #        MUJOCO_EGL_DEVICE_ID=0. 渲染和推理都落在物理卡 $g.
    # 注意: worker0 的目标物理卡本身就是 0, 若再写 "0,0" 会重复枚举同一设备,
    #       NVIDIA 驱动据此返回空设备列表 -> torch.cuda.is_available()=False,
    #       模型落到 CPU, flash-attn 报错退出. 故 worker0 单独用 "0".
    if [ "$g" = "0" ]; then
        _cvd="0"
    else
        _cvd="$g,0"
    fi
    CUDA_VISIBLE_DEVICES="$_cvd" \
    MUJOCO_EGL_DEVICE_ID=0 \
    python lda/eval/eval_libero_video_prompt_rollout.py \
        --model_path "${MODEL_PATH}" \
        --data_root_dir "${DATA_ROOT_DIR}" \
        --suites ${SUITES} \
        --prompt_modes ${PROMPT_MODES} \
        --episodes_per_task ${EPISODES_PER_TASK} \
        --chunk_size ${CHUNK_SIZE} \
        --seed ${SEED} \
        --task_ids ${ids[*]} \
        --output_dir "${OUTPUT_DIR}/worker${i}" \
        ${VIDEO_ARGS} \
        > "${OUTPUT_DIR}/worker${i}.log" 2>&1 &
    pids+=($!)
done

# ------------------------------------------------------------------------------
# 6. 等待全部 worker 结束
# ------------------------------------------------------------------------------
echo ""
echo "waiting for ${NGPU} workers (pids: ${pids[*]}) ..."
fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        fail=1
        echo "[warn] worker pid ${pid} exited non-zero"
    fi
done

if [ $fail -ne 0 ]; then
    echo "=============================================================="
    echo "WARNING: 部分 worker 异常退出, 请检查 ${OUTPUT_DIR}/worker*.log"
    echo "=============================================================="
fi

# ------------------------------------------------------------------------------
# 7. 合并各 worker 结果, 求全局成功率
# ------------------------------------------------------------------------------
echo ""
echo "merging worker results ..."
python scripts/eval_scripts/merge_libero_results.py "${OUTPUT_DIR}"

echo ""
echo "done. merged results -> ${OUTPUT_DIR}/rollout_results_merged.json"
echo "per-worker logs     -> ${OUTPUT_DIR}/worker*.log"
echo "rollout videos      -> ${OUTPUT_DIR}/worker*/videos/"
# ==============================================================================
