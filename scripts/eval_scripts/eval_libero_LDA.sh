#!/usr/bin/env bash
# ==============================================================================
# LDA libero 闭环 rollout 评测脚本(正常评测,不开 video prompt)
#
# 用途:
#   加载一个训练好的 LDA checkpoint,在 libero MuJoCo 仿真里跑闭环 rollout,
#   统计各 benchmark suite 的任务成功率(success rate),并可选保存 rollout 视频。
#
# 评测脚本本体:
#   lda/eval/eval_libero_video_prompt_rollout.py
#   (名字里带 video_prompt,但传 --prompt_modes none 就是纯策略 baseline 评测)
#
# 运行环境:
#   conda env = LDA  (/cpfs01/pnx/miniconda3/envs/LDA)
#   渲染后端  = mujoco EGL  (export MUJOCO_GL=egl)
#
# 重要说明:
#   - 日志里会出现 "Failed to load library 'libOpenGL.so.0'" 的 INFO 行,以及进程
#     退出时 GLContext.__del__ 报的 EGLError —— 这些是无害的清理噪音,不影响渲染,
#     不代表评测失败。看真实进度请 grep "task[0-9]|=>" 。
#   - 评测时 sim 每帧会做 flip180 后再喂模型(与训练数据一致),脚本内部已处理。
#   - 第一个 episode 出现较慢(~2-3 分钟,benchmark 初始化),之后约 45s/episode。
#   - 200 episodes(spatial 全量)约需 2.5-3 小时。
# ==============================================================================
set -euo pipefail

# ------------------------------------------------------------------------------
# 0. 运行环境:激活 LDA conda 环境,设置渲染与并行参数
# ------------------------------------------------------------------------------
source /cpfs01/pnx/miniconda3/bin/activate LDA
cd /cpfs01/pnx/wordmodels/incontext/LDA-1B

# mujoco 渲染后端。egl = 用 GPU 做 offscreen 渲染(本节点 L20Y 可用)。
# 备选: osmesa(纯 CPU 软件渲染,慢)、glx(需 X 显示,一般不用)。
export MUJOCO_GL=egl

# 关闭 huggingface tokenizers 的 fork 并行警告(评测会 fork 子进程,不关会刷屏)。
export TOKENIZERS_PARALLELISM=false

# ------------------------------------------------------------------------------
# 1. 用户可配置参数(按需修改)
# ------------------------------------------------------------------------------

# 要评测的 checkpoint(.pt 文件)。steps 越大训练越充分。
# 可选: steps_{10000..70000}_pytorch_model.pt
MODEL_PATH="/cpfs01/pnx/wordmodels/incontext/LDA-1B/runs/lda_libero/checkpoints/steps_270000_pytorch_model.pt"

# libero 数据集根目录(mixture 路径相对它)。一般不用改。
DATA_ROOT_DIR="/cpfs01/Embodied/datasets"

# 评测哪些 benchmark suite。可选:
#   libero_spatial / libero_object / libero_goal / libero_10
# 完整评测四个都加上:  --suites libero_spatial libero_object libero_goal libero_10
SUITES="libero_spatial"

# video-prompt 模式。none = 不用 video prompt,纯策略 baseline(即"正常 LDA 评测")。
# 其它取值(correct/wrong/shuffled/final_frame)需要 checkpoint 训练时 use_video_prompt=true 才有效。
PROMPT_MODES="none"

# 每个 task 跑多少个 episode。标准评测用 20。
# spatial=10 任务 → 200 episodes;四个 suite 全跑约 50 任务 → 1000 episodes。
EPISODES_PER_TASK=20

# 每次模型推理输出的 action chunk 长度(一次推理执行多少步动作)。
# 需与训练的 future_obs_index(=16)对齐,一般不改。
CHUNK_SIZE=16

# 评测用随机种子(影响初始状态采样)。
SEED=7

# ------------------------------------------------------------------------------
# 2. 视频录制参数
# ------------------------------------------------------------------------------

# 是否录制 rollout 视频(agentview 第一视角 mp4)。
# true = 录制;false = 不录制(只统计成功率)。
RECORD_VIDEOS=true

# 每个 (suite, mode, task) 录制前多少个 episode 的视频。
# 例如 =3 且 episodes_per_task=20,则每 task 只把前 3 个 episode 存成视频,其余 17 个只统计不录像。
# 录太多会占大量磁盘;3 是个看效果够用的值。
VIDEOS_PER_TASK=3

# 视频帧率(fps)。
VIDEO_FPS=20

# ------------------------------------------------------------------------------
# 3. 输出目录(根据 checkpoint steps 自动命名)
# ------------------------------------------------------------------------------
# 从 MODEL_PATH 解析出 steps 编号,用于命名输出目录,避免覆盖。
STEPS=$(basename "$MODEL_PATH" | sed -n 's/steps_\([0-9]*\)_pytorch_model.pt/\1/p')
OUTPUT_DIR="runs/lda_libero/eval_libero/steps_${STEPS}_$(echo $SUITES | tr ' ' '_')"

# ------------------------------------------------------------------------------
# 4. 组装并启动评测
# ------------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"

# 把本启动脚本存一份到输出目录,便于复现。
cp "$0" "$OUTPUT_DIR/launch_script.sh"

# 拼接 video 相关参数(只有 RECORD_VIDEOS=true 时才加)。
VIDEO_ARGS=""
if [ "$RECORD_VIDEOS" = "true" ]; then
    VIDEO_ARGS="--record_videos --videos_per_task ${VIDEOS_PER_TASK} --video_fps ${VIDEO_FPS}"
fi

echo "========================================================================"
echo "LDA libero eval"
echo "  model_path       : ${MODEL_PATH}"
echo "  suites           : ${SUITES}"
echo "  prompt_modes     : ${PROMPT_MODES}"
echo "  episodes_per_task: ${EPISODES_PER_TASK}"
echo "  record_videos    : ${RECORD_VIDEOS} (videos_per_task=${VIDEOS_PER_TASK})"
echo "  output_dir       : ${OUTPUT_DIR}"
echo "  log              : ${OUTPUT_DIR}/eval.log"
echo "========================================================================"

# 直接前台运行(日志同时打到屏幕和文件)。
# 想后台跑可改成:nohup python ... > "$OUTPUT_DIR/eval.log" 2>&1 &
python lda/eval/eval_libero_video_prompt_rollout.py \
    --model_path "${MODEL_PATH}" \
    --data_root_dir "${DATA_ROOT_DIR}" \
    --suites ${SUITES} \
    --prompt_modes ${PROMPT_MODES} \
    --episodes_per_task ${EPISODES_PER_TASK} \
    --chunk_size ${CHUNK_SIZE} \
    --seed ${SEED} \
    --output_dir "${OUTPUT_DIR}" \
    ${VIDEO_ARGS} \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"

# ==============================================================================
# 5. 结果说明
# ------------------------------------------------------------------------------
# 跑完后:
#   - 成功率汇总表会打印到屏幕(每个 suite|mode 的 success/total = x%)。
#   - ${OUTPUT_DIR}/rollout_results.json : 每个 episode 的明细 + 汇总。
#   - ${OUTPUT_DIR}/videos/<suite>/<mode>/ : rollout 视频(RECORD_VIDEOS=true 时)。
#       文件名: task{NN}_ep{NN}_{任务描述}_{OK或x}.mp4  (OK=成功, x=失败)
# ==============================================================================
