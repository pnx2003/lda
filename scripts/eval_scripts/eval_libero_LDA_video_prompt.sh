#!/usr/bin/env bash
# ==============================================================================
# LDA libero 闭环 rollout 评测脚本 —— 带 video prompt 版本
#
# 用途:
#   加载一个 "video-prompt 训练" 的 LDA checkpoint,在 libero MuJoCo 仿真里跑闭环
#   rollout,统计不同 video-prompt 模式下的任务成功率,并可选保存 rollout 视频。
#
# 与 eval_libero_LDA.sh 的区别:
#   - checkpoint 必须是 use_video_prompt=true 训练的 libero 模型,即
#     runs/lda_libero_video_prompt_k2_t4/checkpoints/steps_{N}_pytorch_model.pt
#   - --prompt_modes 可选 correct / wrong / shuffled / final_frame,以及 none(对照)。
#     对比各模式成功率即可判断 video prompt 是否真的帮到策略。
#
# 评测脚本本体:
#   lda/eval/eval_libero_video_prompt_rollout.py
#
# 运行环境:
#   conda env = LDA  (/cpfs01/pnx/miniconda3/envs/LDA)
#   渲染后端  = mujoco EGL  (export MUJOCO_GL=egl)
#
# 重要说明:
#   - 日志里 "Failed to load library 'libOpenGL.so.0'" 和退出时 __del__ 的 EGLError
#     是无害噪音,不影响渲染。看真实进度 grep "task[0-9]|=>"。
#   - 评测时 sim 每帧做 flip180 后喂模型(与训练数据一致),脚本内部已处理。
#   - video prompt 需要 support demo:从同 suite 其它 episode 采样 K 个 demo、每个取
#     T 帧,编码成 prompt token 拼到观测前面。K/T 必须与训练一致(默认 k2_t4)。
#   - 第一个 episode 出现较慢(~2-3 分钟),之后约 45s/episode。每个 prompt_mode 都会
#     跑一遍全部 episode,所以模式越多总耗时越长。
# ==============================================================================
set -euo pipefail

# ------------------------------------------------------------------------------
# 0. 运行环境
# ------------------------------------------------------------------------------
source /cpfs01/pnx/miniconda3/bin/activate LDA
cd /cpfs01/pnx/wordmodels/incontext/LDA-1B

# mujoco 渲染后端。egl = GPU offscreen 渲染(本节点 L20Y 可用)。
# 备选:osmesa(纯 CPU 软件渲染,慢)、glx(需 X 显示,一般不用)。
export MUJOCO_GL=egl

# 关闭 huggingface tokenizers 的 fork 并行警告(评测会 fork 子进程,不关会刷屏)。
export TOKENIZERS_PARALLELISM=false

# ------------------------------------------------------------------------------
# 1. 用户可配置参数(按需修改)
# ------------------------------------------------------------------------------

# 要评测的 checkpoint(.pt 文件)。
# 必须是用 video prompt 训练的 libero checkpoint,即
#   runs/lda_libero_video_prompt_k2_t4/checkpoints/steps_{N}_pytorch_model.pt
# 可选 steps:30000..80000(每 10000 一档)。
# 注意:不要用 runs/lda_video_prompt_k2_t4/ —— 那是 droid 训的,norm_stats 是 droid,
#       state_dim=8/num_views=1,与本脚本不兼容,会报 "Expected 'franka' norm_stats"。
MODEL_PATH="/cpfs01/pnx/wordmodels/incontext/LDA-1B/runs/lda_libero_video_prompt_k2_t4/checkpoints/steps_80000_pytorch_model.pt"

# libero 数据集根目录(support demo 从这里采样)。一般不用改。
DATA_ROOT_DIR="/cpfs01/Embodied/datasets"

# 评测哪些 benchmark suite。可选:
#   libero_spatial / libero_object / libero_goal / libero_10
# 完整评测四个都加:--suites libero_spatial libero_object libero_goal libero_10
SUITES="libero_spatial"

# video-prompt 模式(可多选,空格分隔)。脚本会依次跑每个模式并对比。
#   none        : 不喂 support video(基线,看纯策略本身能力)
#   correct     : K 个同类任务(相同 instruction)的 support demo 作 prompt(主实验)
#   wrong       : 不同任务 instruction 的 support demo(对照:有用的话应低于 correct)
#   shuffled    : correct 的 demo 但帧顺序打乱(对照:验证时序是否有用)
#   final_frame : correct 的 demo 压成最后一帧重复 T 次(对照:验证运动信息是否有用)
# 推荐先跑 "none correct" 对比;完整消融再加 wrong/shuffled/final_frame。
PROMPT_MODES="correct"

# 每个 task 跑多少个 episode。标准评测用 20。
# 注意:每个 prompt_mode 都会跑这么多,总 episode 数 = modes数 × tasks × episodes_per_task。
EPISODES_PER_TASK=20

# 每次模型推理输出的 action chunk 长度(一次执行多少步动作)。需与训练对齐,默认 16。
CHUNK_SIZE=16

# video prompt 的 support demo 设置,必须与训练一致(默认 k2_t4)。
#   NUM_SUPPORT_DEMOS  = K,采多少个 support demo(训练用 2)
#   NUM_SUPPORT_FRAMES = T,每个 demo 取多少帧(训练用 4)
NUM_SUPPORT_DEMOS=2
NUM_SUPPORT_FRAMES=4

# 评测用随机种子(影响初始状态采样与 support demo 采样)。
SEED=7

# ------------------------------------------------------------------------------
# 2. 视频录制参数
# ------------------------------------------------------------------------------

# 是否录制 rollout 视频(agentview 第一视角 mp4)。true/false。
RECORD_VIDEOS=true

# 每个 (suite, mode, task) 录前多少个 episode 的视频。3 是看效果够用的值。
VIDEOS_PER_TASK=3

# 视频帧率(fps)。
VIDEO_FPS=20

# ------------------------------------------------------------------------------
# 3. 输出目录
# ------------------------------------------------------------------------------
# 从 MODEL_PATH 解析 steps 编号,结合 suite 和 prompt_modes 命名,避免覆盖。
STEPS=$(basename "$MODEL_PATH" | sed -n 's/steps_\([0-9]*\)_pytorch_model.pt/\1/p')
MODES_TAG=$(echo "$PROMPT_MODES" | tr ' ' '_')
OUTPUT_DIR="runs/lda_libero/eval_libero/videoprompt_steps_${STEPS}_$(echo $SUITES | tr ' ' '_')_${MODES_TAG}"

# ------------------------------------------------------------------------------
# 4. 组装并启动评测
# ------------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"
cp "$0" "$OUTPUT_DIR/launch_script.sh"

# 只有 RECORD_VIDEOS=true 时才加视频参数。
VIDEO_ARGS=""
if [ "$RECORD_VIDEOS" = "true" ]; then
    VIDEO_ARGS="--record_videos --videos_per_task ${VIDEOS_PER_TASK} --video_fps ${VIDEO_FPS}"
fi

echo "========================================================================"
echo "LDA libero eval (video-prompt checkpoint)"
echo "  model_path          : ${MODEL_PATH}"
echo "  suites              : ${SUITES}"
echo "  prompt_modes        : ${PROMPT_MODES}"
echo "  episodes_per_task   : ${EPISODES_PER_TASK}  (per mode)"
echo "  support demos/frames: K=${NUM_SUPPORT_DEMOS} T=${NUM_SUPPORT_FRAMES}"
echo "  record_videos       : ${RECORD_VIDEOS} (videos_per_task=${VIDEOS_PER_TASK})"
echo "  output_dir          : ${OUTPUT_DIR}"
echo "  log                 : ${OUTPUT_DIR}/eval.log"
echo "========================================================================"

# 前台运行(日志同时进屏幕和文件)。
# 想后台跑改成:nohup python ... > "$OUTPUT_DIR/eval.log" 2>&1 &
python lda/eval/eval_libero_video_prompt_rollout.py \
    --model_path "${MODEL_PATH}" \
    --data_root_dir "${DATA_ROOT_DIR}" \
    --suites ${SUITES} \
    --prompt_modes ${PROMPT_MODES} \
    --episodes_per_task ${EPISODES_PER_TASK} \
    --chunk_size ${CHUNK_SIZE} \
    --num_support_demos ${NUM_SUPPORT_DEMOS} \
    --num_support_frames ${NUM_SUPPORT_FRAMES} \
    --seed ${SEED} \
    --output_dir "${OUTPUT_DIR}" \
    ${VIDEO_ARGS} \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"

# ==============================================================================
# 5. 结果说明
# ------------------------------------------------------------------------------
# 跑完后:
#   - 屏幕打印每个 suite|mode 的 success/total = x% 汇总表,以及 video-prompt 效果
#     对比(correct vs none 等)。
#   - ${OUTPUT_DIR}/rollout_results.json : 每个 episode 明细 + 各 mode 汇总。
#   - ${OUTPUT_DIR}/videos/<suite>/<mode>/ : rollout 视频。
#       文件名: task{NN}_ep{NN}_{任务描述}_{OK或x}.mp4  (OK=成功, x=失败)
# ==============================================================================
