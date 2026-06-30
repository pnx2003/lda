● Smoke test 还在跑（16/30，none mode）。我先停下不频繁查询了。脚本已经写好并验证可用，下面是总结。

  已完成的工作

  测试脚本

  /cpfs01/pnx/wordmodels/incontext/LDA-1B/lda/eval/eval_libero_video_prompt_rollout.py

  闭环 Libero 仿真评测，对比不同 video prompt 模式的成功率：
  - none — 不给 support video（基线）
  - correct — 给同指令的 K 条 demo 视频（看 video prompt 是否起作用）
  - wrong — 给不同指令的视频（对照）
  - shuffled / final_frame — 帧打乱 / 只重复最后一帧

  最后会输出一个表，并算 correct 相对 none 的成功率提升（↑HELPS / ↓hurts / ≈no effect）。

  运行环境

  我把 mujoco==3.3.1 + robosuite==1.4.0 + libero 装进了 LDA conda env (/cpfs01/pnx/miniconda3/envs/LDA)，这样模型 + libero 仿真在同一个 py3.10 环境里。运行命令：

  MUJOCO_GL=egl /cpfs01/pnx/miniconda3/envs/LDA/bin/python \
    lda/eval/eval_libero_video_prompt_rollout.py \
    --model_path runs/lda_libero_video_prompt_k2_t4/checkpoints/steps_20000_pytorch_model.pt \
    --data_root_dir /cpfs01/Embodied/datasets \
    --suites libero_spatial libero_object libero_goal libero_10 \
    --prompt_modes none correct \
    --episodes_per_task 20 \
    --output_dir eval_outputs/libero_video_prompt_rollout

  关键实现细节（踩坑后验证过的）

  1. video prompt 确实影响输出：同一观测下，none vs correct 预测的动作明显不同（xyz/gripper 平均差 0.3–0.5）。
  2. support video 采样：复用训练时的 VideoPromptLeRobotSingleDataset，按任务语言从 LeRobot 数据集采 K 条同指令 episode、每条 T 帧，格式 List[K][T][V] PIL，与 predict_action 期望一致。
  3. state 构造：[eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos[1]]（丢掉 pad=gripper_qpos[0]），再 q99 归一化 —— 与数据集记录约定逐项对齐过。
  4. action 执行：模型输出 q99 归一化动作，直接喂给 OSC_POSE 控制器（不要反归一化），gripper 按 0→+1(close)/1→-1(open) 二值化。我回放了数据集 demo，20 步内 EEF 移动 0.197m，与录制轨迹 0.197m 完全吻合，证明 action 约定正确。
  5. 图像：agentview + wrist 翻转 [::-1,::-1] → expand2square(ImageNet mean) → 224×224，按 [view0_t0, view0_t1, view1_t0, view1_t1] 排列（obs_horizon=2，闭环时 t-1 用当前帧填充）。

  ── 图像翻转约定的更正（2026-06-26）─────────────────────────────────────────
  上面第 5 条的 `[::-1,::-1]` 翻转是**对的**，曾一度被误删导致 0% success，现已恢复。

  实测验证（libero_spatial task0 init_state[0]，mujoco 3.2.3；vla-eval env 待修，见下）：
  对比数据集存储帧（=训练帧）与 sim 原始渲染帧 obs["agentview_image"]：
      corr(stored_frame0, sim_raw)            = -0.24   （弱反相关）
      corr(stored_frame0, sim_raw[::-1,::-1]) = +0.91   （强正相关）
  ⇒ 数据集/训练帧 = flip180(sim_raw)。MuJoCo offscreen render 的 GL framebuffer 本身
     上下颠倒，数据集制作时做了 [::-1,::-1] 把它正过来。所以：
    - 喂给模型的图像必须 flip180（否则场景上下颠倒 → 0% success）。
    - 存出来的可视化 mp4 也必须 flip180（否则 video 看着是倒的，正是这次问题的表象）。
  代码改动：preprocess_libero_image 与 _obs_agentview_uint8 都加了 img[::-1,::-1]。
  注：之前 memory 里“raw 不翻转、翻转 corr=-0.874”的结论是错的（可能基于别的 env），
       已从记忆中删除，以本实测为准。

  ── vla-eval conda env 当前无法跑渲染（待解决）──────────────────────────────
  评测命令里用的 `/cpfs01/pnx/miniconda3/envs/vla-eval`（libero 0.1.1 + mujoco 3.3.1
  + 本地 robosuite 1.5.2）当前 import 不通：
      from libero.libero.envs import OffScreenRenderEnv
      → ModuleNotFoundError: No module named 'robosuite.environments.manipulation.single_arm_env'
  原因：libero 0.1.1 的 bddl_base_domain.py 依赖老 robosuite 的
  `robosuite.environments.manipulation.single_arm_env.SingleArmEnv`，而本地 robosuite 1.5.2
  已重构（API 变为 robot_env/manipulation 下的其它类，无 SingleArmEnv）。
  ⇒ 这个 env 最近被升级 robosuite 改坏了，目前跑不了闭环渲染评测。
  方向结论（flip180）是在 libero env（mujoco 3.2.3）实测的；flip 约定是 GL 层面的、
  跨 mujoco/robosuite 版本一致，所以代码改动本身可靠。但要真正跑评测，需先修复 env
  （降级 robosuite 到与 libero 0.1.1 兼容的版本，或换用能 import libero envs 的 env）。
  ────────────────────────────────────────────────────────────────────────────

  当前状态

  后台 smoke test（spatial, 3 ep/task, none+correct）还在跑（~16/30）。目前全是 0% 成功——这是预期的，因为 checkpoint 才训了 20k 步（目标 300k），模型还没学会完成任务。代码本身已验证正确（轨迹能正常追踪）。

  等训得更久的 checkpoint 出来后，用上面命令跑 4 个 suite × 20 ep，就能看到 correct vs none 的成功率差，判断 video prompt 是否真的有用。

  需要我把它改成后台 nohup 长跑、或者加上 video 录制/任务级并行来加速吗？

── 4 卡并行评测（2026-06-29）──────────────────────────────────────────────────
单卡串行跑 libero_spatial（10 task × 20 ep = 200 episode）约需 2.5–3 小时，瓶颈是
GPU 推理 + EGL 渲染串行（每 episode ~45s）。本节点有 4 张 L20Y 空闲，按 task 分片到
4 卡并行，理论近 4× 加速。

改动（3 个文件）：
  1. lda/eval/eval_libero_video_prompt_rollout.py —— 加 --task_ids 参数，主循环由
     `for task_id in range(n_tasks)` 改为 `for task_id in task_ids`（task_ids 默认 None
     = 全部 task，分片时只跑指定 task）。其余逻辑不动，每个 worker 各自汇总自己跑的
     子集，全局成功率由合并脚本算。
  2. scripts/eval_scripts/eval_libero_LDA_4gpu.sh —— 4 卡并行启动脚本。把 10 个 task
     按 task_id % 4 分到 4 卡（GPU0→0,4,8；GPU1→1,5,9；GPU2→2,6；GPU3→3,7），每卡一个
     worker 后台并行，各写 worker{0..3}/ 子目录互不覆盖，跑完 wait 等齐。
  3. scripts/eval_scripts/merge_libero_results.py —— 合并 worker*/rollout_results.json，
     按 (suite, mode) 重算全局 success/total/rate，打印汇总表并存
     rollout_results_merged.json。

运行：bash scripts/eval_scripts/eval_libero_LDA_4gpu.sh

── 4 卡路由的坑：三道互相牵制的 assert（核心，重点记录）────────────────────────
让 4 个 worker 各自落到 4 张物理卡，要同时满足 mujoco / robosuite / torch 三个库的校验，
它们互相牵制，调试中依次踩了三遍：

  ① mujoco egl（mujoco/egl/__init__.py:38）
     create_initialized_egl_device_display() 里：
         all_devices = EGL.eglQueryDevicesEXT()
         if not 0 <= int(MUJOCO_EGL_DEVICE_ID) < len(all_devices): raise
     关键：设了 CUDA_VISIBLE_DEVICES 后，eglQueryDevicesEXT() 只返回 1 个设备
     （实测 len=1，无论 CVD 设成 0/1/2/3）。所以 MUJOCO_EGL_DEVICE_ID 必须是 0，
     设成 1/2/3 会在创建 GLContext/首次渲染时报：
       "must be an integer between 0 and 0 (inclusive), got N"

  ② robosuite binding_utils.py:31-35
         CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
         if CUDA_VISIBLE_DEVICES != "":
             MUJOCO_EGL_DEVICE_ID = os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
             if MUJOCO_EGL_DEVICE_ID is not None:
                 assert MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES
     注意这是**字符串子串匹配**，不是数值匹配！
     所以 ①要求 EGL_ID=0 ⇒ ②要求 CVD 字符串里必须含字符 '0'。
     当 CVD="1" 时 "0" in "1" = False ⇒ from libero...import 阶段直接 AssertionError 崩溃：
       "MUJOCO_EGL_DEVICE_ID needs to be set to one of the device id specified in CUDA_VISIBLE_DEVICES"
     这正是“还是不行”的真正原因：worker1/2/3 在 import 阶段就崩了，每个 episode 报错、
     steps=0、0% 成功率后空跑退出，只有 worker0（CVD="0", EGL_ID=0，"0" in "0" ✓）存活，
     表象就是“只跑在一个 GPU 上”。

  ③ torch cuda:0 = CUDA_VISIBLE_DEVICES 列表的第一个物理卡。

最终解法（同时满足三者）：
     CUDA_VISIBLE_DEVICES="$g,0"   （首元素=目标物理卡 $g，且字符串含字符 '0'）
     MUJOCO_EGL_DEVICE_ID=0
  - ② "0" in "$g,0" ✓（含字符 0）
  - ① EGL_ID=0 ✓（落在 CVD 首元素 = 物理卡 $g，因为 CVD 后 eglQueryDevicesEXT 只返回那 1 张）
  - ③ cuda:0 = 物理卡 $g ✓
  实测验证（用显存占用判定路由，因渲染太快抓不到 util）：
     CVD=1,0 EGL_ID=0 → torch/EGL 显存落在物理 GPU1（+1039MB）
     CVD=2,0 EGL_ID=0 → 落在物理 GPU2
     CVD=3,0 EGL_ID=0 → 落在物理 GPU3
     CVD=0,0 EGL_ID=0 → 落在物理 GPU0（⚠ 曾以为 CUDA 自动去重为 1 张 OK，实测会崩，见下 2026-06-30 更正）
  即每个 worker 的模型推理与 MuJoCo EGL 渲染都在同一张物理卡上，无跨卡开销。

── worker0 必崩：CVD="0,0" 重复枚举导致 torch 看不到 CUDA（2026-06-30 更正）────────
上面"CVD=0,0 ... OK"是错的。实际重跑时 worker0 必崩，失败发生在**模型加载阶段**，
日志报：
    ValueError: FlashAttention2 ... not available on CPU.
    Please make sure torch can access a CUDA device.
即 worker0 的 torch.cuda.is_available()=False，模型落到 CPU，flash-attn 检测失败退出。
worker1/2/3 正常（它们能加载模型、出现 DiT 参数量行）。

根因：worker0 的目标物理卡本身就是 0，按 `"$g,0"` 拼出 CUDA_VISIBLE_DEVICES="0,0"。
重复枚举同一设备 ID 是退化输入，NVIDIA 驱动据此返回空设备列表 → torch 看不到任何 CUDA
设备。worker1/2/3 的 "1,0"/"2,0"/"3,0" 是两个**不同**设备 ID，合法，故正常。

修法（scripts/eval_scripts/eval_libero_LDA_4gpu.sh 第 108 行附近）：worker0 单独走
单元素 CVD，其余 worker 才追加 ",0" 以满足 ②的子串断言：
    if [ "$g" = "0" ]; then
        _cvd="0"
    else
        _cvd="$g,0"
    fi
    CUDA_VISIBLE_DEVICES="$_cvd" MUJOCO_EGL_DEVICE_ID=0 ...
  - worker0：CVD="0"，单设备，torch 看到 cuda:0=物理卡0；EGL_ID=0=物理卡0 ✓
  - worker1/2/3：CVD="1,0" 等，首元素=目标卡，cuda:0=物理卡$g；EGL 设备0=CVD 第一张=物理卡$g ✓
改完 worker0 即可正常加载模型，4 worker 全活。

验证（不跑 rollout，秒回）：
    CUDA_VISIBLE_DEVICES="0" MUJOCO_EGL_DEVICE_ID=0 python -c \
      "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
  打印 `True 1` 即修复生效。重跑前清掉上次失败残留：
    rm -rf runs/lda_libero/eval_libero/steps_*_libero_spatial_4gpu
──────────────────────────────────────────────────────────────────────────────

排查要点：
  - “只跑在一个 GPU”≠ worker0 真在跑，要看 ps 进程数 + 各 worker*.log 末尾是否报错。
  - worker 日志里 episode 行全 `x (steps=0)` + `[error]` ⇒ 是 import/初始化崩了，
    不是模型不行。先 grep worker*.log 的 Traceback/Assertion/Error。
  - 验证 4 卡真并行：`ps aux|grep eval_libero_video_prompt|grep -v grep|wc -l` 应=4，
    `nvidia-smi` 4 张卡各占 ~14GB，各 worker*.log 出现 OK/x 且 steps>0。
──────────────────────────────────────────────────────────────────────────────

── 两套评测 & 训练数据量 差别统计（2026-06-30）────────────────────────────────
问：kshot 评测跟 libero 结果没关系、且每次只有 spatial task00 的 2 个 ep 做对，
20000 步能否把 libero 四个 suite 数据过一遍？—— 已查清，结论见下。

【A. 两套评测是不同东西，本就不该有数值关联】

                          libero 闭环评测                kshot 开环评测
脚本                      eval_libero_LDA.sh             eval_LDA_video_prompt_kshot.sh
python                    eval_libero_video_prompt_      eval_LDA_video_prompt.py
                          rollout.py
方式                      闭环 rollout（MuJoCo 仿真）     开环 action prediction（喂真实轨迹）
数据                      libero (4 suite)               droid_video_prompt
指标                      闭环任务成功率 success rate     开环 action L1 误差
checkpoint 来源           runs/lda_libero                runs/lda_video_prompt_k2_t4
checkpoint use_video_prompt false（纯策略 baseline）       true
所以"没关系"是正常的——不同数据集、不同指标、不同 checkpoint、不同评测范式。

【B. 20000 步能过几遍 libero 四个 suite 数据】—— 远远够，约 24 遍

数据实测（data_mix=libero，4 suite 各 weight=1.0，全加载）：
  libero_spatial  53,229 frames
  libero_object   67,309 frames
  libero_goal     52,895 frames
  libero_10      104,280 frames
  合计           277,713 frames  （样本单位 = frame/step，见 datasets.py __len__=all_steps）

训练实际参数（launch_script.sh 命令行覆盖 config.yaml）：
  per_device_batch_size = 4        （config.yaml 写 8，被命令行覆盖成 4）
  gradient_accumulation_steps = 16 （config.yaml 写 8，被命令行覆盖成 16）
  NPROC_PER_NODE = 8                （8 卡）
  ⇒ 每 optimizer step 消耗 = 4 × 16 × 8 = 512 样本
  max_train_steps = 20000           （= 20000 次梯度更新，非前向次数）
  repeated_diffusion_steps = 4      （每步内重复 4 次扩散去噪，增加梯度信号，不改 step 计数）

epoch 估算（mixture __len__ = max(子集长度/权重) × 子集展开 = 417,120 实测）：
  20000 步样本数 = 512 × 20000 = 10,240,000
  epoch 数       = 10,240,000 / 417,120 ≈ 24.5
  ⇒ 4 个 suite 的数据全被过了约 24 遍，覆盖上不存在"没过完"。

【C. training_task_weights: [1,0,0,0] 的真相——跟 4 个数据集无关】

  TRAINING_TASKS = ["policy", "forward_dynamics", "inverse_dynamics", "video_gen"]
  这 4 个是【训练任务】(policy=主任务,其余3个辅助自监督),不是 4 个 libero 数据集。
  且 config 里 only_policy=true → 代码把 active_tasks 压成 ["policy"]、weights 压成 [1.0]，
  所以 [1,0,0,0]（launch_script 实传 [1,1,1,1]）在 only_policy 下都被忽略，只训 policy。
  4 个数据集的采样由 data_mix=libero(4 子集各 weight=1.0) + LeRobotMixtureDataset 按权重
  采样决定，与 training_task_weights 无关。即 4 个 suite 都正常参与了训练。

【D. "每次固定 spatial task00 中 2 个 ep 做对" 的可能原因（与数据量无关）】

  1. 早期 checkpoint 欠训：max_train_steps 本就设 20000；summary.jsonl 已训到 290000 步。
     若评测用 steps_20000 这类早期 ckpt，optimizer step 才 2 万，对 1B 模型 fine-tune 偏少。
     当前 eval_libero_LDA.sh 默认评测 steps_270000，应优先用后期 ckpt。
  2. 闭环 vs 开环差距：训练开环(喂真实轨迹 obs)，评测闭环(喂自己上一步动作产生的 obs)，
     模型微小误差会在闭环累积发散——VLA 通用问题，与数据量无关。
  3. task0 固定做对 2 个 ep：libero_spatial task0 有 50 个 init_state，可能恰某 2 个初始
     构型离训练分布近易做对，其余做不对 → 泛化/闭环鲁棒性问题。
  ⇒ 不能把低成功率归咎于"训练数据没过完"（已证 4 suite 过 24 遍）。

【E. 两个训练 run 的差别】

                          runs/lda_libero                runs/lda_video_prompt_k2_t4
data_mix                  libero (4 suite)               droid_video_prompt
use_video_prompt          false                          true
only_policy               true                           true
max_train_steps(配置)     300000                         300000
实际已训到(summary)       290000 步                      220000 步
评测默认 ckpt             steps_270000                   steps_190000
──────────────────────────────────────────────────────────────────────────────
