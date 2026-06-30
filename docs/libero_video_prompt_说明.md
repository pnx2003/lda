# Libero Video Prompt 后训练 —— 实现与机制说明

在 `LDA-1B` 上为 libero（`/cpfs01/Embodied/datasets/libero_mujoco3.3.2`）后训练加入 video prompt context，参考脚本 `scripts/run_scripts/run_lerobot_datasets_LDA_video_prompt.sh`（droid video prompt）。

---

## 1. 背景与代码检查结论

- 仓库里**原本没有** libero 的 dataloader 支持：`mixtures.py`、`data_config.py` 的 `ROBOT_TYPE_CONFIG_MAP`、`embodiment_tags.py` 的 `ROBOT_TYPE_TO_EMBODIMENT_TAG` 三个注册表都没有 libero 条目。`lda/config/training/starvla_cotrain_libero.yaml` 里引用的 `data_mix: libero_goal` 是悬空配置（未定义）。
- libero 是单臂 Franka：state=8（`x,y,z,roll,pitch,yaw,pad,gripper`），action=7（`x,y,z,roll,pitch,yaw,gripper`），2 个相机（`primary_image` / `wrist_image`）。
- **不需要改数据集文件**：libero 的 `modality.json` 没写 `original_key`，但 pydantic schema 默认值自动填成 `observation.state` / `action`，正好对上 parquet 列名（已脚本验证）。`stats_gr00t.json` 已存在且按列名存，匹配 OK。

---

## 2. 改动清单

### 2.1 `lda/dataloader/gr00t_lerobot/data_config.py`
新增 `LiberoFrankaDataConfig` + `LiberoFrankaVideoPromptDataConfig`：
- `state_keys` 用 libero 真实 subkey `x/y/z/roll/pitch/yaw/gripper`（**跳过 `pad`** → state_dim=7）。
- `action_keys` = `x/y/z/roll/pitch/yaw/gripper`（action_dim=7）。
- `video_keys` = `["video.primary_image", "video.wrist_image"]`（2 视角）。
- `observation_indices=[-2,0]`（obs_horizon=2），`future_observation_indices=[16]`，`history_action_indices=list(range(-2,0))`，`action_indices=list(range(-2,16))`，`img_interval=1`。
  - **注意 `action_indices` 必须覆盖 history + 未来窗口**：mixture 的 `__getitem__` 会把前 `len(history_action_indices)`=2 步切走当 history_action，剩下作为 action target；模型再取最后 `future_action_window_size+1`=16 步。所以 `range(-2,16)`=18 步 → 切 2 → 16 步，匹配 `action_horizon=16`。写成 `range(0,16)` 会导致切完后只剩 14 步，训练时 `torch.cat` 报维度不匹配（`Expected 16 but got 14`）。
- 所有 state/action key 用 `q99` 归一化（libero gripper 是连续值，非 binary）。
- `LiberoFrankaVideoPromptDataConfig` 继承前者，设 `use_video_prompt=True, num_support_demos=2, num_support_frames=4, wrong_prompt_prob=0.0, prompt_mode="correct"`。
- 注册到 `ROBOT_TYPE_CONFIG_MAP`：`libero_franka` / `libero_franka_video_prompt`。

### 2.2 `lda/dataloader/gr00t_lerobot/embodiment_tags.py`
`ROBOT_TYPE_TO_EMBODIMENT_TAG` 加：
```python
"libero_franka": EmbodimentTag.FRANKA,
"libero_franka_video_prompt": EmbodimentTag.FRANKA,
```

### 2.3 `lda/dataloader/gr00t_lerobot/mixtures.py`
`DATASET_NAMED_MIXTURES` 新增：
- `libero_video_prompt`：4 子集等权（spatial + object + goal + 10）。
- `libero_spatial_video_prompt` / `libero_object_video_prompt` / `libero_goal_video_prompt` / `libero_10_video_prompt`：单子集变体。

路径形如 `libero_mujoco3.3.2/libero_spatial_no_noops_1.0.0_lerobot`，相对 `data_root_dir=/cpfs01/Embodied/datasets`。

### 2.4 `lda/dataloader/gr00t_lerobot/datasets.py`（**关键修复**）
`pad_action_state_with_key`（line ~107）按 action_key **名字**查维度表。libero 的 `action.x/y/z/roll/pitch/yaw` 不在表里，被误判成 63 维 mano hand → action 被拼成 379 维 → mixture 层 `__getitem__` 报 `could not broadcast (2,379) into (2,7)`。

修复：加一个分支，libero 风格的单坐标 subkey 各为 1 维：
```python
elif action_key.rsplit(".", 1)[-1] in ("x", "y", "z", "roll", "pitch", "yaw", "pad"):
    max_length = 1
```
保守改法，只命中这些精确名字，不影响其他数据集（其他 key 含 `eef_position`/`arm` 等会先命中前面的分支）。

> 这个 bug 只在 mixture 层（训练真正用的 `LeRobotMixtureDataset.__getitem__`）触发，单数据集层不经过该 pad 函数，所以初测没发现。

### 2.5 训练脚本
`scripts/run_scripts/post_train/Libero/run_lerobot_datasets_LDA_libero_video_prompt.sh`，基于参考脚本改：
- `data_root_dir=/cpfs01/Embodied/datasets`，`data_mix=libero_video_prompt`
- `state_dim=7`，`action_dim=7`，`num_views=2`，`obs_horizon=2`
- video prompt：`use_video_prompt=true`，`num_support_demos=2`，`num_support_frames=4`，`max_prompt_tokens=512`，`wrong_prompt_prob=0.0`
- 内存优化：`only_policy=true`，`repeated_diffusion_steps=1`，`per_device_batch_size=4`，`gradient_accumulation_steps=8`
- `freeze_module_list='action_model.vision_encoder'`
- `pretrained_checkpoint` 指向 LDA 预训练 ckpt（同参考脚本）
- 其余（`DIT_TYPE=DiT-L`，`num_layers=16`，`future_obs_index=16`，lr 等）与参考脚本一致

---

## 3. 初始 checkpoint

```
/cpfs01/Embodied/checkpoints/LDA-1B/LDA-pretrain/checkpoints/LDA-pretrain.pt
```
- **存在**，14GB，和参考的 droid video prompt 脚本用的是**同一个** LDA-1B 预训练权重。
- 脚本 `pretrained_checkpoint` 指向它，从它开始后训练。

---

## 4. Video prompt 的选取逻辑

`VideoPromptLeRobotSingleDataset`（`datasets.py:1689`）：

1. 初始化时建 3 个索引：
   - `episode → steps`
   - `episode → language`（从 `meta/episodes.jsonl` 读任务字符串）
   - `language → episodes`（同任务 episode 列表）
2. 取 query sample 时，拿到 query 的 `episode_id` 和 `language`（任务描述）。
3. `_sample_support_episodes`（line 1810）：
   - `prompt_mode="correct"` + `wrong_prompt_prob=0.0`（脚本默认）→ `use_wrong=False` → 从 **`language_to_episodes[query_lang]`**（同任务 episode 池）里，**排除 query 自己**，随机抽 K=2 条。
   - 兜底：若同任务池为空，回退到全部 episode。
4. `_load_support_video`（line 1845）：每条 support demo 在该 episode 全部 step 上 `np.linspace` 均匀采 T=4 帧，每帧取 V=2 视角。
5. `prompt_mode` 其他取值（eval 用）：`none`（不给 support）、`wrong`（不同任务）、`shuffled`（帧序打乱）、`final_frame`（只重复最后一帧）。

### 选取正确性验证（libero_spatial，200 个 query，每个 K=2）
- `language_to_episodes` 索引正确：10 个任务，每任务 42–46 个 episode，语言字符串完整解析。
- **同任务匹配率 400/400 = 100%**（`mismatch=0`）。
- **无自泄漏**：query episode 从不出现在自己的 support 里。
- `prompt_is_correct=True`，跨子集 mixture 路径 support 帧真实解码（mean≈116–128，非空非零）。

---

## 5. Video prompt 参与的计算过程

框架 `QwenMMDiT`（`lda/model/framework/QwenMMDiT.py`）+ action head `MMDiT_ActionHeader`（`lda/model/modules/action_model/MMDiT_ActionHeader.py`）。

### 5.1 前向传播——全程参与

1. **collate**（`QwenMMDiT.forward`）：`support_videos` (List[B][K][T][V] PIL) → `_support_videos_to_numpy` → `support_imgs` `[B,K,T,V,3,224,224]`，按 `repeated_diffusion_steps` 复制。
2. **编码成 prompt tokens**（`encode_support_prompt`，line 590）：
   - support_imgs reshape 成 `[B*K*T, V, 1, 3, 224, 224]`，复用 `transform_obs` + `encode_future_img` → **过 dinov3 vision encoder**。
   - rearrange 成 `[B, N_prompt, D]`，过 `support_prompt_projector`（`nn.Linear`）投影到 `cross_attention_dim`（=Qwen-VL hidden size，2560）。
   - 若 token 数 > `max_prompt_tokens`(512)，`linspace` 降采样到 512。
3. **拼进条件序列**（`_concat_support_to_vl_embs`，line 657）：
   ```python
   vl_embs = torch.cat([vl_embs, support_tokens], dim=1)
   ```
   attention_mask 也拼上全 1 的 support_mask。**support tokens 成为 MMDiT 文本/条件序列的一部分**。
4. **进 MMDiT 主干**（line 925 `self.model(text_tokens=vl_embs, ...)`）：support tokens 和 Qwen-VL 的语言/视觉 hidden states 一起作为 `text_tokens` 参与 cross-attention，条件化 action_tokens 的去噪。
5. **参与 loss**：`pred_actions` 在 support-conditioned context 下预测，`policy_act_loss = F.mse_loss(pred_actions, action_velocity)`。support 是 pred_actions 的条件输入，**通过影响 pred_actions 间接进入 loss**（loss 本身不直接作用在 support tokens 上，无 support 重建 loss）。

### 5.2 反向传播——部分参与

| 模块 | 在 support 路径上 | 是否冻结 | 梯度回传 |
|---|---|---|---|
| `support_prompt_projector` (nn.Linear) | ✅ 专属 support | ❌ 不冻结 | ✅ **有梯度，被训练** |
| MMDiT 主干 (`self.model`) | ✅ support tokens 进 cross-attn | ❌ 不冻结 | ✅ **有梯度** |
| `action_decoder` | ✅ | ❌ 不冻结 | ✅ **有梯度** |
| `obs_merger` / patchifier | ✅ support 过 transform_obs | ❌ 不冻结 | ✅ 有梯度 |
| `vision_encoder` (dinov3) | ✅ support 过它编码 | ✅ **冻结** (`freeze_modules='action_model.vision_encoder'`) | ❌ 无梯度 |

**重要细节**：`encode_future_img` 里 dinov3 调用包在 `with torch.no_grad():` 里（line 535）。即使不冻结 vision_encoder，support 帧过 dinov3 这段也不回传梯度（输出 detached）。所以 vision encoder 对 support 路径本来就是"只前向编码、不训练"的冻结语义，`freeze_modules` 是显式叠加保险。

`freeze_backbones`（`trainer_tools.py:151`）按相对路径精确匹配子模块：`action_model.vision_encoder` 只冻结 dinov3 本身，`support_prompt_projector`、`obs_merger`、MMDiT 主体、`action_decoder` 都不冻结。

### 5.3 结论
- **support video 确实进前向**：编码成 prompt tokens → 拼到 vl_embs → 进 MMDiT cross-attention → 条件化 action 预测 → 影响 policy loss。
- **反向会训练**：`support_prompt_projector` + MMDiT 主干 + `action_decoder` 接收来自 policy loss 的梯度——模型**学着使用 support prompt**。
- **不训练**：dinov3 vision encoder（冻结 + no_grad），只作固定特征提取器。
- 配置（`use_video_prompt=true`、`max_prompt_tokens=512`、freeze vision_encoder）与参考的 droid video prompt 脚本完全对齐。

---

## 6. 运行

```bash
cd /cpfs01/pnx/wordmodels/incontext/LDA-1B
# 单卡 debug 先跑通
DEBUG_MODE=true bash scripts/run_scripts/post_train/Libero/run_lerobot_datasets_LDA_libero_video_prompt.sh
# 8 卡正式训练
bash scripts/run_scripts/post_train/Libero/run_lerobot_datasets_LDA_libero_video_prompt.sh
```

---

## 7. 可调项

- **num_views=2**：libero 两个相机都用了。若想单相机（更省显存，和 droid 参考脚本一致），把 `data_config.py` 里 `video_keys`/`future_video_keys` 改成只留 `primary_image`，脚本里 `num_views=1`。
- **state_dim=7（跳过 pad）**：若想保留满 8 维，`state_keys` 加回 `"state.pad"`，脚本 `state_dim=8`。
- **`wrong_prompt_prob`**：默认 0.0。后续可设 0.1–0.2 做鲁棒性/消融（训练时按此概率采样错误任务的 support）。

---

## 8. 自由选择跑哪些 libero 子集

默认 4 个子集（spatial + object + goal + 10）全跑。在 bash 脚本里用 4 个开关变量排除任意子集，**无需改 Python 代码**。

### 用法
```bash
# 默认: 4 个全跑
bash run_lerobot_datasets_LDA_libero_video_prompt.sh

# 只跑 3 个（排除 libero_10）
RUN_LIBERO_10=false bash run_lerobot_datasets_LDA_libero_video_prompt.sh

# 只跑 3 个（排除 libero_goal）
RUN_LIBERO_GOAL=false bash run_lerobot_datasets_LDA_libero_video_prompt.sh

# 只跑 spatial + 10（排除 object + goal）
RUN_LIBERO_OBJECT=false RUN_LIBERO_GOAL=false bash run_lerobot_datasets_LDA_libero_video_prompt.sh
```

开关变量（默认均为 `true`）：
| 变量 | 控制子集 |
|---|---|
| `RUN_LIBERO_SPATIAL` | libero_spatial |
| `RUN_LIBERO_OBJECT`  | libero_object |
| `RUN_LIBERO_GOAL`    | libero_goal |
| `RUN_LIBERO_10`      | libero_10 |

设为 `false` 即排除该子集。可任意组合。

### 实现机制
- bash 脚本根据 4 个开关拼出 `LIBERO_VIDEO_PROMPT_EXCLUDE` 环境变量（逗号分隔的排除列表，空 = 全跑），`export` 给训练进程。
- `mixtures.py` 末尾在 import 时读取该环境变量，动态覆盖 `DATASET_NAMED_MIXTURES["libero_video_prompt"]` 的子集列表。
- 因为是 import 时读取，**每个 accelerate worker 进程都会拿到同样的选择**（环境变量被子进程继承）。
- 环境变量未设/为空时，保持原 4 子集默认，行为完全不变。
- 兜底：若 4 个全排除，自动回退到全 4 子集（避免空训练）。

