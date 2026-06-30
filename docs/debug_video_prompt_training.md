# Debug: LDA Video Prompt Training 维度不匹配问题

## 运行命令

```bash
bash /cpfs01/pnx/wordmodels/incontext/LDA-1B/scripts/run_scripts/run_lerobot_datasets_LDA_video_prompt.sh
```

---

## 报错4: `AssertionError: Hidden dim mismatch: support_tokens.shape[-1]=1536 vs vl_embs.shape[-1]=2560`

### 错误位置

`MMDiT_ActionHeader.py:663` — `_concat_support_to_vl_embs` 中的 assert

### 根因

`vl_embs` 来自 Qwen VL 的 `last_hidden_state`，维度为 Qwen hidden_size = **2560**。

`encode_support_prompt` 中 `support_prompt_projector` 把 dinov3 特征 (384) 投影到 `self.input_embedding_dim = 1536`（DiT-L 的 `input_embedding_dim`），而非 `self.cross_attention_dim = 2560`（vl_embs 的实际维度）。

拼接时：`support_tokens.shape[-1]=1536 ≠ vl_embs.shape[-1]=2560` → assert 失败。

维度流向：

```
Qwen VL → last_hidden [B, seq, 2560] ──────────────────→ vl_embs (cross_attention_dim=2560)
dinov3-s → features [B, N, 384] → projector(384→1536) → support_tokens (input_embedding_dim=1536)
                                                             ↓
                                              cat(vl_embs, support_tokens) → ❌ 1536 ≠ 2560
```

### 修复

1. **`MMDiT_ActionHeader.py` `__init__`**: 将 `support_prompt_projector` 从懒创建改为正式初始化，投影目标从 `input_embedding_dim`(1536) 改为 `cross_attention_dim`(2560)

```python
# __init__ 中新增：
if getattr(config, "use_video_prompt", False):
    self.support_prompt_projector = nn.Linear(num_chans, self.cross_attention_dim)
else:
    self.support_prompt_projector = None
```

2. **`encode_support_prompt`**: 改用 `cross_attention_dim` 而非 `input_embedding_dim`

```python
# 修改前：
if x.shape[-1] != self.input_embedding_dim:
    if not hasattr(self, 'support_prompt_projector'):
        self.support_prompt_projector = nn.Linear(x.shape[-1], self.input_embedding_dim)
    x = self.support_prompt_projector(x)

# 修改后：
if x.shape[-1] != self.cross_attention_dim:
    assert self.support_prompt_projector is not None, (...)
    x = self.support_prompt_projector(x)
```

修复后维度流向：

```
Qwen VL → last_hidden [B, seq, 2560] ──────────────────→ vl_embs (cross_attention_dim=2560)
dinov3-s → features [B, N, 384] → projector(384→2560) → support_tokens (cross_attention_dim=2560)
                                                             ↓
                                              cat(vl_embs, support_tokens) → ✅ 2560 = 2560
```

---

## 报错1: `RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [16, 12] but got: [16, 58]`

### 错误位置

`MMDiT_ActionHeader.py:74` — `CategorySpecificLinear.forward` 中的 `torch.bmm(x, selected_W)`

### 根因

脚本中 `state_dim=58, action_dim=138` 是 **RoboCasa GR1 机器人**的参数，但实际运行的 demo 数据集是 **Franka 机器人**：

| Demo 数据集 state_keys | 每键维度 | 合计 |
|---|---|---|
| `state.eef_position` | 3 | |
| `state.eef_rotation` | 3 | |
| `state.gripper_width` | 6 | |
| **总计** | | **12** |

| Demo 数据集 action_keys | 每键维度 | 合计 |
|---|---|---|
| `action.eef_position` | 3 | |
| `action.eef_rotation` | 3 | |
| `action.gripper_width` | 6 | |
| **总计** | | **12** |

`CategorySpecificLinear` 的权重 `W` 形状为 `(num_categories, input_dim, hidden_dim)`。模型以 `state_dim=58` 初始化，`W` 的 `input_dim=58`；但实际 state 数据只有 12 维，`torch.bmm(x, selected_W)` 中 `x` 最后一维=12，`selected_W` 的 dim1=58，导致不匹配。

### 修复

```bash
# scripts/run_scripts/run_lerobot_datasets_LDA_video_prompt.sh
state_dim=12
action_dim=12
```

---

## 报错2: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (3216x768 and 1152x1536)`

### 错误位置

`MMDiT_ActionHeader.py:897` — `obs_merger(torch.cat([curr_obs, noisy_next_obs], dim=-1))`

### 根因

`obs_merger` 的输入维度由 `observation_indices` 决定的观测帧数 + 未来观测帧数决定。

**维度推导（dinov3-s, hidden_size=384）：**

```python
# curr_obs 经过 dinov3 编码 + rearrange 后：
#   curr_obs shape = (B, V*n, c*T_curr)
#   其中 T_curr = len(observation_indices)
#   c = num_chans = 384 (dinov3-s hidden_size)

# noisy_next_obs shape = (B, V*n, c)  # 1帧未来观测

# cat 后最后一维 = c*T_curr + c = c*(T_curr + 1)
```

**预训练 checkpoint 分析：**

- `obs_merger.weight` shape = `(1536, 1152)` → 输入维度 1152
- 1152 = 384 × 3 → `T_curr + 1 = 3` → `T_curr = 2`
- 预训练时 GR1 的 `observation_indices = [-5, 0]`（2帧）

**DemoDataConfig 的问题：**

- `DemoDataConfig.observation_indices = [0]`（仅1帧）
- 导致 `T_curr = 1`，cat 后维度 = 384 × 2 = 768
- `obs_merger` 权重期望 1152，实际输入 768 → 不匹配

### 修复

```python
# lda/dataloader/gr00t_lerobot/data_config.py — DemoDataConfig
observation_indices = [-1, 0]  # 从 [0] 改为 [-1, 0]，提供2帧观测
```

修复后维度验证：
- `T_curr = 2`，cat 后 = 384 × (2+1) = 1152
- `obs_merger = nn.Linear(1152, 1536)` ✓ 与预训练权重 `(1536, 1152)` 匹配

---

## 报错3: `UnboundLocalError: local variable 'output_dict' referenced before assignment`

### 错误位置

`train_LDA.py:469` — `finally` 块中的 `del output_dict`

### 根因

```python
def _train_step(self, batch_vla, batch_vlm=None):
    try:
        with self.accelerator.accumulate(self.model):
            ...
            output_dict = self.model.forward(batch_vla)  # 可能抛异常
            ...
    finally:
        del output_dict  # 如果 forward 抛异常，output_dict 未定义
```

当 `forward()` 抛出异常（如报错1/2）时，`output_dict` 从未被赋值，`finally` 块中的 `del output_dict` 引发 `UnboundLocalError`，掩盖了原始异常。

### 修复

```python
def _train_step(self, batch_vla, batch_vlm=None):
    output_dict = None  # ← 新增：预初始化
    try:
        ...
    finally:
        if output_dict is not None:  # ← 修改：条件删除
            del output_dict
```

---

## 预训练 Checkpoint 兼容性分析

预训练 checkpoint: `/cpfs01/Embodied/checkpoints/LDA-1B/LDA-pretrain/checkpoints/LDA-pretrain.pt`

加载方式: `load_state_dict(checkpoint, strict=False)` + compatible padding（`trainer_tools.py:260-304`）

### 维度对照表

| 层 | 新模型 shape | 预训练 shape | 状态 |
|---|---|---|---|
| `obs_merger.weight` | `(1536, 1152)` | `(1536, 1152)` | ✅ 完全匹配 |
| `obs_projector.weight` | `(384, 2560)` | `(384, 2560)` | ✅ 完全匹配 |
| `vision_encoder.*` | dinov3-s | dinov3-s | ✅ 完全匹配 |
| `model.blocks.*` (DiT-L) | 16层, inner_dim=1536 | 16层, inner_dim=1536 | ✅ 完全匹配 |
| `state_encoder.*` | `(32, 12, 2560)` 等 | **无** | 🆕 随机初始化（预训练时 state_dim=None） |
| `action_encoder.W1.W` | `(32, 12, 1536)` | `(32, 138, 1536)` | ❌ dim1 不匹配 → 跳过，随机初始化 |
| `action_encoder.W2.W` | `(32, 3072, 1536)` | `(32, 3072, 1536)` | ✅ 匹配 |
| `action_encoder.W3.W` | `(32, 1536, 1536)` | `(32, 1536, 1536)` | ✅ 匹配 |
| `action_decoder.layer1.W` | `(32, 2560, 2560)` | `(32, 2560, 2560)` | ✅ 匹配 |
| `action_decoder.layer2.W` | `(32, 2560, 12)` | `(32, 2560, 138)` | ❌ dim2 不匹配 → 跳过，随机初始化 |

### 兼容加载逻辑

`trainer_tools.py:260-304` 中的 `load_pretrained_backbones` 对 size mismatch 的处理：

1. **dim=0 不匹配**（如 `num_categories` 不同）→ padding 或 truncation
2. **其他维度不匹配** → **跳过该权重**，使用模型初始化的随机权重

因此 `action_encoder.W1` 和 `action_decoder.layer2` 因 `action_dim` 不同（12 vs 138）被跳过，这些层需要在新数据上重新训练——这是预期行为。

### 预训练时实际参数（从 checkpoint 反推）

| 参数 | 预训练 yaml 值 | checkpoint 实际值 | 说明 |
|---|---|---|---|
| `action_model_type` | DiT-B | **DiT-L** | 命令行覆盖 |
| `hidden_size` | 2560 | 2560 | 一致 |
| `output_dim` | 2560 | 2560 | 一致 |
| `num_layers` | 8 | **16** | 命令行覆盖 |
| `obs_horizon` | 1 | **2** | 命令行覆盖 |
| `action_dim` | 29 | **138** | 命令行覆盖 |
| `state_dim` | 58 | **None** | 未使用 |
| `max_num_embodiments` | 未设 | **32** | 命令行覆盖 |
| `vision_encoder_size` | s | s | 一致 |

> ⚠️ yaml 配置是旧版本，实际预训练时通过命令行覆盖了很多参数。新脚本中需要确保关键参数（DiT-L, num_layers=16, obs_horizon=2, hidden_size=2560, output_dim=2560）与预训练一致。

---

## Video Prompt 完整处理链路分析

### 整体架构

```
数据端                          模型端
┌─────────────────────┐         ┌──────────────────────────────┐
│ VideoPromptDataset   │         │ QwenMMDiT.forward()          │
│ 采样 K 个 support    │ ──────► │  _support_videos_to_numpy()  │
│ episode, 每个 T 帧   │         │      ↓                       │
│ → support_videos     │         │  action_model.forward()      │
│ [B][K][T][V] PIL     │         │      ↓                       │
└─────────────────────┘         │  encode_support_prompt()     │
                                │      ↓                       │
                                │  _concat_support_to_vl_embs()│
                                │  → support tokens 拼到 vl_embs│
                                └──────────────────────────────┘
```

### 1. 数据端：support_videos 的构建

**入口：`VideoPromptLeRobotSingleDataset.__getitem__`**（datasets.py:1779）

```python
def __getitem__(self, index):
    query_ep, _ = self.all_steps[index]
    sample = super().__getitem__(index)     # 普通数据（image, action, state...）
    query_lang = sample["lang"]

    # 从同语言的其他 episode 采样 K 个 support episode
    support_eps, prompt_is_correct = self._sample_support_episodes(
        query_ep=query_ep, query_lang=query_lang
    )

    # 每个 support episode 加载 T 帧视频
    support_videos = [
        self._load_support_video(ep)
        for ep in support_eps
    ]

    sample["support_videos"] = support_videos
    return sample
```

**`_sample_support_episodes`**（datasets.py:1735）：

```python
def _sample_support_episodes(self, query_ep, query_lang):
    use_wrong = random.random() < self.wrong_prompt_prob

    if use_wrong:
        # 采样不同语言的 episode（负样本）
        other_langs = [l for l in self.language_to_episodes if l != query_lang]
        candidates = self.language_to_episodes[random.choice(other_langs)]
    else:
        # 采样同语言的其他 episode
        candidates = self.language_to_episodes.get(query_lang, [])

    candidates = [ep for ep in candidates if ep != query_ep]
    return random.choices(candidates, k=self.num_support_demos), not use_wrong
```

**`_load_support_video`**（datasets.py:1757）：

```python
def _load_support_video(self, ep_id):
    steps = sorted(self.episode_to_steps[ep_id])

    if len(steps) >= self.num_support_frames:
        # 均匀采样 T 帧
        idxs = np.linspace(0, len(steps) - 1, self.num_support_frames).astype(int)
        sampled_steps = [steps[i] for i in idxs]
    else:
        # 不够就重复最后一帧
        sampled_steps = steps + [steps[-1]] * (self.num_support_frames - len(steps))

    video = []
    for base_index in sampled_steps:
        data = self.get_step_data(ep_id, base_index)
        views = []
        for video_key in self.modality_keys["video"]:
            img = Image.fromarray(data[video_key][0]).resize((224, 224))
            views.append(img)
        video.append(views)   # [T][V] PIL.Images

    return video
```

**support_videos 最终结构：**

```
List[B][K][T][V] of PIL.Image
    B = batch size
    K = num_support_demos（脚本设为2，debug模式1）
    T = num_support_frames（脚本设为4，debug模式2）
    V = num_views（1个相机）
```

所以 **不是只选两帧**，而是选 K 个 demo episode，每个 episode 均匀采样 T 帧。脚本默认 K=2, T=4，即 2 个 demo × 4 帧 = 8 帧 support 视频信息。

### 2. 模型端：support tokens 的生成与拼接

**`QwenMMDiT._support_videos_to_numpy`**（QwenMMDiT.py:74）：

```
PIL Images → np.ndarray [B, K, T, V, C, H, W]
```

**`FlowmatchingActionHead.encode_support_prompt`**（MMDiT_ActionHeader.py:583）：

```python
def encode_support_prompt(self, support_imgs):
    # support_imgs: [B, K, T, V, C, H, W]

    B, K, T, V, C, H, W = support_imgs.shape

    # 合并 K*T 到 batch 维度
    x = support_imgs.reshape(B * K * T, V, C, H, W)
    x = x.unsqueeze(2)  # [B*K*T, V, 1, C, H, W]  ← 每帧单独编码

    # 复用 obs 编码管线：transform + dinov3 vision encoder
    x = self.transform_obs(x, B2, V2, T2)
    x = self.encode_future_img(x)

    # dinov3: rearrange to [B, K*T*V*n, c]
    #   n = patch tokens 数量 (196 + cls + 4 registers = 201)
    #   c = 384 (dinov3-s hidden_size)
    x = rearrange(x, "(b kt v) n c -> b (kt v n) c", b=B, kt=K*T)
    # → shape: [B, K*T*V*n, 384]

    # 投影到 vl_embs 的维度 (1536)
    if x.shape[-1] != self.input_embedding_dim:
        x = self.support_prompt_projector(x)  # Linear(384, 1536)

    # 限制 prompt token 长度
    if x.shape[1] > self.max_prompt_tokens:
        ids = torch.linspace(0, x.shape[1]-1, steps=max_prompt_tokens).long()
        x = x[:, ids]

    return x   # [B, N_prompt, 1536]
```

**prompt token 数量计算（默认参数 K=2, T=4, V=1, n=201）：**

```
K * T * V * n = 2 * 4 * 1 * 201 = 1608 tokens
```

脚本设 `max_prompt_tokens=512`（debug模式128），所以 1608 会被均匀下采样到 512 个 token。

**`_concat_support_to_vl_embs`**（MMDiT_ActionHeader.py:647）：

```python
def _concat_support_to_vl_embs(self, vl_embs, support_imgs, encoder_attention_mask):
    if support_imgs is None:
        return vl_embs, encoder_attention_mask

    support_tokens = self.encode_support_prompt(support_imgs)  # [B, 512, 1536]

    # 直接拼接到 VLM 的文本 token 序列后面
    vl_embs = torch.cat([vl_embs, support_tokens], dim=1)
    # attention_mask 也相应拼接

    return vl_embs, encoder_attention_mask
```

### 3. ✅ 已修复：VideoPromptLeRobotSingleDataset 未被使用的问题

**原问题：** `VideoPromptLeRobotSingleDataset` 虽然已定义（datasets.py:1687），但存在两个断裂：

1. `make_LeRobotSingleDataset`（lerobot_datasets.py:84）**没有根据 `use_video_prompt` 切换到此类**，始终创建普通的 `LeRobotSingleDataset`
2. `LeRobotMixtureDataset.__getitem__` **不代理**到子 dataset 的 `__getitem__`，而是自己手动从 `dataset.get_step_data()` 构建数据，所以即使创建了 `VideoPromptLeRobotSingleDataset`，其 `__getitem__` 中的 video prompt 逻辑也不会执行

**修复1：** `lerobot_datasets.py` — `make_LeRobotSingleDataset` 增加 video prompt 分支：

```python
elif hasattr(data_config, "use_video_prompt") and data_config.use_video_prompt:
    num_support_demos = getattr(data_config, "num_support_demos", 2)
    num_support_frames = getattr(data_config, "num_support_frames", 4)
    wrong_prompt_prob = getattr(data_config, "wrong_prompt_prob", 0.0)
    return VideoPromptLeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
        img_interval=img_interval,
        history_action_indices=history_action_indices,
        CoT_prompt=CoT_prompt,
        num_support_demos=num_support_demos,
        num_support_frames=num_support_frames,
        wrong_prompt_prob=wrong_prompt_prob,
    )
```

**修复2：** `datasets.py` — `LeRobotMixtureDataset.__getitem__` 返回前添加 video prompt 逻辑：

```python
# 在构建 result dict 之后、return 之前：
if isinstance(dataset, VideoPromptLeRobotSingleDataset):
    query_ep = trajectory_id
    query_lang = str(language)
    support_eps, prompt_is_correct = dataset._sample_support_episodes(
        query_ep=query_ep, query_lang=query_lang,
    )
    support_videos = [dataset._load_support_video(ep) for ep in support_eps]
    result["support_videos"] = support_videos
    result["support_episode_ids"] = support_eps
    result["query_episode_id"] = query_ep
    result["prompt_is_correct"] = prompt_is_correct

return result
```

**完整数据链路验证：**

```
数据端                                   模型端
make_LeRobotSingleDataset                 QwenMMDiT.forward()
  ↓ use_video_prompt=True                   ↓
VideoPromptLeRobotSingleDataset            examples[0]["support_videos"] 存在
  ↓                                         ↓
LeRobotMixtureDataset.__getitem__          _support_videos_to_numpy()
  ↓ isinstance(dataset, VideoPrompt...)       ↓ [B][K][T][V] PIL → np.ndarray
  ↓                                         action_model.forward(support_imgs=...)
_sample_support_episodes()                   ↓
  ↓                                         encode_support_prompt()
_load_support_video()                        ↓ dinov3 编码 → 投影 → 下采样
  ↓                                         _concat_support_to_vl_embs()
result["support_videos"]                     ↓ cat 到 vl_embs
  ↓
collate_fn (返回 list of dict, 不做合并)
```

**collate_fn 兼容性：** 训练用的是 `collate_fn(batch): return batch`（lerobot_datasets.py:19），直接返回 list of dict，不做任何合并。`support_videos`（嵌套 PIL 列表）原样保留在每个 sample dict 中，由模型端处理。

### 4. Video Prompt 信息量总结

| 参数 | 默认值 | debug模式 | 说明 |
|---|---|---|---|
| `num_support_demos` (K) | 2 | 1 | 采样的 support demo 数量 |
| `num_support_frames` (T) | 4 | 2 | 每个 demo 均匀采样的帧数 |
| `num_views` (V) | 1 | 1 | 相机视角数 |
| `max_prompt_tokens` | 512 | 128 | 投影后最大 token 数 |
| **总帧数** | **8** | **2** | K×T |
| **编码前 tokens** | **1608** | **402** | K×T×V×n (n=201) |
| **编码后 tokens** | **512** | **128** | 下采样到 max_prompt_tokens |

**不是只选两帧**——默认选 K=2 个 demo episode，每个均匀采样 T=4 帧，共 8 帧视频作为 prompt。这些帧经过 dinov3 编码后产生 ~1608 个 token，投影后下采样到 512 个，拼接到 VLM 文本 token 序列后面，作为 in-context 条件。

---

## 修改汇总

### 1. `scripts/run_scripts/run_lerobot_datasets_LDA_video_prompt.sh`

```diff
- state_dim=58
- action_dim=138
+ state_dim=12
+ action_dim=12
```

### 2. `lda/training/train_LDA.py`

```diff
  def _train_step(self, batch_vla, batch_vlm=None):
      """execute single training step"""
+     output_dict = None
      try:
          ...
      finally:
-         del output_dict
+         if output_dict is not None:
+             del output_dict
```

### 3. `lda/dataloader/gr00t_lerobot/data_config.py`

```diff
  class DemoDataConfig:
      ...
-     observation_indices = [0]
+     observation_indices = [-1, 0]
      ...
```
