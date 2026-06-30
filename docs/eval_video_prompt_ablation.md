# Video Prompt Offline Evaluation

## 目标

证明 in-context video prompt 真正被模型利用，而非只是"多了视觉 token 就变好"。需要证明三个结论：

1. **correct video prompt 能提升 action prediction**（优于 no prompt）
2. **wrong / shuffled / final-frame prompt 会明显变差**（说明模型真的读了 prompt 语义和时序）
3. **K-shot scaling**：增加 support demo 数量能持续提升或趋于饱和

---

## 1. 评测模式

| 模式 | support videos 来源 | 目的 |
|---|---|---|
| `none` | 不给 support video | LDA 原始能力基线 |
| `correct` | 同 language 的其他 episode | 主方法 |
| `wrong` | 不同 language 的 episode | 证明 prompt 语义重要 |
| `shuffled` | correct prompt 但帧顺序打乱 | 证明视频时序重要 |
| `final_frame` | 每条 support demo 只重复最后一帧 | 证明完整视频比目标图像强 |

---

## 2. 核心 Gap 指标

```
Prompt Gain      = Error(none)          - Error(correct)         > 0 说明 video prompt 有用
Sensitivity Gap  = Error(wrong)         - Error(correct)         > 0 说明模型读了 prompt 语义
Temporal Gap     = Error(shuffled)      - Error(correct)         > 0 说明模型用了时序信息
Video Gap        = Error(final_frame)   - Error(correct)         > 0 说明视频比目标图像强
```

**四个 Gap 全部 > 0，才能 claim video-conditioned in-context imitation 成立。**

如果只有 `Prompt Gain > 0` 但 `Sensitivity Gap ≈ 0`，说明模型只是因为多了视觉 token 变好，不是真的利用了 prompt 语义。

---

## 3. 期望结果表

### 3.1 Prompt Mode 消融

| Prompt Mode | Action L1 ↓ | EE Position L1 ↓ | EE Rotation L1 ↓ | Gripper L1 ↓ | Action MSE ↓ |
|---|---:|---:|---:|---:|---:|
| none | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| **correct** | **0.XXX** | **0.XXX** | **0.XXX** | **0.XXX** | **0.XXX** |
| wrong | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| shuffled | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| final_frame | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |

期望排序：`correct < none < shuffled < final_frame < wrong` 或 `correct < none < wrong < shuffled < final_frame`

### 3.2 Gap 指标

| Gap | Action L1 | EE Position L1 | EE Rotation L1 | Gripper L1 |
|---|---|---|---|---|
| Prompt Gain | +0.XXX | +0.XXX | +0.XXX | +0.XXX |
| Sensitivity Gap | +0.XXX | +0.XXX | +0.XXX | +0.XXX |
| Temporal Gap | +0.XXX | +0.XXX | +0.XXX | +0.XXX |
| Video Gap | +0.XXX | +0.XXX | +0.XXX | +0.XXX |

### 3.3 K-Shot Scaling

| K (support demos) | Action L1 ↓ | EE Position L1 ↓ | EE Rotation L1 ↓ |
|---:|---:|---:|---:|
| 0 | 0.XXX | 0.XXX | 0.XXX |
| 1 | 0.XXX | 0.XXX | 0.XXX |
| 2 | 0.XXX | 0.XXX | 0.XXX |
| 4 | 0.XXX | 0.XXX | 0.XXX |
| 8 | 0.XXX | 0.XXX | 0.XXX |

期望：K=1 > K=0, K=2 > K=1 稍好, K=4 继续提升或饱和。

---

## 4. 实现架构

### 4.1 数据端：`prompt_mode` 参数

在 `VideoPromptLeRobotSingleDataset` 中新增 `prompt_mode` 参数：

```
VideoPromptLeRobotSingleDataset
  ├── prompt_mode="correct"   (默认，等同 wrong_prompt_prob=0)
  ├── prompt_mode="none"      (support_videos=None)
  ├── prompt_mode="wrong"     (强制采样不同 language 的 episode)
  ├── prompt_mode="shuffled"  (correct + _shuffle_video_frames())
  └── prompt_mode="final_frame" (correct + _make_final_frame_prompt())
```

新增两个方法：

```python
def _shuffle_video_frames(self, support_videos):
    """打乱每条 demo 的帧顺序 — 证明时序重要"""
    out = []
    for demo in support_videos:
        demo = list(demo)
        random.shuffle(demo)
        out.append(demo)
    return out

def _make_final_frame_prompt(self, support_videos):
    """每条 demo 只保留最后一帧重复 T 次 — 证明视频比图像强"""
    out = []
    for demo in support_videos:
        last_frame = demo[-1]
        out.append([last_frame for _ in range(len(demo))])
    return out
```

`prompt_mode` 在数据链路中的传播：

```
DataConfig.prompt_mode = "correct"     ← 默认值，训练不受影响
       ↓
make_LeRobotSingleDataset()
       ↓
VideoPromptLeRobotSingleDataset(prompt_mode=...)
       ↓
LeRobotMixtureDataset.__getitem__()
  ├── prompt_mode="none"     → result["support_videos"] = None
  ├── prompt_mode="correct"  → _sample_support_episodes() + _load_support_video()
  ├── prompt_mode="wrong"    → _sample_support_episodes() (强制 wrong)
  ├── prompt_mode="shuffled" → correct + _shuffle_video_frames()
  └── prompt_mode="final_frame" → correct + _make_final_frame_prompt()
       ↓
collate_fn (返回 list of dict)
       ↓
QwenMMDiT.predict_action()
  ├── support_videos=None → support_imgs=None → 跳过 encode_support_prompt()
  └── support_videos=[...] → support_imgs → encode_support_prompt() → concat to vl_embs
```

### 4.2 模型端：`support_videos=None` 的处理

模型已有 `None` 检查：

```python
# QwenMMDiT.forward() / predict_action()
support_imgs = None
if support_videos is not None and support_videos[0] is not None:
    support_imgs = self._support_videos_to_numpy(support_videos)

# MMDiT_ActionHeader._concat_support_to_vl_embs()
if support_imgs is None:
    return vl_embs, encoder_attention_mask  # 直接返回，不拼接
```

因此 `prompt_mode="none"` 天然兼容，无需额外修改模型代码。

---

## 5. 评测脚本

### 5.1 消融实验

```bash
bash scripts/eval_scripts/eval_LDA_video_prompt_ablation.sh
```

脚本自动循环 5 种 prompt mode，每种模式跑相同的 trajectory 集合，最后运行 summarizer。

编辑脚本中的配置：
```bash
CHECKPOINT=/path/to/your/checkpoint.pt
CONFIG_YAML=lda/config/training/LDA_pretrain.yaml
DATA_ROOT=/path/to/your/lerobot_data_root
DATA_MIX=droid_video_prompt
```

### 5.2 K-Shot Scaling

```bash
bash scripts/eval_scripts/eval_LDA_video_prompt_kshot.sh
```

自动跑 K=0 (none) + K=1,2,4,8 (correct)。

### 5.3 结果汇总

```bash
python lda/eval/summarize_video_prompt_eval.py --eval_dir ./eval_outputs --k 2
```

输出：
- 消融表（5 种 mode 的 L1 指标）
- Gap 指标（Prompt Gain / Sensitivity Gap / Temporal Gap / Video Gap）
- K-shot scaling 表
- Markdown 表格 + JSON 汇总

---

## 6. 文件清单

| 文件 | 类型 | 说明 |
|---|---|---|
| `lda/dataloader/gr00t_lerobot/datasets.py` | 修改 | `VideoPromptLeRobotSingleDataset` 新增 `prompt_mode` + `_shuffle_video_frames()` + `_make_final_frame_prompt()`；`LeRobotMixtureDataset.__getitem__()` 支持 prompt_mode |
| `lda/dataloader/gr00t_lerobot/data_config.py` | 修改 | 所有 VideoPrompt DataConfig 新增 `prompt_mode = "correct"` |
| `lda/dataloader/lerobot_datasets.py` | 修改 | `make_LeRobotSingleDataset()` 传递 `prompt_mode` |
| `lda/eval/eval_LDA_video_prompt.py` | 新建 | 离线评测脚本，支持 prompt_mode + K-shot |
| `lda/eval/summarize_video_prompt_eval.py` | 新建 | 结果汇总 + Gap 计算 + Markdown 输出 |
| `scripts/eval_scripts/eval_LDA_video_prompt_ablation.sh` | 新建 | 5 种 prompt mode 消融 bash 脚本 |
| `scripts/eval_scripts/eval_LDA_video_prompt_kshot.sh` | 新建 | K-shot scaling bash 脚本 |

---

## 7. 评测指标说明

### 7.1 Action L1

逐 timestep 计算预测动作与 GT 动作的 L1 距离，按维度组拆开：

| 维度组 | 说明 | DroidFranka slice |
|---|---|---|
| `ee_position_l1` | 末端执行器位置 L1 | [0:3] |
| `ee_rotation_l1` | 末端执行器旋转 angular L1 | [3:6] |
| `gripper_l1` | 夹爪宽度 L1 | [6:7] |
| `total_action_l1` | 全维度 L1 | [:7] |
| `total_action_mse` | 全维度 MSE | [:7] |

旋转维度使用 angular difference（`[-pi, pi)`），而非直接 L1，避免角度环绕误差。

### 7.2 Unnormalize

预测动作经过 `dataset.transforms.unapply()` 反归一化后再计算 L1，确保指标在原始尺度上可解释。

### 7.3 数据集类型自动检测

脚本自动根据 embodiment tag 选择 unapply 方式：

| Tag | Action 结构 | Unapply Keys |
|---|---|---|
| DroidFranka | pos(3)+rot(3)+grip(1)=7 | `action.eef_position`, `action.eef_rotation`, `action.gripper` |
| Bimanual Franka | left(7)+right(7)=14 | `action.left_eef_position`, ..., `action.right_gripper` |
| RoboCasa | arm+hand+waist=29 | `action.left_arm`, ..., `action.waist` |
| No gripper | pos+rot+hand=24 | `action.left_eef_position`, ..., `action.right_mano_hand_param` |

---

## 8. 向后兼容性

- `prompt_mode` 默认 `"correct"`，等同于 `wrong_prompt_prob=0.0`（训练时的默认行为）
- 训练代码完全不受影响，只有 eval 脚本才设置其他 prompt_mode
- 模型代码无需修改，`support_videos=None` 已天然支持

---

## 9. 论文写作建议

如果你要在论文/报告中写这个实验，建议这样表述：

> We evaluate whether video prompts are actually used by the model through counterfactual prompt ablations. For each query, we compare correct support videos, no support videos, wrong-task support videos, temporally shuffled support videos, and final-frame-only support. A consistent improvement of correct video prompts over all counterfactual conditions indicates that the model leverages demonstration video context rather than merely benefiting from extra visual tokens.

中文版：

> 我们不是只比较有没有 video prompt，而是用错误 prompt、打乱时序 prompt、final-frame prompt 做反事实消融。如果 correct video prompt 明显更好，才能说明模型真的利用了 demonstration video 的任务和动态信息，而非只是因为多了视觉 token 变好。

---

## 10. 进一步实验方向

当前评测是 offline open-loop action prediction。后续可扩展：

1. **Held-out language/task 评测**：训练时完全不出现某些 language，测试时给新 language 的 K 条 support videos，证明 in-context adaptation
2. **Same language, different prompt video**：同一个 query 采多组同 language support，看输出是否稳定
3. **Wrong language but visually similar**：如果 wrong language 但视觉相似也有效，说明 prompt 主要提供视觉目标
4. **Closed-loop sim eval**：在 RoboCasa 或其他仿真环境中跑 closed-loop success rate，5 种 prompt mode 对比
5. **Future latent loss 分析**：如果开启了 `use_img_denoise`，对比 `obs_loss` 在不同 prompt mode 下的差异
