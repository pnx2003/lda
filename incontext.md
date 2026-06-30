下面是**第二版：DINO / world encoder prompt tokens 版**的清晰改造报告。核心原则是：

> **数据格式不变；训练时 dataloader 从当前数据集中动态采几条 trajectory 作为 support videos；ActionHead 复用 LDA 的 DINO/world encoder 把 support videos 编成 prompt tokens；这些 tokens 拼到原来的 `vl_embs` 里；loss 不变，只算 query action / query future latent。**

---

# 1. 总体架构

原始 LDA 大致是：

```text
query image + language
    ↓
Qwen-VL → vl_embs

query image / future image
    ↓
DINO/world encoder

vl_embs + current obs latent + noisy action/future latent
    ↓
MM-DiT
    ↓
query action / query future latent loss
```

你要改成：

```text
support videos
    ↓
same DINO/world encoder
    ↓
support prompt tokens

query image + language
    ↓
Qwen-VL
    ↓
vl_embs

concat(vl_embs, support prompt tokens)
    ↓
MM-DiT
    ↓
query action / query future latent loss
```

LDA 当前 `QwenUWM_MMDiT.forward()` 已经读取 `example["image"]`、`example["lang"]`、`example["action"]`、`example["future_image"]`，并把 Qwen-VL hidden states 作为 `vl_embs` 传给 action model；`FlowmatchingActionHead.forward()` 里又把 `vl_embs` 作为 `encoder_hidden_states` 传进 MM-DiT。这个正好适合把 support video tokens 拼进去。([GitHub][1])

---

# 2. 文件改动总览

你需要改这些文件：

```text
lda/dataloader/gr00t_lerobot/datasets.py
lda/dataloader/gr00t_lerobot/data_config.py
lda/dataloader/gr00t_lerobot/mixtures.py
lda/model/framework/QwenUWM_MMDiT.py
lda/model/modules/action_model/GR00T_ActionHeader_uwm.py
```

建议新增而不是覆盖：

```text
VideoPromptLeRobotSingleDataset
```

不要新建数据格式，也不要新建 video prompt loss。

---

# 3. Dataloader 修改

## 3.1 新增 dataset wrapper

在 `lda/dataloader/gr00t_lerobot/datasets.py` 里，原始 `LeRobotSingleDataset.__getitem__()` 会返回：

```python
dict(action=action, image=images, language=language)
```

这里的 image 来自当前 step 的 multi-view image，language 来自 dataset language key，action 是拼接后的 action chunk。([GitHub][2])

你新增一个子类：

```python
class VideoPromptLeRobotSingleDataset(LeRobotSingleDataset):
    """
    数据格式不变。
    每次 __getitem__ 额外从同任务/同 language 的其他 episode 采 K 条 support videos。
    """

    def __init__(
        self,
        *args,
        num_support_demos: int = 2,
        num_support_frames: int = 4,
        wrong_prompt_prob: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_support_demos = num_support_demos
        self.num_support_frames = num_support_frames
        self.wrong_prompt_prob = wrong_prompt_prob

        self.episode_to_steps = self._build_episode_to_steps()
        self.episode_to_language = self._build_episode_to_language()
        self.language_to_episodes = self._build_language_to_episodes()

    def _build_episode_to_steps(self):
        from collections import defaultdict
        episode_to_steps = defaultdict(list)
        for traj_id, base_index in self.all_steps:
            episode_to_steps[traj_id].append(base_index)
        return dict(episode_to_steps)

    def _build_episode_to_language(self):
        ep_to_lang = {}
        lang_key = self.modality_keys["language"][0]

        for traj_id, steps in self.episode_to_steps.items():
            data = self.get_step_data(traj_id, steps[0])
            lang = data[lang_key][0]
            if isinstance(lang, bytes):
                lang = lang.decode("utf-8")
            ep_to_lang[traj_id] = str(lang)

        return ep_to_lang

    def _build_language_to_episodes(self):
        from collections import defaultdict
        lang_to_eps = defaultdict(list)
        for ep, lang in self.episode_to_language.items():
            lang_to_eps[lang].append(ep)
        return dict(lang_to_eps)

    def _sample_support_episodes(self, query_ep, query_lang):
        import random

        use_wrong = random.random() < self.wrong_prompt_prob

        if use_wrong:
            other_langs = [l for l in self.language_to_episodes if l != query_lang]
            if len(other_langs) > 0:
                candidates = self.language_to_episodes[random.choice(other_langs)]
            else:
                candidates = self.language_to_episodes[query_lang]
        else:
            candidates = self.language_to_episodes[query_lang]

        candidates = [ep for ep in candidates if ep != query_ep]

        if len(candidates) == 0:
            candidates = list(self.episode_to_steps.keys())

        if len(candidates) >= self.num_support_demos:
            return random.sample(candidates, self.num_support_demos), not use_wrong

        return random.choices(candidates, k=self.num_support_demos), not use_wrong

    def _load_support_video(self, ep_id):
        import numpy as np
        from PIL import Image

        steps = sorted(self.episode_to_steps[ep_id])

        if len(steps) >= self.num_support_frames:
            idxs = np.linspace(0, len(steps) - 1, self.num_support_frames).astype(int)
            sampled_steps = [steps[i] for i in idxs]
        else:
            sampled_steps = steps + [steps[-1]] * (self.num_support_frames - len(steps))

        video = []
        for base_index in sampled_steps:
            data = self.get_step_data(ep_id, base_index)

            views = []
            for video_key in self.modality_keys["video"]:
                img = Image.fromarray(data[video_key][0]).resize((224, 224))
                views.append(img)

            video.append(views)

        return video

    def __getitem__(self, index):
        query_ep, _ = self.all_steps[index]

        sample = super().__getitem__(index)

        if "lang" not in sample:
            sample["lang"] = sample.get("language", "")

        query_lang = sample["lang"]

        support_eps, prompt_is_correct = self._sample_support_episodes(
            query_ep=query_ep,
            query_lang=query_lang,
        )

        support_videos = [
            self._load_support_video(ep)
            for ep in support_eps
        ]

        sample["support_videos"] = support_videos
        sample["support_episode_ids"] = support_eps
        sample["query_episode_id"] = query_ep
        sample["prompt_is_correct"] = prompt_is_correct

        return sample
```

输出格式：

```python
sample["support_videos"]  # List[K][T][V] of PIL.Image
```

其中：

```text
K = num_support_demos
T = num_support_frames
V = number of camera views
```

第一版建议：

```text
K = 2
T = 4
```

---

# 4. data_config 修改

在 `data_config.py` 里加一个 video-prompt 版本的 robot config。仓库当前用 `ROBOT_TYPE_CONFIG_MAP` 注册 robot type，例如 `"intern_franka": FrankaDataConfig()`。([GitHub][3])

例如：

```python
class FrankaVideoPromptDataConfig(FrankaDataConfig):
    use_video_prompt = True
    num_support_demos = 2
    num_support_frames = 4
    wrong_prompt_prob = 0.0
```

然后注册：

```python
ROBOT_TYPE_CONFIG_MAP = {
    ...
    "intern_franka_video_prompt": FrankaVideoPromptDataConfig(),
}
```

如果你用的不是 Franka，就继承你自己的原始 config：

```python
class MyRobotVideoPromptDataConfig(MyRobotDataConfig):
    use_video_prompt = True
    num_support_demos = 2
    num_support_frames = 4
    wrong_prompt_prob = 0.0
```

---

# 5. mixtures.py 修改

`mixtures.py` 里每个 mixture 是：

```python
(data_name, weight, robot_type)
```

并且 `get_dataset_mixtures()` 会根据 `data_root_dir` 和 `data_mix` 展开 dataset path。([GitHub][4])

你只需要新增一个 mixture，数据路径不变，只换 robot type：

```python
DATASET_NAMED_MIXTURES = {
    ...
    "my_video_prompt_train": [
        ("your_dataset_path_or_name", 1.0, "intern_franka_video_prompt"),
    ],
}
```

不要复制数据，不要改 LeRobot 文件结构。

---

# 6. Dataset 构造处修改

找到训练代码里实例化 `LeRobotSingleDataset` 的地方，改成：

```python
use_video_prompt = getattr(data_config, "use_video_prompt", False)

dataset_cls = (
    VideoPromptLeRobotSingleDataset
    if use_video_prompt
    else LeRobotSingleDataset
)

dataset = dataset_cls(
    dataset_path=dataset_path,
    modality_configs=modality_configs,
    embodiment_tag=embodiment_tag,
    video_backend=video_backend,
    transforms=transforms,
    data_cfg=data_cfg,

    # video prompt args
    num_support_demos=getattr(data_config, "num_support_demos", 2),
    num_support_frames=getattr(data_config, "num_support_frames", 4),
    wrong_prompt_prob=getattr(data_config, "wrong_prompt_prob", 0.0),
)
```

如果你不想动 dataset factory，也可以直接在原来的 `LeRobotSingleDataset.__getitem__()` 里硬加 support sampling；但我不推荐，因为会破坏原始训练流程。

---

# 7. Framework 修改：`QwenUWM_MMDiT.py`

目标：从 examples 里取出 `support_videos`，转成 tensor，然后传给 action model。

在 `QwenUWM_MMDiT.forward()` 里，原始代码已经有：

```python
batch_images = [example["image"] for example in examples]
instructions = [example["lang"] for example in examples]
actions = [example["action"] for example in examples]
batch_future_images = [example["future_image"] for example in examples]

curr_images = np.array(batch_images).transpose(0, 1, 4, 2, 3)
future_images = np.array(batch_future_images).transpose(0, 1, 4, 2, 3)
```

这些字段来自当前 query sample。([GitHub][1])

你新增：

```python
support_videos = (
    [example.get("support_videos", None) for example in examples]
    if "support_videos" in examples[0]
    else None
)
```

新增 helper：

```python
def _support_videos_to_numpy(self, support_videos):
    """
    support_videos: List[B][K][T][V] of PIL.Image
    return: np.ndarray [B, K, T, V, C, H, W]
    """
    import numpy as np
    from PIL import Image

    batch = []

    for sample_support in support_videos:
        demos = []

        for demo in sample_support:
            frames = []

            for frame in demo:
                views = []

                for img in frame:
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(img)

                    img = img.resize((224, 224))
                    arr = np.asarray(img)

                    # [H, W, C] -> [C, H, W]
                    arr = arr.transpose(2, 0, 1)
                    views.append(arr)

                frames.append(views)

            demos.append(frames)

        batch.append(demos)

    return np.asarray(batch)
```

在 forward 里：

```python
support_imgs = None

if support_videos is not None and support_videos[0] is not None:
    support_imgs = self._support_videos_to_numpy(support_videos)
```

当你把 `curr_images`、`future_images` 转 tensor 后，也转 support：

```python
if support_imgs is not None:
    support_imgs = torch.as_tensor(
        support_imgs,
        device=last_hidden.device,
        dtype=last_hidden.dtype,
    )
```

如果原代码对 batch 做了 repeat，例如 `n_action_samples`，support 也要 repeat：

```python
if support_imgs is not None and repeat_n > 1:
    support_imgs = support_imgs.repeat_interleave(repeat_n, dim=0)
```

然后调用 action model 时加参数：

```python
output_dict = self.action_model(
    vl_embs=last_hidden_repeated,
    actions=actions_target_repeated,
    history_actions=history_actions,
    state=state_repeated,
    future_imgs=future_images_repeated,
    curr_imgs=curr_images_repeated,
    support_imgs=support_imgs,
    embodiment_id=embodiment_ids_repeated,
    encoder_attention_mask=attention_mask_repeated,
)
```

推理函数 `predict_action()` 也要同样加 `support_videos → support_imgs → action_model.predict_action(...)`。当前 `predict_action()` 里最后调用的是：

```python
self.action_model.predict_action(last_hidden, state, curr_imgs, embodiment_ids, attention_mask)
```

你要改成带 `support_imgs`。([GitHub][1])

---

# 8. ActionHead 修改：`GR00T_ActionHeader_uwm.py`

这是第二版的核心。

当前 `FlowmatchingActionHead.forward()` 接收：

```python
def forward(
    self,
    vl_embs,
    actions,
    history_actions=None,
    state=None,
    future_imgs=None,
    curr_imgs=None,
    embodiment_id=None,
    encoder_attention_mask=None,
):
```

并在 MM-DiT 里使用：

```python
encoder_hidden_states=vl_embs
```

这就是你要拼接 support prompt tokens 的位置。([GitHub][5])

---

## 8.1 修改 forward signature

改成：

```python
def forward(
    self,
    vl_embs: torch.Tensor,
    actions: torch.Tensor,
    history_actions: torch.Tensor = None,
    state: torch.Tensor = None,
    future_imgs: torch.Tensor = None,
    curr_imgs: torch.Tensor = None,
    support_imgs: torch.Tensor = None,
    embodiment_id: torch.Tensor = None,
    encoder_attention_mask=None,
):
```

---

## 8.2 新增 support video encoder

在 `FlowmatchingActionHead` 里新增：

```python
def encode_support_prompt(self, support_imgs: torch.Tensor) -> torch.Tensor:
    """
    support_imgs:
        [B, K, T, V, C, H, W]

    return:
        support_tokens: [B, N_prompt, D]
    """
    B, K, T, V, C, H, W = support_imgs.shape

    # 合并 K 和 T，把 support video 当成更多 observation frames/views
    # [B, K, T, V, C, H, W] -> [B, K*T*V, C, H, W]
    x = support_imgs.reshape(B, K * T * V, C, H, W)

    # 复用 curr_imgs 的格式逻辑：
    # current code expects something compatible with transform_obs / image_encoder.
    # 这里转成 [B, V_total, T=1, C, H, W]
    x = x.reshape(B, K * T * V, 1, C, H, W)

    B2, V_total, T2 = x.shape[:3]

    # 复用 LDA 当前 obs transform + DINO/world encoder
    x = self.transform_obs(x, B2, V_total, T2)
    x = self.image_encoder(x)

    # dinov3 常见输出可能是 [(B*V*T), N, D]，这里统一成 [B, N_total, D]
    if x.dim() == 3:
        x = x.reshape(B, V_total * T2, x.shape[-2], x.shape[-1])
        x = x.reshape(B, -1, x.shape[-1])
    else:
        x = x.reshape(B, -1, x.shape[-1])

    # 控制 prompt token 长度，防止显存爆
    max_prompt_tokens = getattr(self.config, "max_prompt_tokens", 512)

    if x.shape[1] > max_prompt_tokens:
        ids = torch.linspace(
            0,
            x.shape[1] - 1,
            steps=max_prompt_tokens,
            device=x.device,
        ).long()
        x = x[:, ids]

    return x
```

注意：这里假设 `self.config` 可访问。如果 `FlowmatchingActionHead` 当前没有保存 action config，你需要在 `__init__` 里加：

```python
self.config = config
self.max_prompt_tokens = getattr(config, "max_prompt_tokens", 512)
```

然后上面用：

```python
max_prompt_tokens = self.max_prompt_tokens
```

---

## 8.3 在 forward 里拼接 prompt tokens

在调用 MM-DiT 之前，即：

```python
model_output = self.model(
    hidden_states=sa_embs,
    ada_cond=curr_obs,
    encoder_hidden_states=vl_embs,
    ...
)
```

之前加入：

```python
if support_imgs is not None:
    support_tokens = self.encode_support_prompt(support_imgs)

    # support tokens 角色等价于额外 context tokens
    vl_embs = torch.cat([vl_embs, support_tokens], dim=1)

    if encoder_attention_mask is not None:
        support_mask = torch.ones(
            support_tokens.shape[:2],
            dtype=encoder_attention_mask.dtype,
            device=encoder_attention_mask.device,
        )

        encoder_attention_mask = torch.cat(
            [encoder_attention_mask, support_mask],
            dim=1,
        )
```

然后原来的 MM-DiT 调用保持：

```python
model_output = self.model(
    hidden_states=sa_embs,
    ada_cond=curr_obs,
    encoder_hidden_states=vl_embs,
    timestep=action_t_discretized,
    return_all_hidden_states=False,
    obs_timestep=obs_t_discretized,
    encoder_attention_mask=encoder_attention_mask,
)
```

loss 完全不改。当前 action head 已经算：

```python
action_loss = F.mse_loss(pred_actions, velocity)

if self.use_img_denoise:
    obs_loss = F.mse_loss(next_obs_noise_pred, obs_velocity)
    loss = action_loss + obs_loss
else:
    loss = action_loss
```

你保持这个逻辑即可。([GitHub][5])

---

# 9. predict_action 也要改

当前 `FlowmatchingActionHead.predict_action()` 接收：

```python
def predict_action(
    self,
    vl_embs,
    state=None,
    curr_imgs=None,
    embodiment_id=None,
    encoder_attention_mask=None,
):
```

改成：

```python
def predict_action(
    self,
    vl_embs,
    state=None,
    curr_imgs=None,
    support_imgs=None,
    embodiment_id=None,
    encoder_attention_mask=None,
):
```

在采样循环前同样加：

```python
if support_imgs is not None:
    support_tokens = self.encode_support_prompt(support_imgs)
    vl_embs = torch.cat([vl_embs, support_tokens], dim=1)

    if encoder_attention_mask is not None:
        support_mask = torch.ones(
            support_tokens.shape[:2],
            dtype=encoder_attention_mask.dtype,
            device=encoder_attention_mask.device,
        )
        encoder_attention_mask = torch.cat([encoder_attention_mask, support_mask], dim=1)
```

然后 `QwenUWM_MMDiT.predict_action()` 调用：

```python
pred_actions = self.action_model.predict_action(
    last_hidden,
    state,
    curr_imgs,
    support_imgs,
    embodiment_ids,
    attention_mask,
)
```

更稳妥地用关键字参数：

```python
pred_actions = self.action_model.predict_action(
    vl_embs=last_hidden,
    state=state,
    curr_imgs=curr_imgs,
    support_imgs=support_imgs,
    embodiment_id=embodiment_ids,
    encoder_attention_mask=attention_mask,
)
```

---

# 10. 配置增加项

在 action model config 里加：

```yaml
framework:
  action_model:
    use_video_prompt: true
    num_support_demos: 2
    num_support_frames: 4
    max_prompt_tokens: 512
```

在 dataset config 里加：

```yaml
datasets:
  vla_data:
    data_mix: my_video_prompt_train
```

或者如果你是 Python config：

```python
use_video_prompt = True
num_support_demos = 2
num_support_frames = 4
wrong_prompt_prob = 0.0
```

---

# 11. 训练时 sample 应该长这样

```python
{
    # 原始字段
    "image": [PIL.Image, PIL.Image, ...],
    "future_image": [PIL.Image, PIL.Image, ...],
    "action": np.ndarray,
    "language": str,
    "lang": str,

    # 新增字段
    "support_videos": [
        # demo 1
        [
            # frame 1: multi-view images
            [PIL.Image, PIL.Image, ...],
            # frame 2
            [PIL.Image, PIL.Image, ...],
        ],
        # demo 2
        [
            [PIL.Image, PIL.Image, ...],
            [PIL.Image, PIL.Image, ...],
        ],
    ],

    "support_episode_ids": [...],
    "query_episode_id": ...,
    "prompt_is_correct": True,
}
```

模型内部转成：

```text
support_imgs: [B, K, T, V, C, H, W]
```

ActionHead 编码后：

```text
support_tokens: [B, N_prompt, D]
```

拼接后：

```text
vl_embs: [B, N_vlm + N_prompt, D]
```

---

# 12. 训练目标不变

不要新增：

```python
loss_prompt
loss_support_video
loss_video_reconstruction
```

你的训练仍然是：

```text
support videos + query obs + language → query action / query future latent
```

loss 只算：

```text
pred query action vs query action gt
pred query future latent vs query future latent gt
```

support video tokens 和 VLM tokens 一样，只是 context。

---

# 13. 必做 debug 检查

训练前先打印这些 shape：

```python
print("support_imgs:", support_imgs.shape)
print("support_tokens:", support_tokens.shape)
print("vl_embs before:", old_vl_embs.shape)
print("vl_embs after:", vl_embs.shape)
print("encoder_attention_mask:", encoder_attention_mask.shape)
```

应该类似：

```text
support_imgs: [B, 2, 4, V, 3, 224, 224]
support_tokens: [B, 512, D]
vl_embs before: [B, N_qwen, D]
vl_embs after: [B, N_qwen + 512, D]
encoder_attention_mask: [B, N_qwen + 512]
```

---

# 14. 第一阶段实验设置

先跑最小版本：

```text
K = 1 or 2
T = 4
max_prompt_tokens = 256 or 512
wrong_prompt_prob = 0.0
```

然后做三个验证：

```text
1. no prompt
2. correct prompt
3. wrong prompt
```

预期：

```text
correct prompt loss < no prompt loss
correct prompt loss < wrong prompt loss
```

如果三者差不多，说明模型没有真正使用 support video tokens。

---

# 15. 最终一句话总结

你选择的第二版应该这样改：

> **在 dataloader 里动态采 K 条 support videos，不改变 LeRobot 数据格式；在 `QwenUWM_MMDiT.py` 里把 `support_videos` 转成 `[B,K,T,V,C,H,W]` 并传给 action model；在 `FlowmatchingActionHead` 里复用原来的 `transform_obs + image_encoder` 把 support videos 编成 DINO/world prompt tokens，然后 concat 到 `vl_embs`，作为 MM-DiT 的额外 `encoder_hidden_states`；loss 完全不变，只训练 query action 和 query future latent。**

[1]: https://github.com/jiangranlv/LDA-1B/blob/main/lda/model/framework/QwenUWM_MMDiT.py "LDA-1B/lda/model/framework/QwenUWM_MMDiT.py at main · jiangranlv/LDA-1B · GitHub"
[2]: https://github.com/jiangranlv/LDA-1B/blob/main/lda/dataloader/gr00t_lerobot/datasets.py "LDA-1B/lda/dataloader/gr00t_lerobot/datasets.py at main · jiangranlv/LDA-1B · GitHub"
[3]: https://github.com/jiangranlv/LDA-1B/blob/main/lda/dataloader/gr00t_lerobot/data_config.py "LDA-1B/lda/dataloader/gr00t_lerobot/data_config.py at main · jiangranlv/LDA-1B · GitHub"
[4]: https://github.com/jiangranlv/LDA-1B/blob/main/lda/dataloader/gr00t_lerobot/mixtures.py "LDA-1B/lda/dataloader/gr00t_lerobot/mixtures.py at main · jiangranlv/LDA-1B · GitHub"
[5]: https://github.com/jiangranlv/LDA-1B/blob/main/lda/model/modules/action_model/GR00T_ActionHeader_uwm.py "LDA-1B/lda/model/modules/action_model/GR00T_ActionHeader_uwm.py at main · jiangranlv/LDA-1B · GitHub"
