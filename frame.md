
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                    In-context LDA 架构变更                                   │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │                                                                             │
  │  新增数据流:                                                                 │
  │                                                                             │
  │  Support Videos (K demos × T frames × V views)                              │
  │       │                                                                     │
  │       ▼                                                                     │
  │  VideoPromptLeRobotSingleDataset  ─────────────────────────────┐            │
  │       │  (动态采样同语言任务的 support videos)                    │            │
  │       ▼                                                        │            │
  │  support_videos: [B][K][T][V] of PIL.Image                     │            │
  │       │                                                        │            │
  │       ▼                                                        ▼            │
  │  QwenUWM_MMDiT._support_videos_to_numpy()    ──────────────► support_imgs   │
  │       │                                                        [B,K,T,V,C,H,W]
  │       │                                                        │            │
  │       ▼                                                        ▼            │
  │  Query: image + lang ──► Qwen3-VL ──► vl_embs   +  support_tokens          │
  │                                                              │              │
  │                                                              ▼              │
  │                                              encode_support_prompt()        │
  │                                              (复用 DINOv3 编码)              │
  │                                                              │              │
  │                                                              ▼              │
  │                                              concat(vl_embs, support_tokens) │
  │                                                              │              │
  │                                                              ▼              │
  │                                                    MM-DiT Backbone          │
  │                                                              │              │
  │                                                              ▼              │
  │                                              query action loss (不变)        │
  │                                                                             │