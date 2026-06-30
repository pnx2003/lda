# Video Prompt Evaluation Plan

## Goal
Prove that the in-context video prompt mechanism works by showing:
1. Correct prompt improves action prediction over no prompt
2. Wrong/shuffled/final-frame prompts degrade performance
3. K-shot scaling shows increasing benefit with more support demos

## Implementation Steps

### Step 1: Add `prompt_mode` to `VideoPromptLeRobotSingleDataset` (datasets.py)
- Add prompt_mode param: correct/none/wrong/shuffled/final_frame
- Modify _sample_support_episodes() to dispatch by prompt_mode
- Add _shuffle_video_frames() and _make_final_frame_prompt()
- Modify __getitem__() to apply transforms and set support_videos=None for none mode

### Step 2: Update LeRobotMixtureDataset.__getitem__() (datasets.py)
- Pass prompt_mode through in the video prompt section
- Skip support video loading for none mode
- Apply transforms for shuffled/final_frame modes

### Step 3: Add prompt_mode to DataConfig and make_LeRobotSingleDataset() 
- data_config.py: Add prompt_mode field to video prompt configs
- lerobot_datasets.py: Pass prompt_mode through to dataset

### Step 4: Create eval_LDA_video_prompt.py (lda/eval/)
- Load checkpoint, create dataset with prompt_mode
- Iterate trajectories, run predict_action(), compute L1 losses
- Save results as JSON per (traj_id, prompt_mode)

### Step 5: Create summarize_video_prompt_eval.py (lda/eval/)
- Aggregate per-mode results into summary table
- Compute gap metrics (Prompt Gain, Sensitivity Gap, etc.)

### Step 6: Create eval_LDA_video_prompt_ablation.sh (scripts/eval_scripts/)
- Loop over prompt modes, run eval, summarize

### Step 7: Create eval_LDA_video_prompt_kshot.sh (scripts/eval_scripts/)
- Loop over K=0,1,2,4,8 with correct prompt mode

## Files to Create/Modify
- MODIFY: lda/dataloader/gr00t_lerobot/datasets.py
- MODIFY: lda/dataloader/gr00t_lerobot/data_config.py
- MODIFY: lda/dataloader/lerobot_datasets.py
- CREATE: lda/eval/eval_LDA_video_prompt.py
- CREATE: lda/eval/summarize_video_prompt_eval.py
- CREATE: scripts/eval_scripts/eval_LDA_video_prompt_ablation.sh
- CREATE: scripts/eval_scripts/eval_LDA_video_prompt_kshot.sh
