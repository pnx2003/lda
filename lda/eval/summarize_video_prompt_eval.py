#!/usr/bin/env python3
"""
Summarize video-prompt ablation results.

Reads per-mode JSON files produced by eval_LDA_video_prompt.py and generates:
  1. Summary table with L1 metrics per prompt mode
  2. Gap metrics (Prompt Gain, Sensitivity Gap, Temporal Gap, Video Gap)
  3. K-shot scaling table (if multiple K values present)

Usage:
  python lda/eval/summarize_video_prompt_eval.py \
    --eval_dir ./eval_outputs
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np


def load_results(eval_dir):
    """Load all eval JSON files from directory.

    Returns:
        dict mapping (prompt_mode, K) -> list of result dicts
    """
    results = defaultdict(list)
    for fname in sorted(os.listdir(eval_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(eval_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        # Extract mode and K from filename: eval_{mode}_k{K}_t{T}.json
        # or from each result's prompt_mode/num_support_demos fields
        for item in data:
            mode = item.get("prompt_mode", "unknown")
            k = item.get("num_support_demos", 2)
            results[(mode, k)].append(item)
    return dict(results)


def compute_summary(results_dict):
    """Compute per-(mode, K) summary statistics.

    Returns:
        dict mapping (mode, K) -> {metric_name: (mean, std)}
    """
    summary = {}
    for key, items in results_dict.items():
        valid = [r for r in items if "error" not in r]
        if not valid:
            continue
        metric_vals = defaultdict(list)
        for r in valid:
            for metric, val in r.items():
                if metric in ("traj_id", "prompt_mode", "num_support_demos",
                              "num_support_frames", "error"):
                    continue
                if isinstance(val, (int, float)):
                    metric_vals[metric].append(val)
        stat = {}
        for metric, vals in metric_vals.items():
            stat[metric] = (float(np.mean(vals)), float(np.std(vals)))
        summary[key] = stat
    return summary


def print_ablation_table(summary, k=2):
    """Print the core ablation table for a given K."""
    modes = ["none", "correct", "wrong", "shuffled", "final_frame"]
    metrics = ["total_action_l1", "total_action_mse",
               "ee_position_l1", "ee_rotation_l1", "gripper_l1"]

    print("\n" + "=" * 80)
    print(f"  Video Prompt Ablation Results (K={k})")
    print("=" * 80)

    # Header
    header = f"{'Prompt Mode':<20}"
    for m in metrics:
        header += f"  {m:>18}"
    print(header)
    print("-" * len(header))

    # Rows
    for mode in modes:
        key = (mode, k)
        if key not in summary:
            continue
        row = f"{mode:<20}"
        for m in metrics:
            if m in summary[key]:
                mean, std = summary[key][m]
                row += f"  {mean:.4f} ± {std:.4f}   "
            else:
                row += f"  {'N/A':>18}"
        print(row)

    print("=" * 80)


def print_gap_metrics(summary, k=2):
    """Compute and print gap metrics.

    Gap > 0 means correct_prompt is better (lower loss).

    Prompt Gain     = Error(none)    - Error(correct)   > 0 good
    Sensitivity Gap = Error(wrong)   - Error(correct)   > 0 good
    Temporal Gap    = Error(shuffled)- Error(correct)    > 0 good
    Video Gap       = Error(final_frame) - Error(correct)> 0 good
    """
    correct_key = ("correct", k)
    if correct_key not in summary:
        print("Cannot compute gap metrics: 'correct' mode results missing.")
        return

    gaps = {
        "Prompt Gain":      ("none", k),
        "Sensitivity Gap":  ("wrong", k),
        "Temporal Gap":     ("shuffled", k),
        "Video Gap":        ("final_frame", k),
    }

    metrics = ["total_action_l1", "ee_position_l1", "ee_rotation_l1", "gripper_l1"]

    print("\n" + "=" * 80)
    print(f"  Gap Metrics (K={k}) — positive means correct prompt is better")
    print("=" * 80)

    header = f"{'Gap':<20}"
    for m in metrics:
        header += f"  {m:>18}"
    print(header)
    print("-" * len(header))

    for gap_name, (baseline_mode, baseline_k) in gaps.items():
        baseline_key = (baseline_mode, baseline_k)
        if baseline_key not in summary:
            continue
        row = f"{gap_name:<20}"
        for m in metrics:
            if m in summary[correct_key] and m in summary[baseline_key]:
                baseline_mean = summary[baseline_key][m][0]
                correct_mean = summary[correct_key][m][0]
                gap = baseline_mean - correct_mean
                sign = "+" if gap > 0 else ""
                row += f"  {sign}{gap:.4f}          "
            else:
                row += f"  {'N/A':>18}"
        print(row)

    print("=" * 80)


def print_kshot_table(summary):
    """Print K-shot scaling table across all K values for 'correct' mode."""
    # Find all K values that have 'correct' mode results
    k_values = sorted(set(k for (mode, k) in summary.keys() if mode == "correct"))
    if len(k_values) < 2:
        print("Not enough K values for K-shot scaling table (need >= 2).")
        return

    metrics = ["total_action_l1", "ee_position_l1", "ee_rotation_l1"]

    print("\n" + "=" * 80)
    print("  K-Shot Scaling (prompt_mode=correct)")
    print("=" * 80)

    header = f"{'K':>4}"
    for m in metrics:
        header += f"  {m:>18}"
    print(header)
    print("-" * len(header))

    for k in k_values:
        key = ("correct", k)
        if key not in summary:
            continue
        row = f"{k:>4}"
        for m in metrics:
            if m in summary[key]:
                mean, std = summary[key][m]
                row += f"  {mean:.4f} ± {std:.4f}   "
            else:
                row += f"  {'N/A':>18}"
        print(row)

    print("=" * 80)


def save_markdown(summary, output_dir, k=2):
    """Save results as a markdown table."""
    md_path = os.path.join(output_dir, "video_prompt_ablation_results.md")
    modes = ["none", "correct", "wrong", "shuffled", "final_frame"]
    metrics = ["total_action_l1", "total_action_mse",
               "ee_position_l1", "ee_rotation_l1", "gripper_l1"]

    lines = []
    lines.append("# Video Prompt Ablation Results\n")
    lines.append(f"## Ablation Table (K={k})\n")
    header = "| Prompt Mode | " + " | ".join(metrics) + " |"
    sep = "| --- | " + " | ".join(["---" for _ in metrics]) + " |"
    lines.append(header)
    lines.append(sep)

    for mode in modes:
        key = (mode, k)
        if key not in summary:
            continue
        vals = []
        for m in metrics:
            if m in summary[key]:
                mean, std = summary[key][m]
                vals.append(f"{mean:.4f} ± {std:.4f}")
            else:
                vals.append("N/A")
        lines.append(f"| {mode} | " + " | ".join(vals) + " |")

    # Gap metrics
    lines.append("\n## Gap Metrics\n")
    lines.append("| Gap | " + " | ".join(metrics) + " |")
    lines.append("| --- | " + " | ".join(["---" for _ in metrics]) + " |")

    correct_key = ("correct", k)
    gaps = {
        "Prompt Gain": ("none", k),
        "Sensitivity Gap": ("wrong", k),
        "Temporal Gap": ("shuffled", k),
        "Video Gap": ("final_frame", k),
    }
    for gap_name, (bmode, bk) in gaps.items():
        bkey = (bmode, bk)
        if bkey not in summary or correct_key not in summary:
            continue
        vals = []
        for m in metrics:
            if m in summary[correct_key] and m in summary[bkey]:
                gap = summary[bkey][m][0] - summary[correct_key][m][0]
                vals.append(f"{gap:+.4f}")
            else:
                vals.append("N/A")
        lines.append(f"| {gap_name} | " + " | ".join(vals) + " |")

    # K-shot scaling
    k_values = sorted(set(kk for (mode, kk) in summary.keys() if mode == "correct"))
    if len(k_values) >= 2:
        lines.append(f"\n## K-Shot Scaling\n")
        lines.append("| K | total_action_l1 | ee_position_l1 | ee_rotation_l1 |")
        lines.append("| --- | --- | --- | --- |")
        for kk in k_values:
            key = ("correct", kk)
            if key not in summary:
                continue
            vals = []
            for m in ["total_action_l1", "ee_position_l1", "ee_rotation_l1"]:
                if m in summary[key]:
                    mean, std = summary[key][m]
                    vals.append(f"{mean:.4f} ± {std:.4f}")
                else:
                    vals.append("N/A")
            lines.append(f"| {kk} | " + " | ".join(vals) + " |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nMarkdown results saved to {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize video prompt eval results")
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="Directory containing eval JSON files")
    parser.add_argument("--k", type=int, default=2,
                        help="K value to use for ablation table (default: 2)")
    args = parser.parse_args()

    results_dict = load_results(args.eval_dir)
    if not results_dict:
        print(f"No results found in {args.eval_dir}")
        return

    summary = compute_summary(results_dict)

    print_ablation_table(summary, k=args.k)
    print_gap_metrics(summary, k=args.k)
    print_kshot_table(summary)
    save_markdown(summary, args.eval_dir, k=args.k)

    # Also save summary as JSON
    summary_json = {}
    for key, metrics in summary.items():
        mode, k = key
        summary_json[f"{mode}_k{k}"] = {
            m: {"mean": v[0], "std": v[1]} for m, v in metrics.items()
        }
    json_path = os.path.join(args.eval_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Summary JSON saved to {json_path}")


if __name__ == "__main__":
    main()
