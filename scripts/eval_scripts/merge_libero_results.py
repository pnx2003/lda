#!/usr/bin/env python3
# 合并多个 worker 的 rollout_results.json, 求全局成功率。
# 用法: python merge_libero_results.py <output_root>
import json
import sys
import glob
from collections import defaultdict

if len(sys.argv) < 2:
    print("usage: merge_libero_results.py <output_root>")
    sys.exit(1)

OUT = sys.argv[1]
eps = []
for f in sorted(glob.glob(f"{OUT}/worker*/rollout_results.json")):
    try:
        eps += json.load(open(f))["episodes"]
    except Exception as e:
        print(f"[warn] skip {f}: {e}")

agg = defaultdict(lambda: [0, 0])  # (suite, mode) -> [succ, total]
for e in eps:
    k = (e["suite"], e["prompt_mode"])
    agg[k][1] += 1
    agg[k][0] += int(e["success"])

print("\n" + "=" * 60)
print(f"{'suite':<16}{'mode':<10}{'succ':>6}{'total':>7}{'rate':>8}")
print("-" * 60)
for (s, m), (ok, tot) in sorted(agg.items()):
    print(f"{s:<16}{m:<10}{ok:>6}{tot:>7}{100*ok/tot:>7.1f}%")
print("=" * 60)

merged = {
    "summary": {f"{s}|{m}": {"success": o, "total": t, "rate": o / t}
                for (s, m), (o, t) in agg.items()},
    "episodes": eps,
}
with open(f"{OUT}/rollout_results_merged.json", "w") as f:
    json.dump(merged, f, indent=2)
print(f"total episodes: {len(eps)}")
print(f"saved -> {OUT}/rollout_results_merged.json")
