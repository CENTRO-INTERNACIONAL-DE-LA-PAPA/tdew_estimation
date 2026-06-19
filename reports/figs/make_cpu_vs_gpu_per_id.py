#!/usr/bin/env python
"""Generate the per-ID training-cost CPU-vs-GPU bar chart for the HPC report.

Derived (NOT a single merged benchmark): CPU per-ID from Job A (KHIPU, ~2.0 s/ID single
core; 11.05x speed-up at 32 cores); GPU per-ID from the 300k end-to-end train run
(879.7 s / 302 449 IDs). Apples-to-apples on a per-ID training-cost basis.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# per-ID training cost (seconds)
CPU1 = 2.007                     # Job A: 1027.66 s / 512 IDs (constant ~2.0 s/ID across sizes)
CPU32 = CPU1 / 11.05             # Job A strong scaling: 11.05x at 32 cores -> 0.182 s/ID
GPU = 879.7 / 302_449            # 300k train run: 0.00291 s/ID

labels = ["CPU — 1 core", "CPU — 32 cores", "GPU — A100 MIG"]
vals = [CPU1, CPU32, GPU]
colors = ["#9ecae1", "#3182bd", "#e6550d"]

fig, ax = plt.subplots(figsize=(7.2, 3.4))
bars = ax.barh(labels, vals, color=colors)
ax.set_xscale("log")
ax.set_xlabel("training cost per ID  (s, log scale)  —  lower is better")
ax.set_title("CPU vs GPU — per-ID training cost")
ax.invert_yaxis()

ann = {
    "CPU — 1 core": f"{CPU1:.2f} s/ID  (1×)",
    "CPU — 32 cores": f"{CPU32:.3f} s/ID  ({CPU1/CPU32:.0f}× vs 1 core)",
    "GPU — A100 MIG": f"{GPU:.4f} s/ID  ({CPU32/GPU:.0f}× vs 32-core CPU, {CPU1/GPU:.0f}× vs 1 core)",
}
for b, lab in zip(bars, labels):
    ax.text(b.get_width() * 1.15, b.get_y() + b.get_height() / 2,
            ann[lab], va="center", ha="left", fontsize=9)

ax.set_xlim(GPU / 2, CPU1 * 12)
ax.grid(axis="x", which="both", ls=":", alpha=0.4)
fig.tight_layout()
out = "gpu/cpu_vs_gpu_per_id.png"
fig.savefig(out, dpi=120)
print("wrote", out)
