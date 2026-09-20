#!/usr/bin/env python
"""Real DEMATEL cause-effect diagram (prominence D+R vs net cause/effect D-R),
gated vs naive, computed from data/results_dematel.json -- the standard
DEMATEL results visualization, not decorative."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("data/results_dematel.json", encoding="utf-8") as f:
    R = json.load(f)

crits = R["criteria"]
prom_naive = R["prominence_naive"]
net_naive = R["net_naive"]
prom_gated = R["prominence_gated"]
net_gated = R["net_gated"]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=220)

for ax, prom, net, title, color in [
    (axes[0], prom_naive, net_naive, "Naive (gate = OFF)", "#B03A2E"),
    (axes[1], prom_gated, net_gated, "Gated (gate = ON)", "#1F618D"),
]:
    ax.axhline(0, color="#888888", linewidth=1, zorder=1)
    ax.scatter(prom, net, s=90, color=color, zorder=3, edgecolor="white", linewidth=1.2)
    for i, c in enumerate(crits):
        ax.annotate(c, (prom[i], net[i]), textcoords="offset points", xytext=(7, 6),
                    fontsize=10, weight="bold", color="#222222")
    ax.set_xlabel("Prominence  (D + R)", fontsize=10)
    ax.set_ylabel("Net cause / effect  (D \u2212 R)", fontsize=10)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.text(0.97, 0.96, "cause \u2191", transform=ax.transAxes, fontsize=8.5,
            color="#2E7D32", va="top", ha="right")
    ax.text(0.97, 0.06, "effect \u2193", transform=ax.transAxes, fontsize=8.5,
            color="#B03A2E", va="bottom", ha="right")
    ax.grid(alpha=0.25)

# keep comparable y-scales across panels for honest visual comparison
ymin = min(min(net_naive), min(net_gated)) * 1.25
ymax = max(max(net_naive), max(net_gated)) * 1.25
for ax in axes:
    ax.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig(r"C:\Users\HP\Documents\Qatar_Applications_2026\Paper2_Aydin_DEMATEL_DataCenter\figure1_causeeffect.png",
            bbox_inches="tight", facecolor="white")
print("saved")
