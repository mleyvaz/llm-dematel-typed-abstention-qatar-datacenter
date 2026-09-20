#!/usr/bin/env python
"""Naive-vs-gated DEMATEL cause-effect scatter, one 2-panel figure per context."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def make(ctx):
    with open(f"data/v2/results_dematel_{ctx}.json", encoding="utf-8") as f:
        R = json.load(f)
    crits = R["criteria"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, mode, title in [(axes[0], "naive", "Naive (gate = OFF)"), (axes[1], "gated", "Gated (gate = ON)")]:
        prom = R[f"prominence_{mode}"]
        net = R[f"net_{mode}"]
        color = "tab:red" if mode == "naive" else "tab:blue"
        ax.scatter(prom, net, color=color, zorder=3)
        for i, c in enumerate(crits):
            ax.annotate(c, (prom[i], net[i]), textcoords="offset points", xytext=(6, 6), fontsize=9)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("Prominence (D + R)")
        ax.set_ylabel("Net cause / effect (D − R)")
        ax.set_title(title)
        ax.text(0.97, 0.95, "cause ↑", transform=ax.transAxes, ha="right", va="top", color="green", fontsize=8)
        ax.text(0.97, 0.05, "effect ↓", transform=ax.transAxes, ha="right", va="bottom", color="firebrick", fontsize=8)
    fig.suptitle(f"DEMATEL cause-effect diagram — {ctx.capitalize()} context")
    fig.tight_layout()
    out = f"figure_causeeffect_{ctx}.png"
    fig.savefig(out, dpi=150)
    print("saved", out)

if __name__ == "__main__":
    ctx = sys.argv[1] if len(sys.argv) > 1 else "qatar"
    make(ctx)
