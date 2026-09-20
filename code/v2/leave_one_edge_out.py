#!/usr/bin/env python
"""
Stronger test than the flip-count null control: does the gate flag edges
that are individually more influential on the DEMATEL net cause/effect
vector than a typical edge, regardless of whether removing a whole SET of
edges flips more signs than a random set of the same size?

For every one of the 42 edges, knock it out alone (single-edge knockout of
the naive matrix), recompute net cause/effect (D-R) for all 7 criteria, and
score its impact as the L2 norm of the change from the true naive D-R
vector. Compare the impact distribution of the gate's CONFLICTING edges
against all other (COHERENT) edges with a Mann-Whitney U test (small,
unequal, non-normal samples -- rank-based, not a t-test).
"""
import json, sys
import numpy as np
from scipy.stats import mannwhitneyu

def dematel(D):
    n = D.shape[0]
    s = max(D.sum(axis=1).max(), D.sum(axis=0).max())
    if s == 0:
        return np.zeros(n), np.zeros(n)
    Dn = D / s
    T = Dn @ np.linalg.inv(np.eye(n) - Dn)
    return T.sum(axis=1), T.sum(axis=0)

def main(ctx):
    with open(f"data/v2/results_dematel_{ctx}.json", encoding="utf-8") as f:
        R = json.load(f)
    CRITERIA = R["criteria"]
    n = len(CRITERIA)
    idx = {c: i for i, c in enumerate(CRITERIA)}
    D_naive = np.array(R["D_naive"])
    D_i, R_i = dematel(D_naive)
    net_naive = D_i - R_i

    withheld = set(tuple(p.split("->")) for p in R["withheld_pairs"])

    impacts = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if D_naive[i, j] == 0.0:
                continue  # no edge to knock out
            D_ko = D_naive.copy()
            D_ko[i, j] = 0.0
            Di, Ri = dematel(D_ko)
            net_ko = Di - Ri
            impact = float(np.linalg.norm(net_ko - net_naive))
            impacts[f"{CRITERIA[i]}->{CRITERIA[j]}"] = impact

    flagged = [impacts[p] for p in impacts if tuple(p.split("->")) in withheld]
    other = [impacts[p] for p in impacts if tuple(p.split("->")) not in withheld]

    u_stat, p_value = mannwhitneyu(flagged, other, alternative="greater")

    out = {
        "context": ctx,
        "n_flagged": len(flagged), "n_other": len(other),
        "flagged_impact_mean": float(np.mean(flagged)), "flagged_impact_median": float(np.median(flagged)),
        "other_impact_mean": float(np.mean(other)), "other_impact_median": float(np.median(other)),
        "mannwhitney_u": float(u_stat), "p_value_flagged_greater": float(p_value),
        "per_edge_impact": impacts,
        "flagged_edges": sorted(f"{a}->{b}" for a, b in withheld),
    }
    with open(f"data/v2/results_leaveoneedgeout_{ctx}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"=== {ctx} ===")
    print(f"flagged (n={len(flagged)}): mean impact={np.mean(flagged):.4f}, median={np.median(flagged):.4f}")
    print(f"other   (n={len(other)}): mean impact={np.mean(other):.4f}, median={np.median(other):.4f}")
    print(f"Mann-Whitney U={u_stat:.1f}, one-sided p (flagged > other) = {p_value:.4f}")

if __name__ == "__main__":
    ctx = sys.argv[1] if len(sys.argv) > 1 else "qatar"
    main(ctx)
