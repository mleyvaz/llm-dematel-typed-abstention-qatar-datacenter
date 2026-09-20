#!/usr/bin/env python
"""
Null-control for the gate's naive-vs-gated sign-flip claim: with 8/42 (Qatar)
or 12/42 (Ireland) edges zeroed, how many sign flips would occur if those
edges were chosen uniformly at random instead of by the typed-abstention
gate? If random removal of the same number of edges produces just as many
flips just as often, the flips are not evidence the gate is doing anything
beyond deleting data. 5000 Monte Carlo draws per context (42-choose-k is too
large to enumerate exactly).
"""
import json, sys, random
import numpy as np

def dematel(D):
    n = D.shape[0]
    s = max(D.sum(axis=1).max(), D.sum(axis=0).max())
    if s == 0:
        return np.zeros(n), np.zeros(n)
    Dn = D / s
    T = Dn @ np.linalg.inv(np.eye(n) - Dn)
    return T.sum(axis=1), T.sum(axis=0)

def main(ctx, n_trials=5000, seed=0):
    with open(f"data/v2/results_dematel_{ctx}.json", encoding="utf-8") as f:
        R = json.load(f)
    CRITERIA = R["criteria"]
    n = len(CRITERIA)
    D_naive = np.array(R["D_naive"])
    D_i, R_i = dematel(D_naive)
    net_naive = D_i - R_i
    sign_naive = net_naive > 0

    all_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    k = len(R["withheld_pairs"])

    actual_flips = sum(1 for i, c in enumerate(CRITERIA)
                        if (R["net_naive"][i] > 0) != (R["net_gated"][i] > 0))

    rng = random.Random(seed)
    flip_counts = []
    for _ in range(n_trials):
        chosen = rng.sample(all_pairs, k)
        D_rand = D_naive.copy()
        for (i, j) in chosen:
            D_rand[i, j] = 0.0
        Di, Ri = dematel(D_rand)
        net_rand = Di - Ri
        sign_rand = net_rand > 0
        flips = int(np.sum(sign_rand != sign_naive))
        flip_counts.append(flips)

    flip_counts = np.array(flip_counts)
    p_value = float(np.mean(flip_counts >= actual_flips))
    out = {
        "context": ctx, "n_trials": n_trials, "k_withheld": k,
        "actual_gate_flips": actual_flips,
        "random_removal_flip_mean": float(flip_counts.mean()),
        "random_removal_flip_distribution": {str(v): int(c) for v, c in
                                              zip(*np.unique(flip_counts, return_counts=True))},
        "p_value_flips_geq_actual": p_value,
    }
    with open(f"data/v2/results_permutation_{ctx}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"=== {ctx} ===")
    print(f"actual gate flips: {actual_flips} (of {n})")
    print(f"random-removal (k={k}, n={n_trials}) mean flips: {flip_counts.mean():.2f}")
    print(f"distribution: {out['random_removal_flip_distribution']}")
    print(f"P(random flips >= actual) = {p_value:.4f}")

if __name__ == "__main__":
    ctx = sys.argv[1] if len(sys.argv) > 1 else "qatar"
    main(ctx)
