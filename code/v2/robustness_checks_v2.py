#!/usr/bin/env python
"""
Same robustness battery as code/robustness_checks.py (row-mean imputation,
exact model-resample enumeration, threshold sensitivity), generalized to
n=7 criteria and parametrized by decision context.
"""
import json, sys, itertools
import numpy as np

MODELS = ["anthropic/claude-haiku-4.5", "openai/gpt-4o-mini", "google/gemini-3.5-flash-lite"]

def dematel(D):
    n = D.shape[0]
    s = max(D.sum(axis=1).max(), D.sum(axis=0).max())
    if s == 0:
        return D.copy(), np.zeros(n), np.zeros(n)
    Dn = D / s
    T = Dn @ np.linalg.inv(np.eye(n) - Dn)
    return T, T.sum(axis=1), T.sum(axis=0)

def type_with_threshold(records, tau_c, tau_o):
    by_model = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r["score"])
    per_model = {}
    for m, scores in by_model.items():
        usable = [s for s in scores if s is not None]
        if not usable:
            per_model[m] = {"usable": False, "rep": None, "conflict": False}
            continue
        spread = max(usable) - min(usable)
        per_model[m] = {"usable": True, "rep": sum(usable) / len(usable), "conflict": spread >= tau_c}
    usable_models = [m for m, d in per_model.items() if d["usable"] and not d["conflict"]]
    any_conflict = any(d["usable"] and d["conflict"] for d in per_model.values())
    n_usable = sum(1 for d in per_model.values() if d["usable"])
    if n_usable < 2:
        return "INSUFFICIENT"
    if any_conflict:
        return "CONFLICTING"
    reps = [per_model[m]["rep"] for m in usable_models]
    return "OPPOSED" if (max(reps) - min(reps)) >= tau_o else "COHERENT"

def main(ctx):
    with open(f"data/v2/raw_dematel_responses_{ctx}.json", encoding="utf-8") as f:
        RAW = json.load(f)
    with open(f"data/v2/results_dematel_{ctx}.json", encoding="utf-8") as f:
        STAT = json.load(f)
    CRITERIA = STAT["criteria"]
    idx = {c: i for i, c in enumerate(CRITERIA)}
    n_crit = len(CRITERIA)

    cells = {}
    for r in RAW:
        cells.setdefault((r["ci"], r["cj"]), []).append(r)

    withheld = set(tuple(p.split("->")) for p in STAT["withheld_pairs"])
    D_zero = np.array(STAT["D_gated"])
    D_rowmean = D_zero.copy()
    for (ci, cj) in withheld:
        i, j = idx[ci], idx[cj]
        other_vals = [D_zero[i, k] for k in range(n_crit) if k not in (i, j)]
        D_rowmean[i, j] = sum(other_vals) / len(other_vals) if other_vals else 0.0

    _, Dg0, Rg0 = dematel(D_zero)
    _, Dg1, Rg1 = dematel(D_rowmean)
    netR_zero = Dg0 - Rg0
    netR_alt = Dg1 - Rg1

    all_resamples = list(itertools.product(MODELS, repeat=3))
    boot_netR = {c: [] for c in CRITERIA}
    for resampled_models in all_resamples:
        D_boot = np.zeros((n_crit, n_crit))
        for (ci, cj), records in cells.items():
            by_model = {}
            for r in records:
                by_model.setdefault(r["model"], []).append(r["score"])
            vals = []
            for m in resampled_models:
                scores = [s for s in by_model.get(m, []) if s is not None]
                if scores:
                    vals.append(sum(scores) / len(scores))
            if (ci, cj) in withheld:
                D_boot[idx[ci], idx[cj]] = 0.0
            else:
                D_boot[idx[ci], idx[cj]] = sum(vals) / len(vals) if vals else 0.0
        _, Db, Rb = dematel(D_boot)
        net = Db - Rb
        for i, c in enumerate(CRITERIA):
            boot_netR[c].append(float(net[i]))

    boot_ci = {c: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)),
                   "n_enumerated": len(v)} for c, v in boot_netR.items()}

    threshold_sensitivity = {}
    for tau_c in (1, 2, 3):
        counts = {"COHERENT": 0, "CONFLICTING": 0, "OPPOSED": 0, "INSUFFICIENT": 0}
        withheld_pairs = []
        for (ci, cj), records in cells.items():
            t = type_with_threshold(records, tau_c, 3)
            counts[t] += 1
            if t in ("CONFLICTING", "INSUFFICIENT"):
                withheld_pairs.append(f"{ci}->{cj}")
        threshold_sensitivity[tau_c] = {"counts": counts, "withheld": withheld_pairs}

    out = {
        "context": ctx,
        "imputation_sensitivity": {
            "zero_impute_netR": {c: float(netR_zero[i]) for i, c in enumerate(CRITERIA)},
            "rowmean_impute_netR": {c: float(netR_alt[i]) for i, c in enumerate(CRITERIA)},
        },
        "bootstrap_netR_gated": boot_ci,
        "threshold_sensitivity": {str(k): v for k, v in threshold_sensitivity.items()},
    }
    with open(f"data/v2/results_robustness_{ctx}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"=== {ctx}: imputation sensitivity ===")
    for c in CRITERIA:
        print(f"  {c}: zero={netR_zero[idx[c]]:+.3f}  rowmean={netR_alt[idx[c]]:+.3f}")
    print(f"=== {ctx}: threshold sensitivity ===")
    for tau_c, d in threshold_sensitivity.items():
        print(f"  tau_c={tau_c}: {d['counts']}")

if __name__ == "__main__":
    ctx = sys.argv[1] if len(sys.argv) > 1 else "qatar"
    main(ctx)
