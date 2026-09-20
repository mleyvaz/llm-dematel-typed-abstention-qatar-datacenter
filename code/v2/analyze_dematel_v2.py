#!/usr/bin/env python
"""
Same typed-abstention typing and DEMATEL algebra as code/analyze_dematel.py,
generalized to n=7 criteria and parametrized by decision context so it runs
once per context (Qatar, Ireland). Usage: python analyze_dematel_v2.py <ctx>
"""
import json, sys
import numpy as np

CRITERIA = ["PUE", "REN", "COOL", "CAPEX", "REG", "WATER", "GRID"]
INTERNAL_CONFLICT_THRESHOLD = 2
OPPOSITION_THRESHOLD = 3

def load(ctx):
    with open(f"data/v2/raw_dematel_responses_{ctx}.json", encoding="utf-8") as f:
        return json.load(f)

def group(raw):
    cells = {}
    for r in raw:
        cells.setdefault((r["ci"], r["cj"]), []).append(r)
    return cells

def type_cell(records):
    by_model = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r["score"])
    per_model = {}
    for m, scores in by_model.items():
        usable = [s for s in scores if s is not None]
        if not usable:
            per_model[m] = {"usable": False, "rep": None, "internal_conflict": False}
            continue
        spread = max(usable) - min(usable)
        per_model[m] = {"usable": True, "rep": sum(usable) / len(usable),
                         "internal_conflict": spread >= INTERNAL_CONFLICT_THRESHOLD}
    usable_models = [m for m, d in per_model.items() if d["usable"] and not d["internal_conflict"]]
    any_conflict = any(d["usable"] and d["internal_conflict"] for d in per_model.values())
    n_usable_total = sum(1 for d in per_model.values() if d["usable"])

    if n_usable_total < 2:
        cell_type = "INSUFFICIENT"
    elif any_conflict:
        cell_type = "CONFLICTING"
    else:
        reps = [per_model[m]["rep"] for m in usable_models]
        spread = max(reps) - min(reps) if reps else 0
        cell_type = "OPPOSED" if spread >= OPPOSITION_THRESHOLD else "COHERENT"

    reps_for_agg = [per_model[m]["rep"] for m in usable_models]
    if cell_type in ("INSUFFICIENT", "CONFLICTING"):
        gated_value = None
    elif cell_type == "OPPOSED":
        import statistics as st
        gated_value = st.median(reps_for_agg)
    else:
        gated_value = sum(reps_for_agg) / len(reps_for_agg)

    all_scores = [r["score"] for r in records if r["score"] is not None]
    naive_value = sum(all_scores) / len(all_scores) if all_scores else 0.0
    all_reps = [d["rep"] for d in per_model.values() if d["usable"]]
    dispersion = (max(all_reps) - min(all_reps)) if len(all_reps) >= 2 else 0.0

    return {"type": cell_type, "gated_value": gated_value, "naive_value": naive_value,
            "dispersion": dispersion, "per_model": per_model}

def dematel(D):
    n = D.shape[0]
    row_max = D.sum(axis=1).max()
    col_max = D.sum(axis=0).max()
    s = max(row_max, col_max)
    if s == 0:
        return D.copy(), np.zeros(n), np.zeros(n)
    Dn = D / s
    T = Dn @ np.linalg.inv(np.eye(n) - Dn)
    return T, T.sum(axis=1), T.sum(axis=0)

def main(ctx):
    raw = load(ctx)
    cells = group(raw)
    results = {}
    type_counts = {"COHERENT": 0, "CONFLICTING": 0, "OPPOSED": 0, "INSUFFICIENT": 0}
    for key, records in cells.items():
        res = type_cell(records)
        results[key] = res
        type_counts[res["type"]] += 1

    n = len(CRITERIA)
    idx = {c: i for i, c in enumerate(CRITERIA)}
    D_gated = np.zeros((n, n))
    D_naive = np.zeros((n, n))
    for (ci, cj), res in results.items():
        i, j = idx[ci], idx[cj]
        D_gated[i, j] = res["gated_value"] if res["gated_value"] is not None else 0.0
        D_naive[i, j] = res["naive_value"]

    T_gated, Dg_i, Rg_i = dematel(D_gated)
    T_naive, Dn_i, Rn_i = dematel(D_naive)

    withheld = [k for k, r in results.items() if r["gated_value"] is None]

    out = {
        "context": ctx, "criteria": CRITERIA, "type_counts": type_counts, "n_pairs": len(results),
        "withheld_pairs": [f"{a}->{b}" for a, b in withheld],
        "D_gated": D_gated.tolist(), "D_naive": D_naive.tolist(),
        "prominence_gated": (Dg_i + Rg_i).tolist(), "net_gated": (Dg_i - Rg_i).tolist(),
        "prominence_naive": (Dn_i + Rn_i).tolist(), "net_naive": (Dn_i - Rn_i).tolist(),
        "dispersion_mean": float(np.mean([r["dispersion"] for r in results.values()])),
        "dispersion_max": float(np.max([r["dispersion"] for r in results.values()])),
        "cells_detail": {
            f"{a}->{b}": {"type": r["type"], "gated_value": r["gated_value"],
                          "naive_value": round(r["naive_value"], 3), "dispersion": round(r["dispersion"], 3)}
            for (a, b), r in results.items()
        },
    }
    with open(f"data/v2/results_dematel_{ctx}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"=== {ctx}: TYPE COUNTS (of {len(results)} pairs) ===", type_counts)
    print("withheld:", out["withheld_pairs"])
    for i, c in enumerate(CRITERIA):
        print(f"  {c}: naive D-R={out['net_naive'][i]:+.3f}  gated D-R={out['net_gated'][i]:+.3f}")

if __name__ == "__main__":
    ctx = sys.argv[1] if len(sys.argv) > 1 else "qatar"
    main(ctx)
