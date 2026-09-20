#!/usr/bin/env python
"""
Responds to the GPT-5 adversarial review (reviews_adversarial_gpt5.md) with
real, computed fixes, not just softened language:
  #1 zero-imputation sensitivity: recompute gated DEMATEL with an
     alternative imputation (row-naive-mean instead of 0) for withheld edges.
  #3 uncertainty: bootstrap CI on D-R (gated) by resampling models.
  #4 threshold sensitivity: recompute typing/withholding at tau_c = 1 and 3.
  #5 semantic-entropy correlation: exact Pearson r p-value at n=20.
  #6 self-report agreement: real per-model confusion matrix + Cohen's kappa,
     not the crude ">=1 model" rule.
"""
import json, random
import numpy as np
from scipy import stats as sps

with open("data/raw_dematel_responses.json", encoding="utf-8") as f:
    RAW = json.load(f)
with open("data/results_dematel.json", encoding="utf-8") as f:
    STAT = json.load(f)
with open("data/raw_selfreport_responses.json", encoding="utf-8") as f:
    SR_RAW = json.load(f)

CRITERIA = STAT["criteria"]
idx = {c: i for i, c in enumerate(CRITERIA)}

def group_raw():
    cells = {}
    for r in RAW:
        cells.setdefault((r["ci"], r["cj"]), []).append(r)
    return cells

CELLS = group_raw()

def dematel(D):
    n = D.shape[0]
    s = max(D.sum(axis=1).max(), D.sum(axis=0).max())
    if s == 0:
        return D.copy(), np.zeros(n), np.zeros(n)
    Dn = D / s
    T = Dn @ np.linalg.inv(np.eye(n) - Dn)
    return T, T.sum(axis=1), T.sum(axis=0)

# ---------- #1: alternative imputation for withheld edges ----------
# NOTE (fixed after adversarial review round 2): using the global naive value
# for exactly the withheld cell is TAUTOLOGICAL, not an independent check --
# for COHERENT pairs gated==naive already, so filling withheld cells with
# their own naive value reconstructs the naive matrix exactly (Table 3's old
# "naive-mean" column was digit-for-digit identical to Table 2's Naive D-R).
# The real alternative implemented here is LOCAL ROW-MEAN imputation: a
# withheld edge D[i,j] is filled with the mean of criterion i's OTHER
# (non-withheld, gated) outgoing edges -- inferring the missing edge from the
# row's own known behavior, independent of the unreliable direct elicitation
# for that specific (i,j) pair.
withheld = set(tuple(p.split("->")) for p in STAT["withheld_pairs"])
D_zero = np.array(STAT["D_gated"])
D_rowmean = D_zero.copy()
n_crit = len(CRITERIA)
for (ci, cj) in withheld:
    i, j = idx[ci], idx[cj]
    other_vals = [D_zero[i, k] for k in range(n_crit) if k not in (i, j)]
    D_rowmean[i, j] = sum(other_vals) / len(other_vals) if other_vals else 0.0

_, Dg0, Rg0 = dematel(D_zero)
_, Dg1, Rg1 = dematel(D_rowmean)
netR_zero = (Dg0 - Rg0)
netR_alt = (Dg1 - Rg1)

# ---------- #3: EXACT enumeration over model resamples (fixed after review) ----------
# NOTE (fixed after adversarial review round 2): with only 3 models resampled
# with replacement, there are exactly 3^3 = 27 ordered outcomes -- a fully
# enumerable, finite support. Monte Carlo sampling 2000 times from a
# 27-point population manufactures false precision (a fake "95% CI" on a
# near-discrete statistic). We enumerate all 27 outcomes exactly instead.
import itertools
MODELS = ["anthropic/claude-haiku-4.5", "openai/gpt-4o-mini", "google/gemini-3.5-flash-lite"]
boot_netR = {c: [] for c in CRITERIA}
all_resamples = list(itertools.product(MODELS, repeat=3))  # exactly 27, enumerated (not sampled)
for resampled_models in all_resamples:
    D_boot = np.zeros((len(CRITERIA), len(CRITERIA)))
    for (ci, cj), records in CELLS.items():
        by_model = {}
        for r in records:
            by_model.setdefault(r["model"], []).append(r["score"])
        vals = []
        for m in resampled_models:
            scores = [s for s in by_model.get(m, []) if s is not None]
            if scores:
                vals.append(sum(scores) / len(scores))
        if (ci, cj) in withheld:
            D_boot[idx[ci], idx[cj]] = 0.0  # keep the gate's withholding decision fixed; resample only the surviving evidence
        else:
            D_boot[idx[ci], idx[cj]] = sum(vals) / len(vals) if vals else 0.0
    _, Db, Rb = dematel(D_boot)
    net = Db - Rb
    for i, c in enumerate(CRITERIA):
        boot_netR[c].append(net[i])

boot_ci = {}
for c in CRITERIA:
    arr = np.array(boot_netR[c])
    # exact enumeration over all 27 ordered resamples -- report the true min/max
    # range and mean over this small, fully-enumerated support, not a
    # percentile-based "95% CI" (misleading precision for n=27 discrete outcomes)
    boot_ci[c] = {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max()),
                  "n_enumerated": len(arr)}

# ---------- #4: threshold sensitivity (tau_c = 1, 2, 3) ----------
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

threshold_sensitivity = {}
for tau_c in (1, 2, 3):
    counts = {"COHERENT": 0, "CONFLICTING": 0, "OPPOSED": 0, "INSUFFICIENT": 0}
    withheld_pairs = []
    for (ci, cj), records in CELLS.items():
        t = type_with_threshold(records, tau_c, 3)
        counts[t] += 1
        if t in ("CONFLICTING", "INSUFFICIENT"):
            withheld_pairs.append(f"{ci}->{cj}")
    threshold_sensitivity[tau_c] = {"counts": counts, "withheld": withheld_pairs}

# ---------- #5: correlation for semantic-entropy vs. dispersion ----------
# NOTE (fixed after adversarial review round 2): H_sem takes only ~5 distinct
# values (heavy ties from small per-pair text counts), so Pearson r overstates
# precision on what is effectively an ordinal variable. We report Spearman's
# rank correlation (tie-corrected) instead, and keep Pearson only as a
# secondary, explicitly-labeled reference value.
dispersions = [STAT["cells_detail"][k]["dispersion"] for k in STAT["cells_detail"]]
entropies = [STAT["cells_detail"][k]["semantic_entropy"] for k in STAT["cells_detail"]]
r_pearson, p_pearson = sps.pearsonr(dispersions, entropies)
r, p = sps.spearmanr(dispersions, entropies)

# ---------- #6: real confusion matrix + Cohen's kappa per model ----------
sr_by_pair_model = {}
for r_ in SR_RAW:
    sr_by_pair_model[(r_["ci"], r_["cj"], r_["model"])] = r_["category"]

def cohens_kappa(pairs_labels):
    # pairs_labels: list of (truth_bool, pred_bool)
    n = len(pairs_labels)
    po = sum(1 for t, p_ in pairs_labels if t == p_) / n
    p_truth = sum(1 for t, _ in pairs_labels if t) / n
    p_pred = sum(1 for _, p_ in pairs_labels if p_) / n
    pe = p_truth * p_pred + (1 - p_truth) * (1 - p_pred)
    return (po - pe) / (1 - pe) if pe != 1 else float("nan"), po

model_confusion = {}
for m in MODELS:
    labels = []
    tp = fp = tn = fn = 0
    for (ci, cj) in [tuple(k.split("->")) for k in STAT["cells_detail"]]:
        truth = STAT["cells_detail"][f"{ci}->{cj}"]["type"] == "CONFLICTING"
        pred_cat = sr_by_pair_model.get((ci, cj, m))
        pred = (pred_cat == "conflicting_evidence")
        labels.append((truth, pred))
        if truth and pred: tp += 1
        elif truth and not pred: fn += 1
        elif not truth and pred: fp += 1
        else: tn += 1
    kappa, po = cohens_kappa(labels)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    model_confusion[m] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                           "accuracy": po, "kappa": kappa, "precision": precision, "recall": recall}

out = {
    "imputation_sensitivity": {
        "zero_impute_netR": {c: float(netR_zero[i]) for i, c in enumerate(CRITERIA)},
        "rowmean_impute_netR": {c: float(netR_alt[i]) for i, c in enumerate(CRITERIA)},
    },
    "bootstrap_netR_gated": boot_ci,
    "threshold_sensitivity": {str(k): v for k, v in threshold_sensitivity.items()},
    "semantic_entropy_corr": {"r": float(r), "p_value": float(p), "n": len(dispersions),
                               "method": "spearman", "pearson_r": float(r_pearson), "pearson_p": float(p_pearson)},
    "selfreport_confusion_per_model": model_confusion,
}
with open("data/results_robustness.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("=== #1 Imputation sensitivity (net D-R, gated) ===")
for c in CRITERIA:
    print(f"  {c}: zero-impute={netR_zero[idx[c]]:+.3f}  row-mean-impute={netR_alt[idx[c]]:+.3f}")

print("\n=== #3 Exact enumeration (27 model-resamples) of net D-R (gated, zero-impute) ===")
for c in CRITERIA:
    d = boot_ci[c]
    print(f"  {c}: mean={d['mean']:+.3f}  range=[{d['min']:+.3f}, {d['max']:+.3f}] (n={d['n_enumerated']} exact)")

print("\n=== #4 Threshold sensitivity (tau_c) ===")
for tau_c, d in threshold_sensitivity.items():
    print(f"  tau_c={tau_c}: {d['counts']}  withheld={d['withheld']}")

print(f"\n=== #5 Semantic entropy correlation ===\nr={r:.3f}, p={p:.3f}, n={len(dispersions)}")

print("\n=== #6 Self-report confusion matrix per model ===")
for m, d in model_confusion.items():
    print(f"  {m}: TP={d['tp']} FP={d['fp']} TN={d['tn']} FN={d['fn']} "
          f"acc={d['accuracy']:.2f} kappa={d['kappa']:.3f} prec={d['precision']:.2f} rec={d['recall']:.2f}")
