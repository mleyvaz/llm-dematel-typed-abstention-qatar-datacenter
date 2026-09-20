#!/usr/bin/env python
"""
Combines the existing DEMATEL causal map (v2: naive/gated prominence D+R
and net cause/effect D-R, 7 criteria, computed from 3-model pairwise
elicitation) with newly LLM-elicited importance weights for the same 7
criteria (v4: 9-model panel, Qwen excluded -- it failed all 6 weight
elicitation calls). Produces weighted prominence = weight_i * (D+R)_i,
normalized so weights sum to 1, which is the standard way DEMATEL is
combined with an external weighting method (cf. DEMATEL-based ANP).
"""
import json
import numpy as np
from scipy.stats import f_oneway

CRITERIA = ["PUE", "REN", "COOL", "CAPEX", "REG", "WATER", "GRID"]

def anova_weights(raw):
    by_model = {}
    for r in raw["responses"]:
        if r["weights"]:
            by_model.setdefault(r["model"], []).append(r["weights"])
    m = len(by_model)
    k = len(next(iter(by_model.values())))
    out = {}
    for c in CRITERIA:
        groups = [[w[c] for w in ws] for ws in by_model.values()]
        model_means = {model: float(np.mean([w[c] for w in ws])) for model, ws in by_model.items()}
        grand_mean = float(np.mean([v for g in groups for v in g]))
        ssw = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups)
        ssb = sum(k * (np.mean(g) - grand_mean) ** 2 for g in groups)
        msw = ssw / (m * (k - 1))
        msb = ssb / (m - 1)
        f_stat, p_value = f_oneway(*groups)
        sigma2_b = max(0.0, (msb - msw) / k)
        out[c] = {"mean": grand_mean, "F": float(f_stat), "p": float(p_value),
                   "sigma2_between": float(sigma2_b), "n_models": m}
    return out

def main(ctx):
    with open(f"data/v2/results_dematel_{ctx}.json", encoding="utf-8") as f:
        dem = json.load(f)
    with open(f"data/v4/raw_dematel_weights_{ctx}.json", encoding="utf-8") as f:
        raw_w = json.load(f)

    w_stats = anova_weights(raw_w)
    raw_weights = {c: w_stats[c]["mean"] for c in CRITERIA}
    total = sum(raw_weights.values())
    norm_weights = {c: raw_weights[c] / total for c in CRITERIA}

    out = {"context": ctx, "criteria": CRITERIA, "weight_stats": w_stats,
           "normalized_weights": norm_weights, "weighted_prominence": {}, "weighted_net": {}}
    for pipeline in ("naive", "gated"):
        prom = dem[f"prominence_{pipeline}"]
        net = dem[f"net_{pipeline}"]
        wp = {c: norm_weights[c] * prom[i] for i, c in enumerate(CRITERIA)}
        wn = {c: norm_weights[c] * net[i] for i, c in enumerate(CRITERIA)}
        out["weighted_prominence"][pipeline] = wp
        out["weighted_net"][pipeline] = wn

    with open(f"data/v4/results_weighted_dematel_{ctx}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"=== {ctx}: weights (9 models, Qwen excluded -- failed all 6 weight calls) ===")
    ranked = sorted(CRITERIA, key=lambda c: -w_stats[c]["sigma2_between"])
    for c in CRITERIA:
        s = w_stats[c]
        print(f"  {c}: weight={norm_weights[c]*100:.1f}%  F={s['F']:.2f} p={s['p']:.3f}  "
              f"sigma2_between={s['sigma2_between']:.2f}")
    print("  most contested (by effect size):", ranked[0], ranked[1])
    print("\n  weighted prominence (gated), ranked:")
    for c, v in sorted(out["weighted_prominence"]["gated"].items(), key=lambda kv: -kv[1]):
        print(f"    {c}: {v:.3f}  (unweighted gated prominence={dem['prominence_gated'][CRITERIA.index(c)]:.3f})")

if __name__ == "__main__":
    import sys
    ctx = sys.argv[1] if len(sys.argv) > 1 else "qatar"
    main(ctx)
