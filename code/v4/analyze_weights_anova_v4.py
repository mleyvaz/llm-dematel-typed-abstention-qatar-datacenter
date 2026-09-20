#!/usr/bin/env python
"""
Corrected version of analyze_weights.py after round-100 adversarial review
(Claude Opus 5): the naive comparison of Var_m(model means) against mean
within-model variance is statistically biased, since E[Var_m(w_bar_m)] =
sigma^2_between + sigma^2_within/k -- the cross-model variance estimator is
itself inflated by within-model noise attenuated by 1/k, not a clean
estimate of between-model disagreement alone.

This version runs a proper one-way ANOVA (3 models x 3 paraphrases per
criterion) via scipy.stats.f_oneway, reporting F(2,6), the p-value, and the
bias-corrected variance-components estimate sigma^2_b = max(0, (MSB-MSW)/k).
"""
import json, sys
import numpy as np
from scipy.stats import f_oneway

def main(ctx):
    with open(f"data/v4/raw_weights_{ctx}.json", encoding="utf-8") as f:
        data = json.load(f)
    criteria = data["criteria"]
    responses = [r for r in data["responses"] if r["weights"]]

    by_model = {}
    for r in responses:
        by_model.setdefault(r["model"], []).append(r["weights"])

    k = len(next(iter(by_model.values())))  # paraphrases per model
    m = len(by_model)  # number of models

    out = {"context": ctx, "criteria": criteria, "per_criterion": {}}
    for c in criteria:
        groups = [[w[c] for w in ws] for ws in by_model.values()]
        model_means = {model: float(np.mean([w[c] for w in ws])) for model, ws in by_model.items()}
        grand_mean = float(np.mean([v for g in groups for v in g]))

        # classic one-way ANOVA sums of squares
        ssw = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups)
        ssb = sum(k * (np.mean(g) - grand_mean) ** 2 for g in groups)
        msw = ssw / (m * (k - 1))
        msb = ssb / (m - 1)
        f_stat, p_value = f_oneway(*groups)
        sigma2_b = max(0.0, (msb - msw) / k)  # bias-corrected between-model variance component

        out["per_criterion"][c] = {
            "overall_mean": grand_mean,
            "model_means": model_means,
            "MSW_within_model": float(msw),
            "MSB_between_model": float(msb),
            "F_stat": float(f_stat),
            "p_value": float(p_value),
            "bias_corrected_between_model_variance": float(sigma2_b),
            "significant_at_0.10": bool(p_value < 0.10),
        }
    with open(f"data/v4/results_weights_anova_{ctx}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"=== {ctx}: one-way ANOVA, F(2,6) per criterion ===")
    for c, d in out["per_criterion"].items():
        sig = "p<0.10" if d["significant_at_0.10"] else "n.s."
        print(f"  {c:32s} mean={d['overall_mean']:5.1f}  F={d['F_stat']:5.2f}  p={d['p_value']:.3f} ({sig})  "
              f"sigma2_between(corrected)={d['bias_corrected_between_model_variance']:.2f}")

if __name__ == "__main__":
    ctx = sys.argv[1] if len(sys.argv) > 1 else "qatar"
    main(ctx)
