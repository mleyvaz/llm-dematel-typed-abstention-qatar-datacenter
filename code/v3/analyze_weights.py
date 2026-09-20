#!/usr/bin/env python
"""
Step 4: aggregate the LLM-elicited weights per criterion (mean, overall
std), and decompose disagreement into within-model (across the 3
paraphrases of one model -- decoding/phrasing noise) vs. cross-model
(across the 3 models' own means -- genuine value disagreement) variance,
analogous to a one-way ANOVA variance decomposition.
"""
import json, sys
import numpy as np

def main(ctx):
    with open(f"data/v3/raw_weights_{ctx}.json", encoding="utf-8") as f:
        data = json.load(f)
    criteria = data["criteria"]
    responses = [r for r in data["responses"] if r["weights"]]

    by_model = {}
    for r in responses:
        by_model.setdefault(r["model"], []).append(r["weights"])

    out = {"context": ctx, "criteria": criteria, "per_criterion": {}}
    for c in criteria:
        all_vals = [r["weights"][c] for r in responses]
        model_means = {}
        within_model_vars = []
        for model, ws in by_model.items():
            vals = [w[c] for w in ws]
            model_means[model] = float(np.mean(vals))
            within_model_vars.append(float(np.var(vals)))
        cross_model_var = float(np.var(list(model_means.values())))
        mean_within_model_var = float(np.mean(within_model_vars))
        total_var = float(np.var(all_vals))
        out["per_criterion"][c] = {
            "overall_mean": float(np.mean(all_vals)),
            "overall_std": float(np.std(all_vals)),
            "model_means": model_means,
            "cross_model_variance": cross_model_var,
            "mean_within_model_variance": mean_within_model_var,
            "disagreement_mostly_cross_model": cross_model_var > mean_within_model_var,
        }
    with open(f"data/v3/results_weights_{ctx}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"=== {ctx} ===")
    for c, d in out["per_criterion"].items():
        tag = "CROSS-MODEL disagreement" if d["disagreement_mostly_cross_model"] else "within-model noise dominates"
        print(f"  {c:32s} mean={d['overall_mean']:5.1f} std={d['overall_std']:4.1f}  "
              f"(cross-model var={d['cross_model_variance']:.1f} vs within-model var="
              f"{d['mean_within_model_variance']:.1f} -> {tag})")

if __name__ == "__main__":
    ctx = sys.argv[1] if len(sys.argv) > 1 else "qatar"
    main(ctx)
