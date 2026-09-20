#!/usr/bin/env python
"""
v4: same clustering approach as v3's analyze_criteria_consensus.py, scored
out of 10 distinct models instead of 3. Consensus fraction (not raw count)
is reported so the two studies are comparable.
"""
import json, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

DIST_THRESHOLD = 0.55

def main(ctx):
    with open(f"data/v4/raw_criteria_{ctx}.json", encoding="utf-8") as f:
        raw = json.load(f)

    items = []
    n_models = len(set(r["model"] for r in raw))
    for r in raw:
        if not r["criteria"]:
            continue
        for phrase in r["criteria"]:
            items.append((phrase.lower(), r["model"]))

    phrases = [p for p, m in items]
    models = [m for p, m in items]

    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(phrases)
    sims = cosine_similarity(X)
    dist = 1 - sims
    dist[dist < 0] = 0

    clu = AgglomerativeClustering(n_clusters=None, distance_threshold=DIST_THRESHOLD,
                                   metric="precomputed", linkage="average")
    labels = clu.fit_predict(dist)

    clusters = {}
    for lbl, phrase, model in zip(labels, phrases, models):
        clusters.setdefault(lbl, {"phrases": [], "models": set()})
        clusters[lbl]["phrases"].append(phrase)
        clusters[lbl]["models"].add(model)

    ranked = sorted(clusters.values(), key=lambda c: -len(c["models"]))
    out = {
        "context": ctx, "n_models": n_models, "n_total_proposals": len(items),
        "n_clusters": len(clusters),
        "clusters": [
            {
                "n_distinct_models": len(c["models"]), "models": sorted(c["models"]),
                "n_mentions": len(c["phrases"]),
                "example_phrases": sorted(set(c["phrases"]))[:8],
                "representative_label": max(set(c["phrases"]), key=c["phrases"].count),
            }
            for c in ranked
        ],
    }
    with open(f"data/v4/results_criteria_consensus_{ctx}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"=== {ctx}: {out['n_clusters']} clusters from {out['n_total_proposals']} proposals, {n_models} models ===")
    for c in out["clusters"]:
        print(f"  [{c['n_distinct_models']}/{n_models} models] {c['representative_label']!r} <- {c['example_phrases']}")

if __name__ == "__main__":
    ctx = sys.argv[1] if len(sys.argv) > 1 else "qatar"
    main(ctx)
