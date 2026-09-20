#!/usr/bin/env python
"""
Step 2: cluster the 45 proposed criteria phrases (9 responses x 5 criteria)
per context by TF-IDF/cosine similarity, then score each cluster by how
many DISTINCT MODELS (not just how many total mentions, and not
paraphrase-repeats of the same model) proposed a criterion in it. This is
cross-model consensus on problem structuring, a different question from
the within-model paraphrase-consistency typing used for judgments.
"""
import json, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

DIST_THRESHOLD = 0.55  # 1 - cosine similarity; fixed before inspecting results

def main(ctx):
    with open(f"data/v3/raw_criteria_{ctx}.json", encoding="utf-8") as f:
        raw = json.load(f)

    items = []  # (phrase, model)
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
        "context": ctx,
        "n_total_proposals": len(items),
        "n_clusters": len(clusters),
        "clusters": [
            {
                "n_distinct_models": len(c["models"]),
                "models": sorted(c["models"]),
                "n_mentions": len(c["phrases"]),
                "example_phrases": sorted(set(c["phrases"]))[:6],
                "representative_label": max(set(c["phrases"]), key=c["phrases"].count),
            }
            for c in ranked
        ],
    }
    with open(f"data/v3/results_criteria_consensus_{ctx}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"=== {ctx}: {out['n_clusters']} clusters from {out['n_total_proposals']} proposals ===")
    for c in out["clusters"]:
        print(f"  [{c['n_distinct_models']}/3 models] {c['representative_label']!r} <- {c['example_phrases']}")

if __name__ == "__main__":
    ctx = sys.argv[1] if len(sys.argv) > 1 else "qatar"
    main(ctx)
