#!/usr/bin/env python
"""
Computes: (1) typed-abstention typing per criterion pair (reusing the
transportation paper's internal-conflict/opposition thresholds, unchanged);
(2) the DEMATEL total-relation matrix, gated vs naive; (3) a lexical
(TF-IDF + agglomerative clustering) proxy for semantic entropy over each
pair's justification texts, and its correlation with the ordinal-dispersion
typing. Every number in the paper's Results section comes from this script.
"""
import json, math, statistics as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

CRITERIA = ["PUE", "REN", "COOL", "CAPEX", "REG"]
INTERNAL_CONFLICT_THRESHOLD = 2
OPPOSITION_THRESHOLD = 3
CLUSTER_DISTANCE_THRESHOLD = 0.65  # 1 - cosine similarity; fixed before inspecting results

def load():
    with open("data/raw_dematel_responses.json", encoding="utf-8") as f:
        return json.load(f)

def group(raw):
    cells = {}
    for r in raw:
        key = (r["ci"], r["cj"])
        cells.setdefault(key, []).append(r)
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
        gated_value = None  # will become 0 in the DEMATEL matrix (withheld edge)
    elif cell_type == "OPPOSED":
        gated_value = st.median(reps_for_agg)
    else:
        gated_value = sum(reps_for_agg) / len(reps_for_agg)

    all_scores = [r["score"] for r in records if r["score"] is not None]
    naive_value = sum(all_scores) / len(all_scores) if all_scores else 0.0
    all_reps = [d["rep"] for d in per_model.values() if d["usable"]]
    dispersion = (max(all_reps) - min(all_reps)) if len(all_reps) >= 2 else 0.0

    return {
        "type": cell_type, "gated_value": gated_value, "naive_value": naive_value,
        "dispersion": dispersion, "per_model": per_model,
    }

def semantic_entropy(records, vectorizer, all_vectors, index_map):
    texts = [r["justification"] for r in records if r.get("justification")]
    idxs = [index_map[id(r)] for r in records if r.get("justification")]
    if len(texts) < 2:
        return 0.0
    vecs = all_vectors[idxs]
    sims = cosine_similarity(vecs)
    dist = 1 - sims
    dist[dist < 0] = 0
    n = len(texts)
    if n == 2:
        d = dist[0, 1]
        labels = [0, 0] if d < CLUSTER_DISTANCE_THRESHOLD else [0, 1]
    else:
        clu = AgglomerativeClustering(n_clusters=None, distance_threshold=CLUSTER_DISTANCE_THRESHOLD,
                                       metric="precomputed", linkage="average")
        labels = clu.fit_predict(dist)
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    probs = [c / n for c in counts.values()]
    h = -sum(p * math.log2(p) for p in probs if p > 0)
    h_norm = h / math.log2(n) if n > 1 else 0.0
    return h_norm

def dematel(D):
    n = D.shape[0]
    row_max = D.sum(axis=1).max()
    col_max = D.sum(axis=0).max()
    s = max(row_max, col_max)
    if s == 0:
        return D.copy(), np.zeros(n), np.zeros(n)
    Dn = D / s
    T = Dn @ np.linalg.inv(np.eye(n) - Dn)
    D_i = T.sum(axis=1)  # influence given
    R_i = T.sum(axis=0)  # influence received
    return T, D_i, R_i

def main():
    raw = load()
    cells = group(raw)

    all_texts = [r["justification"] if r.get("justification") else "" for r in raw]
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    all_vectors = vectorizer.fit_transform(all_texts)
    index_map = {id(r): i for i, r in enumerate(raw)}

    results = {}
    type_counts = {"COHERENT": 0, "CONFLICTING": 0, "OPPOSED": 0, "INSUFFICIENT": 0}
    for key, records in cells.items():
        res = type_cell(records)
        res["semantic_entropy"] = semantic_entropy(records, vectorizer, all_vectors, index_map)
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

    prominence_gated = (Dg_i + Rg_i).tolist()
    net_gated = (Dg_i - Rg_i).tolist()
    prominence_naive = (Dn_i + Rn_i).tolist()
    net_naive = (Dn_i - Rn_i).tolist()

    dispersions = [r["dispersion"] for r in results.values()]
    entropies = [r["semantic_entropy"] for r in results.values()]
    # Pearson correlation between dispersion and semantic entropy (real, computed)
    if len(dispersions) > 1 and st.pstdev(dispersions) > 0 and st.pstdev(entropies) > 0:
        mean_d, mean_e = st.mean(dispersions), st.mean(entropies)
        cov = sum((d - mean_d) * (e - mean_e) for d, e in zip(dispersions, entropies)) / len(dispersions)
        corr = cov / (st.pstdev(dispersions) * st.pstdev(entropies))
    else:
        corr = 0.0

    withheld = [k for k, r in results.items() if r["gated_value"] is None]

    out = {
        "criteria": CRITERIA,
        "type_counts": type_counts,
        "n_pairs": len(results),
        "withheld_pairs": [f"{a}->{b}" for a, b in withheld],
        "D_gated": D_gated.tolist(), "D_naive": D_naive.tolist(),
        "prominence_gated": prominence_gated, "net_gated": net_gated,
        "prominence_naive": prominence_naive, "net_naive": net_naive,
        "dispersion_mean": st.mean(dispersions), "dispersion_max": max(dispersions),
        "semantic_entropy_mean": st.mean(entropies), "semantic_entropy_max": max(entropies),
        "corr_dispersion_semanticentropy": corr,
        "cells_detail": {
            f"{a}->{b}": {
                "type": r["type"], "gated_value": r["gated_value"], "naive_value": round(r["naive_value"], 3),
                "dispersion": round(r["dispersion"], 3), "semantic_entropy": round(r["semantic_entropy"], 3),
            } for (a, b), r in results.items()
        },
    }
    with open("data/results_dematel.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("=== TYPE COUNTS (of", len(results), "pairs) ===", type_counts)
    print("withheld pairs:", out["withheld_pairs"])
    print("\n=== GATED DEMATEL ===")
    for i, c in enumerate(CRITERIA):
        print(f"  {c}: D+R (prominence)={prominence_gated[i]:.3f}  D-R (net cause/effect)={net_gated[i]:+.3f}")
    print("\n=== NAIVE DEMATEL ===")
    for i, c in enumerate(CRITERIA):
        print(f"  {c}: D+R (prominence)={prominence_naive[i]:.3f}  D-R (net cause/effect)={net_naive[i]:+.3f}")
    print("\nsemantic entropy mean/max:", round(out["semantic_entropy_mean"], 3), round(out["semantic_entropy_max"], 3))
    print("corr(dispersion, semantic entropy):", round(corr, 3))

if __name__ == "__main__":
    main()
