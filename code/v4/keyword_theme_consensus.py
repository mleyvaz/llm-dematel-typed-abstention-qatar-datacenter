#!/usr/bin/env python
"""
Keyword-based theme tagging as a transparent, reproducible alternative to
TF-IDF clustering for the 10-model panel (150 proposals per context is too
large for reliable single-annotator hand reconciliation, and the automated
clustering both over- and under-merges depending on local vocabulary
overlap -- see the manuscript's methods note). Each theme is a fixed
keyword substring list, defined by inspecting the full proposal vocabulary
once, before scoring. A model "proposes" a theme if ANY of its 3
paraphrase variants contains a matching phrase.
"""
import json, sys

THEMES = {
    "Energy efficiency": ["energy efficiency", "energy consumption", "power usage"],
    "Renewable energy": ["renewable"],
    "Water usage / cooling": ["water", "cooling", "cool ", "thermal"],
    "Capital / operating cost": ["capital", "cost", "expenditure", "cost-effect"],
    "Scalability / reliability": ["scalab", "reliab", "resilien", "uptime"],
    "Regulatory compliance": ["regulat", "compliance", "certification", "sustainability report"],
    "Grid connection capacity": ["grid"],
    "Carbon footprint / emissions": ["carbon", "emission"],
    "Environmental impact": ["environmental impact"],
    "Market growth / demand": ["market growth", "demand", "growth adaptab"],
}

def main(ctx):
    with open(f"data/v4/raw_criteria_{ctx}.json", encoding="utf-8") as f:
        raw = json.load(f)
    by_model = {}
    for r in raw:
        if not r["criteria"]:
            continue
        by_model.setdefault(r["model"], set())
        for phrase in r["criteria"]:
            by_model[r["model"]].add(phrase.lower())
    n_models = len(by_model)

    out_themes = []
    for theme, keywords in THEMES.items():
        models_hit = set()
        for model, phrases in by_model.items():
            if any(any(kw in p for kw in keywords) for p in phrases):
                models_hit.add(model)
        out_themes.append({"theme": theme, "n_distinct_models": len(models_hit),
                            "models": sorted(models_hit)})
    out_themes.sort(key=lambda t: -t["n_distinct_models"])
    out = {"context": ctx, "n_models": n_models, "themes": out_themes}
    with open(f"data/v4/results_keyword_consensus_{ctx}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"=== {ctx}: keyword-theme consensus out of {n_models} models ===")
    for t in out_themes:
        print(f"  {t['n_distinct_models']}/{n_models}  {t['theme']}")

if __name__ == "__main__":
    ctx = sys.argv[1] if len(sys.argv) > 1 else "qatar"
    main(ctx)
