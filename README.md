# LLM-Elicited DEMATEL with Typed Abstention — Data Center Sustainability Pilot

Code, raw model responses, and computed results for a pilot that elicits DEMATEL
direct-influence judgments among five data-center sustainability criteria from three
LLMs, gates them through a typed-abstention rule before building the causal map, and
tests the robustness of the resulting comparisons to edge imputation and model
resampling.

Criteria: energy efficiency (PUE), renewable energy sourcing (REN), cooling technology
(COOL), capital expenditure (CAPEX), regulatory/compliance risk (REG).

## Contents

- `code/run_dematel_experiment.py` — calls three LLMs (Claude Haiku 4.5, GPT-4o-mini,
  Gemini 3.5 Flash-Lite) via the OpenRouter API for all 20 ordered criterion pairs,
  2 paraphrased prompt variants each (both variants are given verbatim in
  `prompt_variant()`), temperature 0.7. Produces `data/raw_dematel_responses.json`.
- `code/analyze_dematel.py` — typed-abstention typing (COHERENT / CONFLICTING /
  OPPOSED / INSUFFICIENT; within-model conflict threshold τ_c=2, cross-model
  opposition threshold τ_o=3), the naive and gated DEMATEL direct-influence and
  total-relation matrices, and a TF-IDF/agglomerative-clustering proxy for semantic
  entropy over each pair's justification texts. Produces `data/results_dematel.json`.
- `code/robustness_checks.py` — independent row-mean imputation for withheld edges
  (vs. the gate's own 0-imputation), exact enumeration of all 3³=27 model-resample
  outcomes, threshold sensitivity (τ_c ∈ {1,2,3}), and the Spearman/Pearson
  correlation between semantic entropy and numeric dispersion. Produces
  `data/results_robustness.json`.
- `code/make_causeeffect_figure.py` — the naive-vs-gated DEMATEL cause-effect
  scatter plot (`figures/figure1_causeeffect.png`).
- `data/raw_dematel_responses.json` — all 120 raw API calls (model, prompt variant,
  parsed score, justification text).
- `data/results_dematel.json`, `data/results_robustness.json` — computed outputs;
  every number reported in the manuscript's Results section comes from these two
  files.

## Reproducing

```bash
pip install numpy scipy scikit-learn
export OPENROUTER_API_KEY=...        # only needed to re-run the LLM calls
python code/run_dematel_experiment.py   # optional: regenerates data/raw_dematel_responses.json
python code/analyze_dematel.py          # regenerates data/results_dematel.json
python code/robustness_checks.py        # regenerates data/results_robustness.json
python code/make_causeeffect_figure.py  # regenerates figures/figure1_causeeffect.png
```

`data/raw_dematel_responses.json` is committed, so `analyze_dematel.py` and
downstream scripts reproduce the paper's numbers deterministically without needing
API access; only `run_dematel_experiment.py` requires an OpenRouter key, and re-running
it will draw new (temperature-0.7) LLM samples rather than reproduce the committed
raw data exactly.

## Status

Pilot study by Maikel Leyva-Vázquez, run in support of a manuscript in preparation
that grew out of a discussion with Nezir Aydin (Hamad Bin Khalifa University), who
proposed the data-center/DEMATEL direction and the semantic-entropy idea. Aydin's
co-authorship of the resulting manuscript, and the criteria set used here, are
pending his review and confirmation — not yet finalized. The manuscript itself has
not been submitted to any journal.

## License

Code: MIT (see `LICENSE`), copyright Maikel Leyva-Vázquez. Data (`data/*.json`): CC0
— no rights reserved.
