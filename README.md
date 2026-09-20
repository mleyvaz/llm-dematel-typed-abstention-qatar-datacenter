# LLM-Elicited DEMATEL with Typed Abstention — Data Center Sustainability Pilot

Code, raw model responses, and computed results for a pilot that elicits DEMATEL
direct-influence judgments among data-center sustainability criteria from three LLMs,
gates them through a typed-abstention rule before building the causal map, and tests
the robustness of the resulting comparisons to edge imputation, model resampling, and
internal-conflict threshold choice.

Two independent studies are included:

- **v1** (`code/*.py`, `data/*.json`, `figures/figure1_causeeffect.png`): 5 criteria
  (PUE, REN, COOL, CAPEX, REG), k=2 paraphrases, one context (Qatar), 120 calls.
- **v2** (`code/v2/`, `data/v2/`, `figures/figure_causeeffect_{qatar,ireland}.png`):
  7 criteria (v1's five plus water-scarcity risk (WATER) and grid connection/power-
  supply risk (GRID)), k=3 paraphrases, two independent real decision contexts
  (Qatar, Ireland), 756 calls total (378 per context). v2 supersedes v1 as the
  manuscript's primary empirical study; v1 is kept for provenance/comparison.

## Contents

### v1 (original 5-criterion, 1-context pilot)

- `code/run_dematel_experiment.py`, `code/analyze_dematel.py`,
  `code/robustness_checks.py`, `code/make_causeeffect_figure.py` — as described in
  earlier revisions of this README; see docstrings in each file.
- `data/raw_dematel_responses.json`, `data/results_dematel.json`,
  `data/results_robustness.json`.

### v2 (7-criterion, 2-context pilot)

- `code/v2/run_dematel_experiment_v2.py` — calls the same three LLMs (Claude Haiku
  4.5, GPT-4o-mini, Gemini 3.5 Flash-Lite) via OpenRouter for all 7×6=42 ordered
  pairs, k=3 paraphrased prompt variants each (verbatim in `prompt_variant()`),
  temperature 0.7, run independently for the Qatar and Ireland contexts (`CONTEXTS`
  dict). Produces `data/v2/raw_dematel_responses_{qatar,ireland}.json`.
- `code/v2/analyze_dematel_v2.py` — same typed-abstention typing and DEMATEL algebra
  as v1, generalized to n=7 criteria; run once per context. Produces
  `data/v2/results_dematel_{qatar,ireland}.json`.
- `code/v2/robustness_checks_v2.py` — same battery as v1 (row-mean imputation vs.
  0-imputation, exact enumeration of 3³=27 model-resamples, threshold sensitivity at
  τ_c∈{1,2,3}), run once per context. Produces
  `data/v2/results_robustness_{qatar,ireland}.json`.
- `code/v2/make_causeeffect_figure_v2.py` — naive-vs-gated cause-effect scatter plot,
  one per context.

## Reproducing

```bash
pip install numpy scipy matplotlib
export OPENROUTER_API_KEY=...   # only needed to re-run the LLM calls

# v2 (primary study)
python code/v2/run_dematel_experiment_v2.py all      # optional: redraws raw responses
python code/v2/analyze_dematel_v2.py qatar
python code/v2/analyze_dematel_v2.py ireland
python code/v2/robustness_checks_v2.py qatar
python code/v2/robustness_checks_v2.py ireland
python code/v2/make_causeeffect_figure_v2.py qatar
python code/v2/make_causeeffect_figure_v2.py ireland
```

All raw response files are committed, so the `analyze`/`robustness`/`figure` scripts
reproduce the manuscript's numbers deterministically without API access; only the
`run_*_experiment*.py` scripts require an OpenRouter key, and re-running them draws
new (temperature-0.7) LLM samples rather than reproducing the committed raw data
exactly.

## Status

Pilot study by Maikel Leyva-Vázquez, run in support of a manuscript in preparation
that grew out of a discussion with Nezir Aydin (Hamad Bin Khalifa University), who
proposed the data-center/DEMATEL direction. Aydin's co-authorship of the resulting
manuscript, and the criteria set used here, are pending his review and confirmation
— not yet finalized. The manuscript itself has not been submitted to any journal.

## License

Code: MIT (see `LICENSE`), copyright Maikel Leyva-Vázquez. Data (`data/**/*.json`):
CC0 — no rights reserved.
