# LLM-Elicited MCDM Reliability — Data Center Sustainability Pilots

Code, raw model responses, and computed results for two related pilots that test
where LLM input to multicriteria decision-making (MCDM) is reliable, at two different
pipeline stages, for a data center sustainability decision in two real markets (Qatar,
Ireland).

Three studies are included:

- **v1** (`code/*.py`, `data/*.json`, `figures/figure1_causeeffect.png`): the original
  DEMATEL pilot — 5 criteria (PUE, REN, COOL, CAPEX, REG), k=2 paraphrases, one context
  (Qatar), 120 calls.
- **v2** (`code/v2/`, `data/v2/`, `figures/figure_causeeffect_{qatar,ireland}.png`):
  the DEMATEL pilot extended to 7 criteria (v1's five plus water-scarcity risk (WATER)
  and grid connection/power-supply risk (GRID)), k=3 paraphrases, two independent real
  decision contexts (Qatar, Ireland), 756 calls total. Tests a typed-abstention gate
  that flags pairwise causal judgments where a model contradicts itself across
  paraphrases. **Result: two independent checks (random-edge-withholding null control,
  leave-one-edge-out influence ranking) found the gate's flagged edges indistinguishable
  from, or slightly less influential than, arbitrary edges of the same count — the gate
  does not identify structurally special edges on this data.**
- **v3** (`code/v3/`, `data/v3/`): a reframing after v2's null result. Instead of
  testing reliability at the pairwise-judgment stage, tests it one stage earlier, at
  problem structuring itself: three LLMs independently propose criteria (no fixed
  list) for the same two markets, cross-model consensus on the proposals is scored,
  and importance weights for the consensus criteria are elicited and decomposed into
  within-model noise vs. cross-model disagreement. **Result: most weights show
  within-model noise as the dominant source of spread (models substantively agree);
  a minority (grid connection capacity in Ireland, most starkly) show genuine
  cross-model disagreement — a usable, threshold-free reliability signal that v2's
  gate did not provide.** v3 is the current manuscript's primary contribution.

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
- `code/v2/permutation_null_control.py` — random-edge-withholding null control: 5,000
  Monte Carlo draws of random edge subsets the same size as the gate's withheld set,
  per context. Produces `data/v2/results_permutation_{qatar,ireland}.json`.
- `code/v2/leave_one_edge_out.py` — knocks out each of the 42 edges individually and
  ranks them by influence on the net cause/effect vector; compares the gate's flagged
  edges against all others with a Mann-Whitney U test. Produces
  `data/v2/results_leaveoneedgeout_{qatar,ireland}.json`.

### v3 (criteria + weight elicitation, problem-structuring reliability)

- `code/v3/run_criteria_elicitation.py` — asks the same three LLMs, 3 paraphrases
  each, to independently propose the top-5 decision criteria for each context with no
  candidate list supplied (18 calls). Produces `data/v3/raw_criteria_{qatar,ireland}.json`.
- `code/v3/analyze_criteria_consensus.py` — TF-IDF/agglomerative clustering of the 45
  proposed phrases per context, scored by the number of distinct models (0–3)
  proposing a concept. Produces `data/v3/results_criteria_consensus_{qatar,ireland}.json`.
  Note: this lexical clustering under-merges true synonyms (e.g. "cooling efficiency
  optimization" / "free air cooling" / "cooling technology"); the manuscript's Table 1a/1b
  reports hand-reconciled consensus counts checked against the raw responses, not this
  script's raw cluster output directly — both are in the repo for comparison.
- `code/v3/run_weight_elicitation.py` — for the hand-reconciled ≥2-of-3-model consensus
  criteria (5 for Qatar, 7 for Ireland including carbon footprint), asks the same three
  LLMs (3 paraphrases each) to distribute 100 importance points across them (27 calls
  total). Produces `data/v3/raw_weights_{qatar,ireland}.json`. Takes an optional
  `qatar`/`ireland` argument to re-run a single context.
- `code/v3/analyze_weights.py` — an earlier, **statistically biased** decomposition:
  compares raw within-model variance against Var(model means) directly, which is
  inflated by attenuated within-model noise (E[Var_m(w̄_m)] = σ²_between + σ²_within/k)
  and is not a fair comparison. Kept for provenance; **not** the manuscript's source of
  truth.
- `code/v3/analyze_weights_anova.py` — the corrected version: runs a proper one-way
  ANOVA per criterion (3 models × 3 paraphrases, F(2,6)) via `scipy.stats.f_oneway`,
  reporting F, p-value, and a bias-corrected variance-components estimate
  σ̂²_between = max(0, (MSB−MSW)/k). This is what the manuscript's Tables 2a/2b report.
  Produces `data/v3/results_weights_anova_{qatar,ireland}.json`.

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
python code/v2/permutation_null_control.py qatar
python code/v2/permutation_null_control.py ireland
python code/v2/leave_one_edge_out.py qatar
python code/v2/leave_one_edge_out.py ireland

# v3 (current primary manuscript)
python code/v3/run_criteria_elicitation.py       # optional: redraws raw proposals
python code/v3/analyze_criteria_consensus.py qatar
python code/v3/analyze_criteria_consensus.py ireland
python code/v3/run_weight_elicitation.py         # optional: redraws raw weights
python code/v3/analyze_weights_anova.py qatar    # corrected version -- use this one
python code/v3/analyze_weights_anova.py ireland
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
