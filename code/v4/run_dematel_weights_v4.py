#!/usr/bin/env python
"""
Derives importance weights for the ORIGINAL 7-criterion DEMATEL set
(PUE, REN, COOL, CAPEX, REG, WATER, GRID -- from the companion DEMATEL
pilot, code/v2/), using the 10-model panel, so the existing DEMATEL causal
map (prominence D+R, net cause/effect D-R, already computed in
data/v2/results_dematel_{ctx}.json) can be combined with an importance
weight per criterion into a weighted-prominence ranking. This does not
re-run the pairwise causal-influence elicitation itself (that would need
n(n-1) pairs x k paraphrases x 10 models, ~1800 more calls); it reuses the
existing D+R/D-R values and adds a weight layer on top, which is the
standard way DEMATEL is combined with a weighting method (e.g. weighted
DEMATEL / DEMATEL-based ANP) in the literature.
"""
import json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = [
    "anthropic/claude-haiku-4.5", "openai/gpt-4o-mini", "google/gemini-3.5-flash-lite",
    "deepseek/deepseek-chat", "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-small-3.1-24b-instruct", "qwen/qwen-2.5-72b-instruct",
    "amazon/nova-lite-v1", "cohere/command-r-08-2024", "x-ai/grok-4.3",
]

CRIT_LABEL = {
    "PUE": "Energy efficiency", "REN": "Renewable energy sourcing",
    "COOL": "Cooling technology", "CAPEX": "Capital expenditure",
    "REG": "Regulatory/compliance risk", "WATER": "Water usage / water-scarcity risk",
    "GRID": "Grid connection capacity / power-supply risk",
}
CRITERIA = list(CRIT_LABEL.keys())

CONTEXTS = {
    "qatar": ("a data center in Qatar's data center market (hot arid climate, water scarcity, "
              "rapid market growth, MEEZA Q.P.S.C. Tier III facilities, Microsoft's Middle East "
              "hyperscale region)"),
    "ireland": ("a data center in Ireland's data center market (data centers consumed 23% of "
                "Ireland's metered national electricity in 2025; a pause on new grid connections "
                "in Dublin until 2028; new facilities must source at least 80% of annual energy "
                "demand from new renewable sources; cool maritime climate)"),
}

def prompt_variant(ctx_text, variant):
    clist = ", ".join(f'"{CRIT_LABEL[c]}"' for c in CRITERIA)
    if variant == 0:
        q = (f"For a multicriteria decision about improving sustainability in {ctx_text}, assign "
             f"a relative importance weight (0-100) to each of these criteria, so the weights sum "
             f"to approximately 100: {clist}. Respond ONLY with compact JSON: "
             f'{{"weights": {{"<criterion>": <weight>, ...}}}}')
    elif variant == 1:
        q = (f"Given this decision context: {ctx_text}. How important is each of the following "
             f"criteria relative to the others, on a 0-100 scale summing to about 100? {clist}. "
             f'Return compact JSON only: {{"weights": {{"<criterion>": <weight>, ...}}}}')
    else:
        q = (f"You are weighting decision criteria for {ctx_text}. Distribute 100 points across "
             f"these criteria according to their relative importance: {clist}. Answer with "
             f'compact JSON only, no extra text: {{"weights": {{"<criterion>": <weight>, ...}}}}')
    return q

def call_model(model, content):
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": content}],
        "max_tokens": 350, "temperature": 0.7,
    }).encode("utf-8")
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json",
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2 * (attempt + 1))

OBJ_RE = re.compile(r'"weights"\s*:\s*(\{.*?\})', re.S)

def parse(raw_text):
    if not raw_text:
        return None
    m = OBJ_RE.search(raw_text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return None
    label_to_code = {v.lower(): k for k, v in CRIT_LABEL.items()}
    out = {}
    for k, v in obj.items():
        code = label_to_code.get(k.strip().lower())
        if code:
            try:
                out[code] = float(v)
            except Exception:
                pass
    return out if len(out) == len(CRITERIA) else None

def run_context(ctx_name, ctx_text, out_path):
    jobs = [(model, variant) for model in MODELS for variant in (0, 1, 2)]
    def worker(job):
        model, variant = job
        q = prompt_variant(ctx_text, variant)
        resp_text = call_model(model, q)
        weights = parse(resp_text)
        return {"model": model, "variant": variant, "raw_response": resp_text, "weights": weights}
    raw = []
    done = 0
    with ThreadPoolExecutor(max_workers=15) as ex:
        futs = {ex.submit(worker, j): j for j in jobs}
        for fut in as_completed(futs):
            raw.append(fut.result())
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"[{ctx_name}] {done}/{len(jobs)} done")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"criteria": CRITERIA, "responses": raw}, f, indent=2, ensure_ascii=False)
    n_fail = sum(1 for r in raw if not r["weights"])
    print(f"[{ctx_name}] saved {out_path}; parse failures: {n_fail}/{len(raw)}")

def main():
    os.makedirs("data/v4", exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for ctx_name, ctx_text in CONTEXTS.items():
        if which != "all" and which != ctx_name:
            continue
        run_context(ctx_name, ctx_text, f"data/v4/raw_dematel_weights_{ctx_name}.json")

if __name__ == "__main__":
    main()
