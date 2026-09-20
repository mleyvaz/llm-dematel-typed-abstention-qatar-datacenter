#!/usr/bin/env python
"""
v4: weight elicitation with the 10-model panel, on the majority-consensus
(>=5-of-10-models) criteria sets from keyword_theme_consensus.py.
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

CONSENSUS_CRITERIA = {
    "qatar": {
        "criteria": ["Energy efficiency", "Water usage / cooling", "Renewable energy",
                     "Capital and operating cost", "Scalability and reliability"],
        "context": ("a data center in Qatar's data center market (hot arid climate, water "
                     "scarcity, rapid market growth)"),
    },
    "ireland": {
        "criteria": ["Renewable energy", "Water usage / cooling", "Energy efficiency",
                     "Capital and operating cost", "Grid connection capacity",
                     "Regulatory compliance"],
        "context": ("a data center in Ireland's data center market (grid connection pause in "
                     "Dublin until 2028, 80% new-renewable sourcing mandate, cool maritime "
                     "climate)"),
    },
}

def prompt_variant(ctx_text, criteria, variant):
    clist = ", ".join(f'"{c}"' for c in criteria)
    if variant == 0:
        q = (f"For a multicriteria decision about improving sustainability in {ctx_text}, "
             f"assign a relative importance weight (0-100) to each of these criteria, so the "
             f"weights sum to approximately 100: {clist}. "
             f'Respond ONLY with compact JSON: {{"weights": {{"<criterion>": <weight>, ...}}}}')
    elif variant == 1:
        q = (f"Given this decision context: {ctx_text}. How important is each of the following "
             f"criteria relative to the others, on a 0-100 scale summing to about 100? "
             f"{clist}. Return compact JSON only: "
             f'{{"weights": {{"<criterion>": <weight>, ...}}}}')
    else:
        q = (f"You are weighting decision criteria for {ctx_text}. Distribute 100 points across "
             f"these criteria according to their relative importance: {clist}. Answer with "
             f'compact JSON only, no extra text: {{"weights": {{"<criterion>": <weight>, ...}}}}')
    return q

def call_model(model, content):
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": content}],
        "max_tokens": 300, "temperature": 0.7,
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

def parse(raw_text, criteria):
    if not raw_text:
        return None
    m = OBJ_RE.search(raw_text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return None
    out = {}
    for c in criteria:
        for k, v in obj.items():
            if k.strip().lower() == c.strip().lower():
                try:
                    out[c] = float(v)
                except Exception:
                    pass
    return out if len(out) == len(criteria) else None

def run_context(ctx_name, spec, out_path):
    criteria = spec["criteria"]
    jobs = [(model, variant) for model in MODELS for variant in (0, 1, 2)]
    def worker(job):
        model, variant = job
        q = prompt_variant(spec["context"], criteria, variant)
        resp_text = call_model(model, q)
        weights = parse(resp_text, criteria)
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
        json.dump({"criteria": criteria, "responses": raw}, f, indent=2, ensure_ascii=False)
    n_fail = sum(1 for r in raw if not r["weights"])
    print(f"[{ctx_name}] saved {out_path}; parse failures: {n_fail}/{len(raw)}")

def main():
    os.makedirs("data/v4", exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for ctx_name, spec in CONSENSUS_CRITERIA.items():
        if which != "all" and which != ctx_name:
            continue
        run_context(ctx_name, spec, f"data/v4/raw_weights_{ctx_name}.json")

if __name__ == "__main__":
    main()
