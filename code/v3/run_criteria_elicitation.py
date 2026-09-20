#!/usr/bin/env python
"""
Step 1 of the criteria-elicitation pivot: instead of a hand-picked criteria
set, ask each of 3 LLMs (3 paraphrased prompts each) to independently
propose the criteria for a data-center sustainability MCDM problem in a
given real market. This tests LLM-to-LLM (cross-model) consensus on
problem structuring itself, not just on judgments given a fixed criteria
set.
"""
import json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = ["anthropic/claude-haiku-4.5", "openai/gpt-4o-mini", "google/gemini-3.5-flash-lite"]

CONTEXTS = {
    "qatar": (
        "a data center in Qatar's data center market (hot arid climate, water scarcity, "
        "rapid market growth, MEEZA Q.P.S.C. Tier III facilities, Microsoft's Middle East "
        "hyperscale region, targeted 800 MW renewable-capacity boost)"
    ),
    "ireland": (
        "a data center in Ireland's data center market (data centers consumed 23% of "
        "Ireland's metered national electricity in 2025; a pause on new grid connections "
        "in Dublin and the Greater Dublin Area is in effect until 2028; new facilities must "
        "source at least 80% of annual energy demand from new renewable sources; cool "
        "maritime climate enabling extensive free-air cooling)"
    ),
}

def prompt_variant(ctx_text, variant):
    base = (f"You are helping structure a multicriteria decision-making problem for improving "
            f"the sustainability and operational efficiency of {ctx_text}.")
    if variant == 0:
        q = (f"{base} List the 5 most important decision criteria, one per line, as short noun "
             f"phrases (2-4 words each, e.g. \"energy efficiency\"), ranked from most to least "
             f"important. Respond ONLY with compact JSON, no markdown: "
             f'{{"criteria": ["<criterion 1>", "<criterion 2>", "<criterion 3>", "<criterion 4>", "<criterion 5>"]}}')
    elif variant == 1:
        q = (f"{base} What are the 5 criteria a decision-maker should weigh? Give short noun "
             f"phrases (2-4 words), most important first. Return compact JSON only: "
             f'{{"criteria": ["<criterion 1>", "<criterion 2>", "<criterion 3>", "<criterion 4>", "<criterion 5>"]}}')
    else:
        q = (f"{base} Identify the top 5 factors relevant to this decision, as short noun "
             f"phrases (2-4 words), ordered by importance. Answer with compact JSON only, no "
             f"extra text: "
             f'{{"criteria": ["<criterion 1>", "<criterion 2>", "<criterion 3>", "<criterion 4>", "<criterion 5>"]}}')
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

LIST_RE = re.compile(r'"criteria"\s*:\s*\[(.*?)\]', re.S)
ITEM_RE = re.compile(r'"([^"]+)"')

def parse(raw_text):
    if not raw_text:
        return None
    m = LIST_RE.search(raw_text)
    if not m:
        return None
    items = ITEM_RE.findall(m.group(1))
    return [it.strip() for it in items if it.strip()]

def run_context(ctx_name, ctx_text, out_path):
    jobs = [(model, variant) for model in MODELS for variant in (0, 1, 2)]
    def worker(job):
        model, variant = job
        q = prompt_variant(ctx_text, variant)
        resp_text = call_model(model, q)
        criteria = parse(resp_text)
        return {"model": model, "variant": variant, "raw_response": resp_text, "criteria": criteria}
    raw = []
    with ThreadPoolExecutor(max_workers=9) as ex:
        futs = {ex.submit(worker, j): j for j in jobs}
        for fut in as_completed(futs):
            raw.append(fut.result())
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)
    print(f"[{ctx_name}] saved {out_path} ({len(raw)} calls)")
    for r in raw:
        print(f"  {r['model']:28s} v{r['variant']}: {r['criteria']}")

def main():
    os.makedirs("data/v3", exist_ok=True)
    for ctx_name, ctx_text in CONTEXTS.items():
        run_context(ctx_name, ctx_text, f"data/v3/raw_criteria_{ctx_name}.json")

if __name__ == "__main__":
    main()
