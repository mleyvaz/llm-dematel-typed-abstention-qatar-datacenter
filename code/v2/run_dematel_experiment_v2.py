#!/usr/bin/env python
"""
Extended real LLM-DEMATEL pilot: 7 criteria (up from 5), k=3 paraphrases
(up from 2), 3 models, run independently in two real decision contexts
(Qatar and Ireland) so the typed-abstention gate is tested on more than
one instance, per the round-5 adversarial review's evidential-weight
objection.
"""
import json, os, sys, time, re, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = ["anthropic/claude-haiku-4.5", "openai/gpt-4o-mini", "google/gemini-3.5-flash-lite"]

CRITERIA = [
    ("PUE", "Energy efficiency (Power Usage Effectiveness)"),
    ("REN", "Renewable energy sourcing"),
    ("COOL", "Cooling technology advancement (e.g., liquid cooling)"),
    ("CAPEX", "Capital expenditure"),
    ("REG", "Regulatory and compliance risk"),
    ("WATER", "Water usage / water-scarcity risk (Water Usage Effectiveness)"),
    ("GRID", "Grid connection capacity / power-supply reliability risk"),
]

CONTEXTS = {
    "qatar": (
        "a data center in Qatar's data center market (MEEZA Q.P.S.C.'s Tier III facilities, "
        "Microsoft's Middle East hyperscale region, hot arid climate, water scarcity, rapid "
        "market growth, ~800 MW renewable-capacity target)"
    ),
    "ireland": (
        "a data center in Ireland's data center market (data centers consumed 23% of Ireland's "
        "metered national electricity in 2025, up from 5% in 2015, per the Central Statistics "
        "Office; a pause on new data center grid connections in Dublin and the Greater Dublin "
        "Area is in effect until 2028; new facilities must source at least 80% of annual energy "
        "demand from new renewable sources per the Commission for Regulation of Utilities; cool "
        "maritime climate enabling extensive free-air cooling)"
    ),
}

def prompt_variant(ci_name, cj_name, variant, ctx_text):
    scale = "0 (no influence), 1 (low), 2 (medium), 3 (high), 4 (very high influence)"
    if variant == 0:
        q = (f"Using the standard DEMATEL direct-influence scale {scale}, how much does \"{ci_name}\" "
             f"directly influence/cause changes in \"{cj_name}\" in the context of improving the "
             f"sustainability and operational efficiency of {ctx_text}? "
             f'Respond ONLY with compact JSON, no markdown: {{"score": <0-4 integer>, "justification": "<one short sentence>"}}')
    elif variant == 1:
        q = (f"Considering causal influence in the context of {ctx_text}, to what degree does "
             f"\"{ci_name}\" directly drive or cause \"{cj_name}\"? Use the DEMATEL scale: {scale}. "
             f'Return compact JSON only: {{"score": <0-4 integer>, "justification": "<one short sentence>"}}')
    else:
        q = (f"In {ctx_text}, rate the direct causal impact of \"{ci_name}\" on \"{cj_name}\" using "
             f"the DEMATEL 0-4 scale ({scale}). Answer with compact JSON only, no extra text: "
             f'{{"score": <0-4 integer>, "justification": "<one short sentence>"}}')
    return q

def call_model(model, content):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 220,
        "temperature": 0.7,
    }).encode("utf-8")
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
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

SCORE_RE = re.compile(r'"score"\s*:\s*"?([0-4])"?')
FALLBACK_RE = re.compile(r'\b([0-4])\b')
JUST_RE = re.compile(r'"justification"\s*:\s*"([^"]*)"')

def parse(raw_text):
    if not raw_text:
        return None, None
    m = SCORE_RE.search(raw_text)
    score = int(m.group(1)) if m else None
    if score is None:
        m2 = FALLBACK_RE.search(raw_text)
        score = int(m2.group(1)) if m2 else None
    jm = JUST_RE.search(raw_text)
    justification = jm.group(1) if jm else (raw_text[:200] if raw_text else "")
    return score, justification

def run_context(ctx_name, ctx_text, out_path, max_workers=10):
    pairs = [(ci, cj) for ci in CRITERIA for cj in CRITERIA if ci[0] != cj[0]]
    jobs = []
    for (ci_id, ci_name), (cj_id, cj_name) in pairs:
        for model in MODELS:
            for variant in (0, 1, 2):
                jobs.append((ci_id, cj_id, ci_name, cj_name, model, variant))
    total = len(jobs)
    print(f"[{ctx_name}] {total} calls queued")

    def worker(job):
        ci_id, cj_id, ci_name, cj_name, model, variant = job
        q = prompt_variant(ci_name, cj_name, variant, ctx_text)
        resp_text = call_model(model, q)
        score, justification = parse(resp_text)
        return {
            "ci": ci_id, "cj": cj_id, "model": model, "variant": variant,
            "raw_response": resp_text, "score": score, "justification": justification,
        }

    raw = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(worker, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            raw.append(r)
            done += 1
            if done % 25 == 0 or done == total:
                print(f"[{ctx_name}] {done}/{total} done")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)
    print(f"[{ctx_name}] saved {out_path}")

def main():
    os.makedirs("data/v2", exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for ctx_name, ctx_text in CONTEXTS.items():
        if which != "all" and which != ctx_name:
            continue
        run_context(ctx_name, ctx_text, f"data/v2/raw_dematel_responses_{ctx_name}.json")

if __name__ == "__main__":
    main()
