#!/usr/bin/env python
"""
Real LLM elicitation for an LLM-DEMATEL causal-criteria pilot, data center
sustainability decision. Direct-influence scores (0-4, standard DEMATEL
scale) are elicited from 3 LLMs, k=2 paraphrases each, for every ordered
pair of 5 criteria (20 pairs). Typed-abstention typing (reused from the
transportation paper's analyze_experiment.py) flags unreliable pairs before
building the DEMATEL total-relation matrix. Justification texts are kept
for the semantic-entropy analysis in analyze_dematel.py.
"""
import json, os, re, time, urllib.request

API_KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = ["anthropic/claude-haiku-4.5", "openai/gpt-4o-mini", "google/gemini-3.5-flash-lite"]

CRITERIA = [
    ("PUE", "Energy efficiency (Power Usage Effectiveness)"),
    ("REN", "Renewable energy sourcing"),
    ("COOL", "Cooling technology advancement (e.g., liquid cooling)"),
    ("CAPEX", "Capital expenditure"),
    ("REG", "Regulatory and compliance risk"),
]

def prompt_variant(ci_name, cj_name, variant):
    scale = "0 (no influence), 1 (low), 2 (medium), 3 (high), 4 (very high influence)"
    ctx = "in the context of improving the sustainability and operational efficiency of a data center"
    if variant == 0:
        q = (f"Using the standard DEMATEL direct-influence scale {scale}, how much does \"{ci_name}\" "
             f"directly influence/cause changes in \"{cj_name}\" {ctx}? "
             f'Respond ONLY with compact JSON, no markdown: {{"score": <0-4 integer>, "justification": "<one short sentence>"}}')
    else:
        q = (f"Considering causal influence {ctx}, to what degree does \"{ci_name}\" directly drive or cause "
             f"\"{cj_name}\"? Use the DEMATEL scale: {scale}. "
             f'Return compact JSON only: {{"score": <0-4 integer>, "justification": "<one short sentence>"}}')
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

def main():
    raw = []
    pairs = [(ci, cj) for ci in CRITERIA for cj in CRITERIA if ci[0] != cj[0]]
    total = len(pairs) * len(MODELS) * 2
    done = 0
    for (ci_id, ci_name), (cj_id, cj_name) in pairs:
        for model in MODELS:
            for variant in (0, 1):
                q = prompt_variant(ci_name, cj_name, variant)
                resp_text = call_model(model, q)
                score, justification = parse(resp_text)
                raw.append({
                    "ci": ci_id, "cj": cj_id, "model": model, "variant": variant,
                    "raw_response": resp_text, "score": score, "justification": justification,
                })
                done += 1
                print(f"[{done}/{total}] {ci_id}->{cj_id} {model:28s} v{variant} -> {score}")
    with open("data/raw_dematel_responses.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)
    print("Saved data/raw_dematel_responses.json")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    main()
