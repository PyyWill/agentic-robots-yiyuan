"""Claude text-memory baseline with identical inputs: the same skill names, candidate predicates and values,
outcomes and test sets that the Overhypothesis agent sees (training is replayed with the same seeds).

    OPENROUTER_API_KEY=... python -m experiments.claude_baseline --conditions base noise0.2 varied --seeds 5 --m 16

Writes results/raw/claude_<condition>.jsonl, one line per test case.
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from overhyp.toolworld import attempt
from experiments.run import setup, train_next

MODEL = "anthropic/claude-fable-5.1"
URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM = ("You are the decision module of a robot that learns tool use from its own experience. Each row of the "
          "experience log is one past attempt of a skill: the situation it was made in, the yes/no values of some "
          "properties of that (situation, tool) pair, and the outcome. Attempts can occasionally fail at random. "
          "Use the experience to choose, in a new situation, the option most likely to succeed. "
          "End your reply with a line 'ANSWER: <letter>'.")
ROOT = Path(__file__).resolve().parents[1]


def bits(x):
    return " ".join("1" if v else "0" for v in x)


def prompt(name, cands, log, options, failed):
    head = " ".join(f"P{i + 1}" for i in range(len(cands)))
    lines = [f"Skill: {name}", "Properties:"] + [f"  P{i + 1} = {c}" for i, c in enumerate(cands)]
    lines += ["", f"Experience (situation | {head} | outcome):"]
    lines += [f"s{sid} | {bits(x)} | {'success' if y else 'failure'}" for sid, x, y in log]
    lines += ["", f"New situation. Options ({head}):"] + [f"{chr(65 + k)} | {bits(x)}" for k, x in enumerate(options)]
    if failed:
        lines += ["", "Already tried in this situation, and failed: " + ", ".join(chr(65 + k) for k in failed)]
    return "\n".join(lines + ["", "Which option is most likely to succeed?"])


def ask(text, max_tokens):
    body = {"model": MODEL, "max_tokens": max_tokens, "usage": {"include": True},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]}
    for k in range(6):
        try:
            r = requests.post(URL, json=body, timeout=600,
                              headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
            if r.status_code == 200 and r.json().get("choices"):
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 ** k)
    raise RuntimeError(f"OpenRouter request failed: {r.status_code} {r.text[:300]}")


def solve(case, max_tokens, attempts=2):
    """One test situation: up to two choices; the failed option is shown before the second."""
    name, cands, log, s, actions, X, noise, seed = case
    rng = np.random.default_rng(seed)
    out = {"choices": [], "outcomes": [], "cost": 0.0, "tokens": [0, 0], "parse_fail": 0, "answers": []}
    for _ in range(attempts):
        resp = ask(prompt(name, cands, log, X, out["choices"]), max_tokens)
        text = resp["choices"][0]["message"]["content"] or ""
        u = resp.get("usage", {})
        out["cost"] += u.get("cost", 0.0) or 0.0
        out["tokens"][0] += u.get("prompt_tokens", 0)
        out["tokens"][1] += u.get("completion_tokens", 0)
        found = re.findall(r"ANSWER:\s*\**\s*([A-Z])", text)
        k = ord(found[-1]) - 65 if found else -1
        if not 0 <= k < len(actions) or k in out["choices"]:
            out["parse_fail"] += 1
            k = int(rng.choice([i for i in range(len(actions)) if i not in out["choices"]]))
        out["answers"].append(text[-400:])
        y = attempt(s, actions[k], rng, noise)
        out["choices"].append(k), out["outcomes"].append(bool(y))
        if y:
            break
    return out


def cases_for(name, seed, m, n_l2, n_inv):
    """Replay training to m situations; return the test cases with the agent's final predicates."""
    cfg, agent, rng, sets = setup(name, seed)
    seen = []
    for _ in range(m):
        train_next(agent, rng, cfg, seen)
    out = []
    for si, (key, n) in enumerate((("L2_ood", n_l2), ("invent", n_inv))):
        for i, (s, actions) in enumerate(sets.get(key, [])[:n]):
            skill, sk = agent.skill(s)
            log = [(tr.s.sid, x, tr.y) for tr, x in zip(agent.trials(skill), sk.X([(tr.s, tr.t) for tr in agent.trials(skill)]))]
            X = sk.X([(s, t) for t in actions])
            out.append(((key, i), (skill, sk.cands, log, s, actions, X, agent.noise, [seed, 6, si, i])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["base", "noise0.2", "varied"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--n-l2", type=int, default=30)
    ap.add_argument("--n-inv", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--budget", type=float, default=150.0, help="stop after this many USD")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    spent = 0.0
    for name in a.conditions:
        path = ROOT / "results" / "raw" / f"claude_{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        done = {(r["seed"], r["m"], r["set"], r["case"]) for r in map(json.loads, open(path))} if path.exists() else set()
        for seed in range(a.seeds):
            todo = [(key, c) for key, c in cases_for(name, seed, a.m, a.n_l2, a.n_inv) if (seed, a.m, *key) not in done]
            with ThreadPoolExecutor(a.threads) as pool:
                results = list(pool.map(lambda kc: solve(kc[1], a.max_tokens), todo))
            with open(path, "a") as f:
                for (key, _), r in zip(todo, results):
                    f.write(json.dumps({"condition": name, "seed": seed, "m": a.m, "set": key[0], "case": key[1],
                                        "model": MODEL, **r}) + "\n")
            spent += sum(r["cost"] for r in results)
            ok = [r["outcomes"] for (key, _), r in zip(todo, results) if key[0] == "L2_ood"]
            print(name, seed, f"L2 0-shot {np.mean([o[0] for o in ok]):.2f} within2 {np.mean([any(o) for o in ok]):.2f}"
                  if ok else "", f"spent ${spent:.2f}", flush=True)
            if spent > a.budget:
                print("budget reached; stopping")
                return


if __name__ == "__main__":
    main()
