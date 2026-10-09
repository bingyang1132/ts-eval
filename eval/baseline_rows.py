"""Model-free rows for the zero-shot table: uniform-random legal choice (expected value) and the
greedy baseline agent's actual choice on each held-out state. Needs ts-env only (no torch).

    python eval/baseline_rows.py --data data/oracle_heldout_value_r2_public.jsonl            # -> results/baseline_rows.jsonl
    python eval/baseline_rows.py --data data/oracle_heldout_value_r2_public.jsonl --limit 50 --out /tmp/x.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "oracle"))
import tsenv  # noqa: E402,F401  (puts the ts-env submodule on sys.path)
from baselines import GreedyAgent  # noqa: E402
from twilight import Game  # noqa: E402

NEAR = 0.05


def greedy_seed(rec_id: str) -> int:
    """Tie-break seed for GreedyAgent: a stable hash of the record id, so the greedy row does not depend on the
    order of the records in the file."""
    return int(hashlib.blake2b(rec_id.encode(), digest_size=8).hexdigest(), 16)


def metrics(values: dict[str, float], key: str | None, vmax: float, vmin: float) -> dict:
    if key is None:
        return {"oracle_value": 0.0, "top1": 0.0, "near_best": 0.0, "regret": vmax - vmin, "legal": 0.0}
    v = values[key]
    ov = 1.0 if vmax - vmin < 1e-12 else (v - vmin) / (vmax - vmin)
    return {"oracle_value": ov, "top1": float(v >= vmax - 1e-12), "near_best": float(v >= vmax - NEAR),
            "regret": vmax - v, "legal": 1.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "baseline_rows.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N decisions (smoke test)")
    args = ap.parse_args()
    tsenv.require('data')
    recs = [json.loads(l) for l in open(args.data)]
    if args.limit:
        recs = recs[: args.limit]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rand_rows, greedy_rows = [], []
    for i, r in enumerate(recs):
        values = {o["key"]: o["value"] for o in r["options"]}
        vmax, vmin = r["value_max"], r["value_min"]
        meta = {"game": r["game_seed"], "type": r["decision_type"], "side": r["side"]}
        # uniform random: expectation over legal options
        per = [metrics(values, k, vmax, vmin) for k in values]
        rand_rows.append({**meta, **{m: sum(p[m] for p in per) / len(per) for m in per[0]}})
        game = Game(seed=r["game_seed"])
        for key in r["history"]:
            game.step(key)
        action, _ = GreedyAgent(seed=greedy_seed(r["id"])).act(game, game.decision)
        greedy_rows.append({**meta, **metrics(values, action.key, vmax, vmin)})
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(recs)}", flush=True)
    with open(args.out, "w") as f:
        f.write(json.dumps({"row": "random_legal (expected)", "rows": rand_rows}) + "\n")
        f.write(json.dumps({"row": "greedy_agent", "rows": greedy_rows}) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
