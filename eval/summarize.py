"""Leaderboard table: one row per vf-eval run under results/<row>/, plus model-free baseline rows.

Cluster bootstrap over games (decisions from one game are correlated), 2000 resamples, 95% CI.
    python eval/summarize.py                     # rows under ./results, baselines from results/baseline_rows.jsonl
    python eval/summarize.py --results other_dir
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("oracle_value", "top1", "near_best", "regret", "legal")


def game_of(info: dict) -> int:
    return int(info["id"].split("_")[0][1:])


def load_run(row_dir: Path) -> list[dict]:
    files = sorted(row_dir.rglob("results.jsonl"))
    if not files:
        return []
    out = []
    for line in files[-1].open():
        r = json.loads(line)
        info = r["info"] if isinstance(r["info"], dict) else json.loads(r["info"])
        out.append({"game": game_of(info), "type": info["decision_type"], "side": info["side"],
                    **{m: float(r[m]) for m in METRICS}})
    return out


def cluster_ci(rows: list[dict], metric: str, n_boot: int = 2000, seed: int = 0):
    by_game = defaultdict(list)
    for r in rows:
        by_game[r["game"]].append(r[metric])
    games = list(by_game)
    rng = random.Random(seed)
    mean = statistics.fmean(r[metric] for r in rows)
    boots = []
    for _ in range(n_boot):
        sample = [v for g in rng.choices(games, k=len(games)) for v in by_game[g]]
        boots.append(statistics.fmean(sample))
    boots.sort()
    return mean, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def fmt(rows, metric):
    if not rows:
        return "-"
    m, lo, hi = cluster_ci(rows, metric)
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"), help="folder of vf-eval runs, one sub-folder per row")
    ap.add_argument("--baselines", default=None, help="baseline_rows.py output (default: <results>/baseline_rows.jsonl)")
    args = ap.parse_args()
    root = Path(args.results)
    runs: dict[str, list[dict]] = {}
    for p in sorted(root.iterdir()) if root.exists() else []:
        if p.is_dir():
            rows = load_run(p)
            if rows:
                runs[p.name] = rows
    base = Path(args.baselines) if args.baselines else root / "baseline_rows.jsonl"
    if base.exists():
        for line in base.open():
            d = json.loads(line)
            runs[d["row"]] = d["rows"]
    if not runs:
        sys.exit("no results")
    print(f"| row | n | games | oracle_value | top1 | near_best | regret | legal |")
    print("|---|---|---|---|---|---|---|---|")
    for name, rows in runs.items():
        ng = len({r["game"] for r in rows})
        print(f"| {name} | {len(rows)} | {ng} | " + " | ".join(fmt(rows, m) for m in METRICS) + " |")
    print("\n### oracle_value by decision type\n")
    types = sorted({r["type"] for rows in runs.values() for r in rows})
    print("| row | " + " | ".join(types) + " |")
    print("|---|" + "---|" * len(types))
    for name, rows in runs.items():
        cells = []
        for t in types:
            sub = [r for r in rows if r["type"] == t]
            cells.append(f"{statistics.fmean(r['oracle_value'] for r in sub):.3f} (n={len(sub)})" if sub else "-")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\n### oracle_value by side\n")
    print("| row | USSR | USA |")
    print("|---|---|---|")
    for name, rows in runs.items():
        cells = []
        for s in ("USSR", "USA"):
            sub = [r for r in rows if r["side"] == s]
            cells.append(fmt(sub, "oracle_value"))
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
