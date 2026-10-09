"""Rollout-oracle labelling for the single-step decision task.

Two phases, so the expensive part can be parallelised and resumed:

    python oracle/label_oracle.py gen   --games 400 --out candidates.jsonl
    python oracle/label_oracle.py label --candidates candidates.jsonl --out oracle_k32.jsonl \
                                        --k 32 --workers 64
    python oracle/label_oracle.py stats --data oracle_k32.jsonl

In ts-eval this file is kept for `gen` (the candidate sampler behind the shipped dataset), `rebuild`
(state replay) and `EpsGreedy` (used by gen_positions.py and value_agent.py); the shipped labels come from
the value network (relabel_with_value.py), not from the rollout `label` phase, whose labels were too
noisy (see oracle/README.md).

``gen`` plays games between mixed baseline agents and reservoir-samples decisions that pass
the filters (turn >= 2, 3..MAX_OPTIONS legal options, one of FIVE decision types), keeping the
exact action-key prefix so any state can be rebuilt with ``Game(seed)`` + ``step``.

``label`` rebuilds each sampled state and, for every legal option, plays K rollouts to the
end of the game with an epsilon-greedy policy on both sides. The option's value is the
mover's mean outcome (win 1, draw 0.5, loss 0). Rollout r uses the same agent seeds for every
option (common random numbers), so option values are compared under matched continuations.

Needs ts-env only (oracle/tsenv.py picks the submodule: the data commit by default; TS_ENV_TASK=games
to generate new candidates on the games commit).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tsenv  # noqa: E402,F401  (puts the ts-env submodule on sys.path)

from baselines import GreedyAgent, SafeRandomAgent  # noqa: E402
from twilight import Game, Side  # noqa: E402
from twilight.observe import observe  # noqa: E402
from twilight.record import call_agent  # noqa: E402
from twilight.render import render  # noqa: E402

FIVE_TYPES = ("headline", "play_card", "card_use", "place_influence", "coup_target")
MIN_TURN = 2
MIN_OPTIONS = 3
MAX_OPTIONS = 40
PER_TYPE_PER_GAME = 3
ROLLOUT_EPS = 0.2
MAX_ROLLOUT_STEPS = 5000
HELDOUT_MOD = 5  # game_seed % 5 == 0 -> heldout


class EpsGreedy:
    """Greedy with probability 1-eps, otherwise safe-random. Returns a bare action."""

    name = "eps_greedy"

    def __init__(self, seed: int = 0, eps: float = ROLLOUT_EPS) -> None:
        self.rng = random.Random(seed)
        self.eps = eps
        self.greedy = GreedyAgent(seed=seed + 1)
        self.safe = SafeRandomAgent(seed=seed + 2)

    def act(self, game, decision):
        if self.rng.random() < self.eps:
            return self.safe.act(game, decision)
        return self.greedy.act(game, decision)[0]


POLICIES = {
    "greedy": GreedyAgent,
    "safe_random": SafeRandomAgent,
    "eps_greedy": EpsGreedy,
}
PAIRINGS = [(a, b) for a in POLICIES for b in POLICIES]


def play_to_end(game: Game, agents: dict, max_steps: int = MAX_ROLLOUT_STEPS) -> None:
    n = 0
    while game.decision is not None and n < max_steps:
        d = game.decision
        game.step(call_agent(agents[d.player.label], game, d)[0])
        n += 1


def outcome_for(game: Game, side: Side) -> float:
    w = game.state.winner
    if game.decision is not None or w is None:
        return 0.5
    return 1.0 if w is side else 0.0


# --------------------------------------------------------------------------- #
# gen
# --------------------------------------------------------------------------- #


def gen(args: argparse.Namespace) -> None:
    out = Path(args.out)
    rng = random.Random(args.seed)
    types = None if args.types == 'all' else (FIVE_TYPES if args.types == 'five' else tuple(args.types.split(',')))
    quota = args.per_type
    n_written = 0
    per_type = Counter()
    with out.open("w") as f:
        for i in range(args.games):
            game_seed = args.seed + i
            pa, pb = PAIRINGS[i % len(PAIRINGS)]
            agents = {
                "USSR": POLICIES[pa](seed=game_seed * 7 + 1),
                "USA": POLICIES[pb](seed=game_seed * 7 + 2),
            }
            game = Game(seed=game_seed)
            reservoir: dict[str, list[dict]] = defaultdict(list)
            seen: Counter = Counter()
            step = 0
            while game.decision is not None:
                d = game.decision
                t = str(d.type)
                if (
                    (types is None or t in types)
                    and game.state.turn >= args.min_turn
                    and MIN_OPTIONS <= len(d.options) <= MAX_OPTIONS
                ):
                    cand = {
                        "game_seed": game_seed,
                        "pairing": [pa, pb],
                        "step_index": step,
                        "side": d.player.label,
                        "turn": game.state.turn,
                        "action_round": game.state.action_round,
                        "decision_type": t,
                        "n_options": len(d.options),
                        "history": list(game.history),
                    }
                    seen[t] += 1
                    bucket = reservoir[t]
                    if len(bucket) < quota:
                        bucket.append(cand)
                    else:
                        j = rng.randrange(seen[t])
                        if j < quota:
                            bucket[j] = cand
                game.step(call_agent(agents[d.player.label], game, d)[0])
                step += 1
            for t, bucket in reservoir.items():
                for cand in bucket:
                    cand["id"] = f"g{cand['game_seed']}_s{cand['step_index']}"
                    cand["split"] = "heldout" if game_seed % HELDOUT_MOD == 0 else "train"
                    cand["final_turn"] = game.state.turn
                    f.write(json.dumps(cand) + "\n")
                    n_written += 1
                    per_type[t] += 1
            if (i + 1) % 25 == 0:
                print(f"gen: {i + 1}/{args.games} games, {n_written} candidates", flush=True)
    print(f"gen done: {n_written} candidates -> {out}")
    print("per type:", dict(per_type))


# --------------------------------------------------------------------------- #
# label
# --------------------------------------------------------------------------- #

_K = 32
_ROLLOUT_POLICY = "eps_greedy"
_HORIZON = "game"      # game | turn1 | turn2
_SCORER = GreedyAgent(seed=0)
#: Score given to a game that ends inside a turn-limited horizon, on top of the position score.
TERMINAL_BONUS = 50.0


def _init(k: int, policy: str, horizon: str = "game") -> None:
    global _K, _ROLLOUT_POLICY, _HORIZON
    _K, _ROLLOUT_POLICY, _HORIZON = k, policy, horizon


def play_horizon(game: Game, agents: dict, stop_turn: int | None) -> None:
    """Play until the game ends or ``state.turn`` reaches *stop_turn*."""
    n = 0
    while game.decision is not None and n < MAX_ROLLOUT_STEPS:
        if stop_turn is not None and game.state.turn >= stop_turn:
            return
        d = game.decision
        game.step(call_agent(agents[d.player.label], game, d)[0])
        n += 1


def rollout_value(game: Game, mover: Side, stop_turn: int | None) -> float:
    """Win rate contribution (horizon=game) or greedy position score (turn horizons)."""
    if stop_turn is None:
        return outcome_for(game, mover)
    score = _SCORER.position_score(game, mover)
    if game.decision is None:
        w = game.state.winner
        score += TERMINAL_BONUS if w is mover else (-TERMINAL_BONUS if w is not None else 0.0)
    return score


def rebuild(cand: dict) -> Game:
    game = Game(seed=cand["game_seed"])
    for key in cand["history"]:
        game.step(key)
    d = game.decision
    if d is None or str(d.type) != cand["decision_type"] or d.player.label != cand["side"]:
        raise RuntimeError(f"{cand['id']}: replay mismatch")
    return game


def label_one(cand: dict) -> dict:
    t0 = time.perf_counter()
    game = rebuild(cand)
    d = game.decision
    mover = d.player
    # ORACLE_SEED_OFFSET gives an independent second labelling (test-retest of the oracle).
    base = (hash((cand["game_seed"], cand["step_index"])) & 0xFFFFFF) + int(os.environ.get("ORACLE_SEED_OFFSET", "0"))
    stop_turn = None if _HORIZON == "game" else game.state.turn + int(_HORIZON[-1])
    options = []
    for a in d.options:
        wins = draws = 0
        total = 0.0
        for r in range(_K):
            c = game.clone()
            c.step(a)
            mk = POLICIES[_ROLLOUT_POLICY]
            agents = {
                "USSR": mk(seed=base * 1000 + r),
                "USA": mk(seed=base * 1000 + r + 500),
            }
            play_horizon(c, agents, stop_turn)
            total += rollout_value(c, mover, stop_turn)
            if c.decision is None:
                o = outcome_for(c, mover)
                wins += o == 1.0
                draws += o == 0.5
        value = total / _K
        options.append(
            {"key": a.key, "label": a.label, "value": value, "wins": wins, "draws": draws}
        )
    values = [o["value"] for o in options]
    best = max(values)
    rec = {
        "id": cand["id"],
        "split": cand["split"],
        "game_seed": cand["game_seed"],
        "step_index": cand["step_index"],
        "pairing": cand["pairing"],
        "side": cand["side"],
        "turn": cand["turn"],
        "action_round": cand["action_round"],
        "decision_type": cand["decision_type"],
        "prompt_text": d.prompt,
        "view": render(observe(game.state, mover, d)),
        "legal_keys": list(d.legal_keys),
        "options": options,
        "best_keys": [o["key"] for o in options if o["value"] == best],
        "value_max": best,
        "value_min": min(values),
        "k": _K,
        "rollout_policy": _ROLLOUT_POLICY,
        "rollout_eps": ROLLOUT_EPS,
        "horizon": _HORIZON,
        "value_kind": "winrate" if _HORIZON == "game" else "position_score",
        "history": cand["history"],
        "label_seconds": round(time.perf_counter() - t0, 2),
    }
    return rec


def _safe_label(cand: dict):
    try:
        return label_one(cand)
    except Exception as e:  # noqa: BLE001
        return {"id": cand["id"], "error": repr(e)}


def label(args: argparse.Namespace) -> None:
    cands = [json.loads(l) for l in Path(args.candidates).open()]
    if args.limit:
        cands = cands[: args.limit]
    out = Path(args.out)
    done = set()
    if out.exists():
        for l in out.open():
            try:
                done.add(json.loads(l)["id"])
            except Exception:  # noqa: BLE001
                pass
    todo = [c for c in cands if c["id"] not in done]
    print(f"label: {len(cands)} candidates, {len(done)} done, {len(todo)} to do, "
          f"k={args.k}, workers={args.workers}, policy={args.policy}, horizon={args.horizon}", flush=True)
    t0 = time.perf_counter()
    n = 0
    errs = 0
    with out.open("a") as f, mp.Pool(
        args.workers, initializer=_init, initargs=(args.k, args.policy, args.horizon)
    ) as pool:
        for rec in pool.imap_unordered(_safe_label, todo, chunksize=1):
            if "error" in rec:
                errs += 1
                print(f"  error {rec['id']}: {rec['error']}", flush=True)
                continue
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
            if n % 50 == 0:
                el = time.perf_counter() - t0
                print(f"  {n}/{len(todo)} labelled, {el / 60:.1f} min, "
                      f"{el / n:.1f} s/decision wall, eta {(len(todo) - n) * el / n / 60:.0f} min",
                      flush=True)
    print(f"label done: {n} written, {errs} errors, {(time.perf_counter() - t0) / 60:.1f} min")


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #


def stats(args: argparse.Namespace) -> None:
    recs = [json.loads(l) for l in Path(args.data).open()]
    print(f"{len(recs)} labelled decisions")
    by_type = defaultdict(list)
    for r in recs:
        by_type[r["decision_type"]].append(r)
    print(f"{'type':<17}{'n':>5}{'train':>6}{'held':>6}{'opts':>6}{'tie%':>6}"
          f"{'gap':>6}{'spread':>7}{'sec':>7}")
    for t, rs in sorted(by_type.items()):
        n = len(rs)
        ties = sum(len(r["best_keys"]) > 1 for r in rs)
        gaps = []
        for r in rs:
            vs = sorted((o["value"] for o in r["options"]), reverse=True)
            gaps.append(vs[0] - vs[1])
        spread = [r["value_max"] - r["value_min"] for r in rs]
        print(f"{t:<17}{n:>5}{sum(r['split'] == 'train' for r in rs):>6}"
              f"{sum(r['split'] == 'heldout' for r in rs):>6}"
              f"{statistics.median(len(r['options']) for r in rs):>6.0f}"
              f"{100 * ties / n:>6.0f}{statistics.median(gaps):>6.2f}"
              f"{statistics.median(spread):>7.2f}"
              f"{statistics.median(r['label_seconds'] for r in rs):>7.0f}")
    print("pairings:", dict(Counter(tuple(r["pairing"]) for r in recs)))
    print("turns:", dict(sorted(Counter(r["turn"] for r in recs).items())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--games", type=int, default=400)
    g.add_argument("--seed", type=int, default=1000)
    g.add_argument("--out", default="candidates.jsonl")
    g.add_argument("--types", default="five", help="'five' (default), 'all', or a comma list of decision types")
    g.add_argument("--per-type", type=int, default=PER_TYPE_PER_GAME)
    g.add_argument("--min-turn", type=int, default=MIN_TURN)
    g.set_defaults(fn=gen)
    lb = sub.add_parser("label")
    lb.add_argument("--candidates", default="candidates.jsonl")
    lb.add_argument("--out", required=True)
    lb.add_argument("--k", type=int, default=32)
    lb.add_argument("--workers", type=int, default=64)
    lb.add_argument("--policy", default="eps_greedy", choices=sorted(POLICIES))
    lb.add_argument("--horizon", default="game", choices=["game", "turn1", "turn2"],
                    help="game: play to the end, value = win rate; turnN: stop when the turn "
                         "counter has advanced by N, value = greedy position score")
    lb.add_argument("--limit", type=int, default=0)
    lb.set_defaults(fn=label)
    st = sub.add_parser("stats")
    st.add_argument("--data", required=True)
    st.set_defaults(fn=stats)
    args = ap.parse_args()
    tsenv.require(tsenv.TASK if args.fn is gen else 'data')  # new candidates: TS_ENV_TASK=games
    args.fn(args)


if __name__ == "__main__":
    main()
