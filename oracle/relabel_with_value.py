"""Relabel the single-step candidate decisions with the value net: value(option) = V(mover, state after option).

    python oracle/label_oracle.py gen --games 300 --seed 1000 --out cands.jsonl       # candidate decisions
    python oracle/relabel_with_value.py --value oracle/value.pt --candidates cands.jsonl --out labels.jsonl --workers 32
Output has the same schema as label_oracle.py so the verifiers env, eval scripts and trainers read it unchanged.
The defaults are the public oracle of data/oracle_heldout_value_r2_public.jsonl: dice=expect, k=32,
completer=hashed, salt "ts-eval-v1" (a label is then a pure function of position, option and salt).
--dice peek --completer shared --salt '' reproduces the behaviour of the legacy labels (dice visible to the
look-ahead, one completer RNG shared across decisions; not reproducible bit for bit). Each option also gets
'rng' / 'tie' flags (its first evaluation consumed the game RNG / the completer broke a tie).
Rebuilding positions needs ts-env at the data commit 515622c (third_party/ts-env-legacy; oracle/tsenv.py selects it).
"""
from __future__ import annotations

import argparse, json, sys, time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import label_oracle as lo  # noqa: E402
import tsenv  # noqa: E402
from twilight.render import render  # noqa: E402
from twilight.observe import observe  # noqa: E402
from value_agent import ValueGreedyAgent  # noqa: E402

_AGENT = None
_TAG = ''


_RULES = False


_CFG = {}


def _init(path, tag, rules=False, cfg=None):
    global _AGENT, _TAG, _RULES, _CFG
    torch.set_num_threads(1)
    _CFG = dict(cfg or {})
    _AGENT = ValueGreedyAgent(path, rules=rules, **_CFG); _TAG = tag; _RULES = rules


def label(cand):
    t0 = time.perf_counter()
    game = lo.rebuild(cand); d = game.decision; mover = d.player
    v = _AGENT.option_values(game, d)
    flags = list(_AGENT.last_flags)
    if _RULES:
        # options the expert hard rules forbid get the decision's minimum value, so the
        # normalised reward of a rule violation is 0 regardless of what V thinks of it
        allowed = set(a.key for a in _AGENT.apply_rules(game, d, list(d.options)))
        if allowed and len(allowed) < len(d.options):
            floor = float(min(v))
            v = [x if a.key in allowed else floor for a, x in zip(d.options, v)]
    options = [{'key': a.key, 'label': a.label, 'value': float(x), 'wins': 0, 'draws': 0, 'rng': bool(fl[0]), 'tie': bool(fl[1])}
               for a, x, fl in zip(d.options, v, flags)]
    best = max(o['value'] for o in options)
    return {
        'id': cand['id'], 'split': cand['split'], 'game_seed': cand['game_seed'], 'step_index': cand['step_index'],
        'pairing': cand['pairing'], 'side': cand['side'], 'turn': cand['turn'], 'action_round': cand['action_round'],
        'decision_type': cand['decision_type'], 'prompt_text': d.prompt,
        'view': render(observe(game.state, mover, d)), 'legal_keys': list(d.legal_keys), 'options': options,
        'best_keys': [o['key'] for o in options if o['value'] >= best - 1e-9],
        'value_max': best, 'value_min': min(o['value'] for o in options), 'k': 1, 'rollout_policy': 'value_net',
        'rollout_eps': 0.0, 'horizon': 'value_net' + ('+rules' if _RULES else ''), 'value_kind': _TAG, 'v_before': _AGENT.value.of_game(game, mover),
        'history': cand['history'], 'label_seconds': round(time.perf_counter() - t0, 2),
        'dice': _CFG.get('dice', 'peek'), 'dice_k': _CFG.get('k', 1) if _CFG.get('dice') == 'expect' else 1,
        'completer': _CFG.get('completer', 'shared'), 'completer_seed': _CFG.get('completer_seed', 0), 'salt': _CFG.get('salt', ''),
        'expect_completer': _CFG.get('expect_completer', True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--value', required=True)
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--rules', action='store_true', help='floor the options the expert hard rules forbid')
    ap.add_argument('--dice', default='expect', choices=['peek', 'independent', 'expect'])
    ap.add_argument('--k', type=int, default=32, help='with --dice expect')
    ap.add_argument('--completer', default='hashed', choices=['shared', 'hashed'])
    ap.add_argument('--completer-seed', type=int, default=0, help='with --completer shared')
    ap.add_argument('--salt', default='ts-eval-v1', help='salt of the hashed seeds')
    ap.add_argument('--no-expect-completer', action='store_true', help='with --dice expect --completer hashed: keep the completer seed of sample 0 (vary the dice only)')
    ap.add_argument('--chunksize', type=int, default=4)
    ap.add_argument('--value-kind', default=None, help="value_kind field of the output (default: value_r2 for oracle/value.pt, else the checkpoint's folder name)")
    args = ap.parse_args()
    tsenv.require('data')
    cands = [json.loads(l) for l in open(args.candidates)]
    if args.limit:
        cands = cands[:args.limit]
    # the shipped checkpoint oracle/value.pt is V round 2; other checkpoints are named by their folder
    tag = args.value_kind or ('value_r2' if Path(args.value).resolve() == (Path(__file__).parent / 'value.pt').resolve() else Path(args.value).parent.name)
    cfg = {'dice': args.dice, 'k': args.k, 'completer': args.completer, 'completer_seed': args.completer_seed, 'salt': args.salt,
           'expect_completer': not args.no_expect_completer}
    t0 = time.perf_counter(); n = 0
    with open(args.out, 'w') as f, mp.get_context('spawn').Pool(args.workers, initializer=_init, initargs=(args.value, tag, args.rules, cfg)) as pool:
        for rec in pool.imap(label, cands, chunksize=args.chunksize):
            f.write(json.dumps(rec) + '\n'); n += 1
            if n % 500 == 0:
                print(f'  {n}/{len(cands)}, {(time.perf_counter()-t0)/60:.1f} min', flush=True)
    print(f'done: {n} decisions -> {args.out} in {(time.perf_counter()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
