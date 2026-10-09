"""Generate (position, outcome) pairs for the value network from mixed-policy games.

Each game: both sides drawn from a policy mixture; every decision state is recorded with probability
--p (both perspectives), terminal states always (both perspectives). A record is the flattened numeric
observation (3081 float16) plus the outcome for the observer (1 / 0.5 / 0) and metadata.

    python oracle/gen_positions.py --games 2000 --seed 100000 --workers 48 --out positions/shard_a
Writes <out>_<worker>.npz pieces and a summary line. Needs ts-env (and torch when --value is given).
"""
from __future__ import annotations

import argparse, json, random, sys, time
import multiprocessing as mp
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import tsenv  # noqa: E402,F401  (puts the ts-env submodule on sys.path)
from baselines import GreedyAgent, SafeRandomAgent  # noqa: E402
from label_oracle import EpsGreedy  # noqa: E402
from twilight import Game, Side  # noqa: E402
from twilight.encode import encode, flatten  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from features import featurize, AUX_DIM  # noqa: E402
from twilight.observe import observe  # noqa: E402
from twilight.record import call_agent  # noqa: E402

# policy mixture: name -> (constructor, weight). Greedy-heavy so positions look like competent play,
# random-heavy enough that DEFCON suicides and held scoring cards appear in the data.
MIX = {
    'greedy': (lambda s: GreedyAgent(seed=s), 0.40),
    'eps10': (lambda s: EpsGreedy(seed=s, eps=0.10), 0.25),
    'eps30': (lambda s: EpsGreedy(seed=s, eps=0.30), 0.20),
    'safe_random': (lambda s: SafeRandomAgent(seed=s), 0.15),
}
POL_INDEX = {n: i for i, n in enumerate(MIX)}
POL_INDEX['value_greedy'] = len(POL_INDEX)
VALUE_PATH = None  # set by --value: adds the value-greedy agent to the mixture (round 2+)
VALUE_RULES = False  # --rules: that agent also applies the expert hard rules


def _value_agent(seed):
    from value_agent import ValueGreedyAgent
    return ValueGreedyAgent(VALUE_PATH, seed=seed, prune=10, rules=VALUE_RULES)


def _init(value_path, aux=False, rules=False):
    global VALUE_PATH, AUX, VALUE_RULES
    VALUE_PATH = value_path; AUX = aux; VALUE_RULES = rules
    if value_path:
        import torch
        torch.set_num_threads(1)
        sys.path.insert(0, str(Path(__file__).parent))
        MIX['value_greedy'] = (_value_agent, 0.5)  # half of all seats; the rest keep their relative weights


def outcome(game: Game, side: Side) -> float:
    w = game.state.winner
    return 0.5 if w is None else (1.0 if w is side else 0.0)


AUX = False


def feat(game: Game, side: Side) -> np.ndarray:
    return featurize(game, side, aux=AUX).astype(np.float16)


def play(job):
    seed, p_record, args_out = job
    rng = random.Random(seed)
    names = list(MIX); weights = [MIX[n][1] for n in names]
    if 'value_greedy' in MIX:
        rest = sum(MIX[n][1] for n in names if n != 'value_greedy')
        weights = [MIX[n][1] if n == 'value_greedy' else MIX[n][1] * 0.5 / rest for n in names]
    pa, pb = rng.choices(names, weights)[0], rng.choices(names, weights)[0]
    agents = {'USSR': MIX[pa][0](seed * 7 + 1), 'USA': MIX[pb][0](seed * 7 + 2)}
    game = Game(seed=seed)
    cap = 2500  # safety cap: cut a game rather than hang a worker
    X, meta = [], []  # meta: (seed, step, turn, perspective(0 USSR/1 USA), to_move, pol_a, pol_b, is_terminal)
    step = 0
    while game.decision is not None and step < cap:
        d = game.decision
        if rng.random() < p_record:
            for side in (Side.USSR, Side.USA):
                X.append(feat(game, side))
                meta.append((seed, step, game.state.turn, int(side is Side.USA), int(d.player is Side.USA),
                             POL_INDEX[pa], POL_INDEX[pb], 0))
        game.step(call_agent(agents[d.player.label], game, d)[0])
        step += 1
    for side in (Side.USSR, Side.USA):
        X.append(feat(game, side))
        meta.append((seed, step, game.state.turn, int(side is Side.USA), -1, POL_INDEX[pa], POL_INDEX[pb], 1))
    if game.decision is not None:  # aborted loop: no reliable label, drop the game
        return np.zeros((0, 3081 + (AUX_DIM if AUX else 0)), np.float16), np.zeros(0, np.float32), np.zeros((0, 8), np.int32), (seed, pa, pb, step, game.state.turn, 'aborted')
    y = np.array([outcome(game, Side.USA if m[3] else Side.USSR) for m in meta], dtype=np.float32)
    reason = game.state.win_reason.value if game.state.win_reason else 'draw'
    return np.stack(X), y, np.array(meta, dtype=np.int32), (seed, pa, pb, step, game.state.turn, reason)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=200)
    ap.add_argument('--seed', type=int, default=100000)
    ap.add_argument('--p', type=float, default=0.15, help='probability of recording a decision state')
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--out', required=True)
    ap.add_argument('--aux', action='store_true', help='append the hand-crafted aux features (features.py)')
    ap.add_argument('--rules', action='store_true', help='value-greedy seats also apply the expert rules')
    ap.add_argument('--value', default=None, help='value.pt: add the value-greedy agent to the policy mixture')
    args = ap.parse_args()
    tsenv.require('games')
    t0 = time.perf_counter()
    jobs = [(args.seed + i, args.p, args.out) for i in range(args.games)]
    Xs, ys, Ms, summ = [], [], [], []
    ctx = mp.get_context('spawn') if args.value else mp.get_context('fork')
    with ctx.Pool(args.workers, initializer=_init, initargs=(args.value, args.aux, args.rules)) as pool:
        for i, (X, y, M, s) in enumerate(pool.imap_unordered(play, jobs, chunksize=4), 1):
            Xs.append(X); ys.append(y); Ms.append(M); summ.append(s)
            if i % 500 == 0:
                print(f'{i}/{args.games} games, {sum(len(y) for y in ys)} positions, {(time.perf_counter()-t0)/60:.1f} min', flush=True)
    X = np.concatenate(Xs); y = np.concatenate(ys); M = np.concatenate(Ms)
    np.savez(args.out + '.npz', X=X, y=y, meta=M)
    reasons = {}
    for s in summ:
        reasons[s[5]] = reasons.get(s[5], 0) + 1
    info = {'games': args.games, 'positions': int(len(y)), 'terminal': int(M[:, 7].sum()), 'p': args.p,
            'mean_steps': float(np.mean([s[3] for s in summ])), 'mean_final_turn': float(np.mean([s[4] for s in summ])),
            'ussr_win_rate': float(np.mean([y[i] for i in range(len(y)) if M[i, 3] == 0 and M[i, 7] == 1])), 'value_policy': args.value, 'aux': args.aux, 'rules': args.rules,
            'endings': reasons, 'minutes': round((time.perf_counter() - t0) / 60, 1), 'dim': int(X.shape[1])}
    json.dump(info, open(args.out + '.json', 'w'), indent=1)
    print(json.dumps(info))


if __name__ == '__main__':
    main()
