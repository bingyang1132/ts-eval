"""One-ply value-greedy agent and a full-game runner against the baselines.

    python oracle/value_agent.py --value oracle/value.pt --games 100 --opponent greedy --workers 24
    python oracle/value_agent.py --value a.pt --rules --opponent value --opponent-value b.pt --opponent-rules   # V vs V
    python oracle/value_agent.py --value oracle/value.pt --rules --opponent eps_greedy --opponent-eps 0.10
Each legal option is applied on a clone, the mover's V of the resulting state is read, argmax wins (random
tie-break). Also usable as a library: ValueGreedyAgent(value_path).

Two random sources inside one option evaluation (oracle/README.md):
  dice       the clone replays the real seed, so by default ('peek') its next die roll / reshuffle is the one
             the real game will produce. 'independent' reseeds the clone's state.rng with
             hash(salt, game seed, step, option key); 'expect' averages V over k such seeds for every option
             whose evaluation consumed the game RNG (dice, reshuffles), one evaluation otherwise.
  completer  the greedy completer breaks score ties at random. 'shared' (default, the behaviour all labels up to
             10-07 were made with) is one GreedyAgent(completer_seed) whose RNG carries over across options and
             decisions; 'hashed' reseeds it per (salt, game seed, step, option key, sample), so a label is a
             function of the position alone. With dice='expect' and completer='hashed' the k samples also vary
             the completer seed whenever the completer hit a tie (expect_completer=True).
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, random, statistics, sys, time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import tsenv  # noqa: E402,F401  (puts the ts-env submodule on sys.path)
from baselines import AGENTS  # noqa: E402
from twilight import Game, Side  # noqa: E402
from twilight.encode import encode, flatten  # noqa: E402
from twilight.observe import observe  # noqa: E402
from twilight.record import play_game  # noqa: E402
from value_model import Value  # noqa: E402
from features import featurize  # noqa: E402
from baselines import degrades_defcon  # noqa: E402
from twilight.data import CARDS, COUNTRIES  # noqa: E402
from twilight.decisions import ActionKind, DecisionType  # noqa: E402
from twilight.enums import OpsUse  # noqa: E402


def _h(*parts) -> int:
    """Stable 64-bit seed from any repr-able parts (Python's hash() is salted per process)."""
    return int.from_bytes(hashlib.blake2b(repr(parts).encode(), digest_size=8).digest(), 'big')


class _TieRng(random.Random):
    """random.Random that records whether choice() was asked to pick among more than one item."""
    tie = False

    def choice(self, seq):
        if len(seq) > 1:
            self.tie = True
        return super().choice(seq)


class ValueGreedyAgent:
    name = 'value_greedy'

    def __init__(self, value_path: str, seed: int = 0, device: str = 'cpu', prune: int | None = None, rules: bool = False,
                 dice: str = 'peek', k: int = 4, completer: str = 'shared', completer_seed: int = 0, salt: str = '',
                 expect_completer: bool = True):
        self.value = Value(value_path, device)
        self.rng = random.Random(seed)
        if dice not in ('peek', 'independent', 'expect'):
            raise ValueError(f'dice must be peek, independent or expect, not {dice!r}')
        if completer not in ('shared', 'hashed'):
            raise ValueError(f'completer must be shared or hashed, not {completer!r}')
        self.dice, self.k, self.completer, self.completer_seed = dice, max(1, int(k)), completer, completer_seed
        self.salt, self.expect_completer = salt, expect_completer
        #: per option of the last option_values call: (consumed the game RNG, completer broke a tie), first sample
        self.last_flags: list[tuple[bool, bool]] = []
        #: Expert hard rules (B1, B2; see oracle/README.md): never end a turn holding a scoring
        #: card, never let a DEFCON-lowering event of the opponent fire at DEFCON <= 2.
        self.rules = rules
        #: In full games, evaluate V only on the `prune` options the greedy heuristic rates highest
        #: (plus greedy's own pick). Each option costs a clone that replays the whole history, so a
        #: 40-country influence decision late in a game is otherwise seconds of work. The oracle
        #: labeller never prunes: every legal option needs a value.
        self.prune = prune

    #: After applying an option, the mover's remaining consecutive decisions (pick a target, how
    #: many, confirm...) are completed by the greedy heuristic before V is read, up to this many
    #: atomic steps. Without it a one-ply V cannot see that "use:coup" at DEFCON 2 loses on the
    #: very next click when every target is a battleground.
    COMPLETE_STEPS = 12

    def _completer(self):
        if not hasattr(self, '_greedy'):
            from baselines import GreedyAgent
            self._greedy = GreedyAgent(seed=self.completer_seed)
            self._greedy.rng = _TieRng(self.completer_seed)  # same stream as random.Random(seed), records ties
        return self._greedy

    def _eval_option(self, game: Game, a, mover, dice_seed, comp_seed):
        """One evaluation: clone, (re)seed, apply, let greedy finish the mover's chain. Returns
        (features, exact outcome or None, consumed the game RNG, completer broke a tie)."""
        comp = self._completer()
        c = game.clone()
        if dice_seed is not None:
            c.state.rng.seed(dice_seed)
        if comp_seed is not None:
            comp.rng.seed(comp_seed)
        comp.rng.tie = False
        s0 = c.state.rng.getstate()
        c.step(a)
        n = 0
        while c.decision is not None and c.decision.player is mover and n < self.COMPLETE_STEPS:
            c.step(comp.act(c, c.decision)[0]); n += 1
        exact = None
        if c.decision is None:  # the game ended: exact outcome, not the net
            w = c.state.winner
            exact = 0.5 if w is None else (1.0 if w is mover else 0.0)
        return featurize(c, mover, aux=self.value.aux), exact, c.state.rng.getstate() != s0, comp.rng.tie

    def _seeds(self, game: Game, key: str, j: int):
        step = len(game.history)
        dice_seed = None if self.dice == 'peek' else _h(self.salt, 'dice', game.seed, step, key, j)
        if self.completer == 'shared':
            comp_seed = None
        else:
            comp_seed = _h(self.salt, 'completer', game.seed, step, key, j if self.expect_completer else 0)
        return dice_seed, comp_seed

    def option_values(self, game: Game, decision, options=None) -> np.ndarray:
        mover = decision.player
        options = list(decision.options) if options is None else options
        feats, exact, owner = [], {}, []  # owner[f] = option index of feature row f
        self.last_flags = []
        for i, a in enumerate(options):
            f, x, used_rng, tie = self._eval_option(game, a, mover, *self._seeds(game, a.key, 0))
            self.last_flags.append((used_rng, tie))
            samples = [(f, x)]
            stochastic = used_rng or (tie and self.completer == 'hashed' and self.expect_completer)
            if self.dice == 'expect' and stochastic:
                for j in range(1, self.k):
                    f, x, _, _ = self._eval_option(game, a, mover, *self._seeds(game, a.key, j))
                    samples.append((f, x))
            for f, x in samples:
                if x is not None:
                    exact[len(feats)] = x
                feats.append(f); owner.append(i)
        v = self.value(np.stack(feats))
        for r, x in exact.items():
            v[r] = x
        owner = np.asarray(owner)
        return np.array([float(v[owner == i].mean()) for i in range(len(options))])

    def apply_rules(self, game: Game, decision, options):
        state = game.state; me = decision.player
        ars_left = max(0, state.action_rounds_this_turn - state.action_round)
        if decision.type is DecisionType.PLAY_CARD:
            scoring = [a for a in options if a.kind is ActionKind.CARD and CARDS[a.value].is_scoring]
            if scoring and len(scoring) >= ars_left:  # B1: no room left to delay a scoring card
                return scoring
            if state.defcon <= 2:  # B2: do not play the opponent's DEFCON-lowering cards for ops
                safe = [a for a in options if not (a.kind is ActionKind.CARD and CARDS[a.value].side is me.opponent and degrades_defcon(a.value))]
                return safe
        if decision.type is DecisionType.HEADLINE and state.defcon <= 2:
            return [a for a in options if not (a.kind is ActionKind.CARD and degrades_defcon(a.value))]
        if decision.type is DecisionType.CARD_USE and state.defcon <= 2:
            card = state.playing_card
            if card is not None and degrades_defcon(card) and CARDS[card].side is not me:
                return [a for a in options if a.value == OpsUse.SPACE.value] or options
            # Coups themselves are left to V: a non-battleground coup at DEFCON 2 is legal and often the
            # only way to meet the military-ops requirement. Forbidding every coup here (the first
            # version of B2) taught the LLM to concede those VP every turn: 23.5% vs 51% in full games.
        if decision.type is DecisionType.COUP_TARGET and state.defcon <= 2:
            safe = [a for a in options if a.kind is not ActionKind.COUNTRY or not COUNTRIES[a.value].battleground]
            return safe
        return options

    def act(self, game: Game, decision):
        if len(decision.options) == 1:
            return decision.options[0], None
        options = list(decision.options)
        if self.rules:
            options = self.apply_rules(game, decision, options) or options
        if self.prune and len(options) > self.prune:
            comp = self._completer()
            scored = sorted(options, key=lambda a: comp._value(game, decision, decision.player, a), reverse=True)
            options = scored[:self.prune]
        v = self.option_values(game, decision, options)
        best = float(v.max()); tied = [a for a, x in zip(options, v) if x >= best - 1e-9]
        a = self.rng.choice(tied)
        return a, f'V={best:.3f}, min {float(v.min()):.3f}, {len(tied)} tied' if len(tied) > 1 else f'V={best:.3f}, min {float(v.min()):.3f}'


_ARGS = None


def _init(args):
    global _ARGS
    _ARGS = args
    torch.set_num_threads(1)


def make_opponent(args, seed):
    """The baselines, plus eps-greedy (oracle-dataset policy) and a second value-greedy agent (V vs V)."""
    if args.opponent == 'value':
        return ValueGreedyAgent(args.opponent_value, seed=seed, prune=args.prune, rules=args.opponent_rules)
    if args.opponent == 'eps_greedy':
        from label_oracle import EpsGreedy
        return EpsGreedy(seed=seed, eps=args.opponent_eps)
    return AGENTS[args.opponent](seed=seed)


def opponent_label(args):
    if args.opponent == 'value':
        return f"value_greedy({Path(args.opponent_value).parent.name}{', rules' if args.opponent_rules else ''})"
    if args.opponent == 'eps_greedy':
        return f'eps_greedy({args.opponent_eps:g})'
    return args.opponent


def one_game(job):
    seed, side_label = job
    me = Side.from_label(side_label); opp = Side.USA if me is Side.USSR else Side.USSR
    agents = {me: ValueGreedyAgent(_ARGS.value, seed=seed, prune=_ARGS.prune, rules=_ARGS.rules, dice=_ARGS.dice, k=_ARGS.k,
                                   completer=_ARGS.completer, salt=_ARGS.salt), opp: make_opponent(_ARGS, seed + 9999)}
    t0 = time.perf_counter()
    from twilight.record import call_agent
    game = Game(seed=seed); n = 0
    while game.decision is not None and n < _ARGS.max_steps and time.perf_counter() - t0 < _ARGS.max_seconds:
        d = game.decision
        game.step(call_agent(agents[d.player], game, d)[0]); n += 1
    aborted = game.decision is not None
    w = game.state.winner
    winner = None if aborted else ('draw' if w is None else w.label)
    reason = 'aborted' if aborted else (game.state.win_reason.value if game.state.win_reason else 'draw')
    return {'seed': seed, 'side': side_label, 'winner': winner, 'win_reason': reason,
            'final_turn': game.state.turn, 'steps': n, 'won': winner == side_label,
            'draw': winner == 'draw', 'aborted': aborted, 'seconds': round(time.perf_counter() - t0, 1),
            'last_actions': game.history[-8:] if aborted else []}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--value', required=True)
    ap.add_argument('--games', type=int, default=100, help='per side')
    ap.add_argument('--seed', type=int, default=7000)
    ap.add_argument('--opponent', default='greedy', choices=sorted(AGENTS) + ['eps_greedy', 'value'])
    ap.add_argument('--opponent-value', default=None, help='with --opponent value: the opponent value.pt')
    ap.add_argument('--opponent-rules', action='store_true', help='with --opponent value: the opponent applies B1/B2')
    ap.add_argument('--opponent-eps', type=float, default=0.10, help='with --opponent eps_greedy')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--out', default=None)
    ap.add_argument('--rules', action='store_true', help='apply the expert hard rules B1/B2 before V')
    ap.add_argument('--max-steps', type=int, default=2500)
    ap.add_argument('--max-seconds', type=float, default=1200)
    ap.add_argument('--prune', type=int, default=10, help='evaluate V on at most this many greedy-ranked options (0 = all)')
    ap.add_argument('--dice', default='peek', choices=['peek', 'independent', 'expect'], help='random stream of the look-ahead clone (see module docstring)')
    ap.add_argument('--k', type=int, default=4, help='with --dice expect: samples per RNG-dependent option')
    ap.add_argument('--completer', default='shared', choices=['shared', 'hashed'], help='seeding of the greedy completer')
    ap.add_argument('--salt', default='', help='salt of the hashed seeds')
    ap.add_argument('--resume', action='store_true', help='append to --out, skipping the (seed, side) games it already has')
    args = ap.parse_args()
    tsenv.require('games')
    args.prune = args.prune or None
    if args.opponent == 'value' and not args.opponent_value:
        ap.error('--opponent value needs --opponent-value')
    jobs = [(args.seed + i, s) for i in range(args.games) for s in ('USSR', 'USA')]
    rows = []
    if args.resume and args.out and Path(args.out).exists():  # keep the finished (seed, side) rows of a cut run
        rows = [json.loads(l) for l in open(args.out) if l.strip()]
        done = {(r['seed'], r['side']) for r in rows}
        jobs = [j for j in jobs if j not in done]
        print(f'  resume: {len(rows)} games kept, {len(jobs)} to play', flush=True)
    t0 = time.perf_counter()
    out = open(args.out, 'a' if args.resume else 'w') if args.out else None  # rows are written as they finish, so a cut run keeps them
    with mp.get_context('spawn').Pool(args.workers, initializer=_init, initargs=(args,)) as pool:
        for i, r in enumerate(pool.imap_unordered(one_game, jobs), 1):
            rows.append(r)
            if out:
                out.write(json.dumps(r) + '\n'); out.flush()
            if i % 20 == 0 or i <= 4:
                print(f'  {i}/{len(jobs)} games, {(time.perf_counter()-t0)/60:.1f} min, wins so far {sum(x["won"] for x in rows)}', flush=True)
    if out:
        out.close()
    print(f"\nvalue_greedy({Path(args.value).parent.name}{', rules' if args.rules else ''}, dice={args.dice}{f', k={args.k}' if args.dice == 'expect' else ''}, completer={args.completer}) vs {opponent_label(args)}")
    for side in ('USSR', 'USA', 'both'):
        rs = rows if side == 'both' else [r for r in rows if r['side'] == side]
        w = sum(r['won'] for r in rs); n = len(rs); lo, hi = wilson(w, n)
        ends = {}
        for r in rs:
            ends[r['win_reason']] = ends.get(r['win_reason'], 0) + 1
        print(f"{side:<5} n={n} wins {w} ({w/n:.1%}, CI {lo:.1%}-{hi:.1%}) draws {sum(r['draw'] for r in rs)} aborted {sum(r['aborted'] for r in rs)} | "
              f"mean end turn {statistics.fmean(r['final_turn'] for r in rs):.1f} | {statistics.fmean(r['seconds'] for r in rs):.0f} s/game | {ends}")


if __name__ == '__main__':
    main()
