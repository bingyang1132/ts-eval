"""Full games: a served model (base or LoRA adapter name) vs a reference agent, both sides, in parallel.

    python eval/play_vs_greedy.py --model grpo_sft_v2big --base-url http://localhost:8000/v1 \
        --games 100 --workers 12 --out results/games/grpo_sft_v2big.jsonl
    ... --rule-mask                      # filter the menu with the expert rules B1/B2 before the model sees it
    ... --opponent value_rules           # second reference line: value-greedy + rules (oracle/value.pt)
Uses ts-env's LanguageModelAgent (same prompt as the single-step task) against any OpenAI-compatible
chat endpoint; the bearer token is read from $TS_EVAL_TOKEN (default EMPTY, which local vLLM accepts).
Rows are appended as games finish, so a cut run resumes where it stopped.
"""
from __future__ import annotations

import argparse, json, math, os, statistics, sys, time
import urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'oracle'))
import tsenv  # noqa: E402,F401  (puts the ts-env submodule on sys.path)
from baselines import AGENTS  # noqa: E402
from llm_agent import LanguageModelAgent  # noqa: E402
from twilight.data import CARDS, COUNTRIES  # noqa: E402
from twilight.decisions import ActionKind, Decision, DecisionType  # noqa: E402
from twilight.enums import OpsUse  # noqa: E402


def degrades_defcon(name):
    from baselines import degrades_defcon as _d
    return _d(name)


def rule_filter(game, decision):
    """Expert hard rules B1/B2 (narrow form) as a menu filter: the options the model never sees."""
    state = game.state; me = decision.player; options = list(decision.options)
    ars_left = max(0, state.action_rounds_this_turn - state.action_round)
    if decision.type is DecisionType.PLAY_CARD:
        scoring = [a for a in options if a.kind is ActionKind.CARD and CARDS[a.value].is_scoring]
        if scoring and len(scoring) >= ars_left:
            return scoring
        if state.defcon <= 2:
            safe = [a for a in options if not (a.kind is ActionKind.CARD and CARDS[a.value].side is me.opponent and degrades_defcon(a.value))]
            return safe or options
    if decision.type is DecisionType.HEADLINE and state.defcon <= 2:
        return [a for a in options if not (a.kind is ActionKind.CARD and degrades_defcon(a.value))] or options
    if decision.type is DecisionType.CARD_USE and state.defcon <= 2:
        card = state.playing_card
        if card is not None and degrades_defcon(card) and CARDS[card].side is not me:
            return [a for a in options if a.value == OpsUse.SPACE.value] or options
    if decision.type is DecisionType.COUP_TARGET and state.defcon <= 2:
        return [a for a in options if a.kind is not ActionKind.COUNTRY or not COUNTRIES[a.value].battleground] or options
    return options


class RuleMaskedAgent:
    """Wraps the LLM agent: presents a rule-filtered menu, so a forbidden move cannot be chosen."""

    def __init__(self, inner):
        self.inner = inner; self.name = inner.name + '+rules'; self.retries = inner.retries
        self.max_retries = inner.max_retries; self.rationale = None; self.extra = {}

    def act(self, game, decision):
        allowed = rule_filter(game, decision)
        if len(allowed) < len(decision.options):
            decision = Decision(decision.type, decision.player, decision.prompt, tuple(allowed), dict(decision.context))
        key = self.inner.act(game, decision)
        self.rationale = self.inner.rationale; self.extra = self.inner.extra
        return key
from twilight import Side  # noqa: E402
from twilight.record import play_game  # noqa: E402

ARGS = None


def make_opponent(name, seed):
    # 'value_rules' = the value-greedy agent with the expert rules (second reference line);
    # 'value' = the same without rules. Both use the checkpoint in --value.
    if name in ('value', 'value_rules'):
        import torch
        torch.set_num_threads(1)
        from value_agent import ValueGreedyAgent
        return ValueGreedyAgent(ARGS.value, seed=seed, prune=10, rules=(name == 'value_rules'))
    return AGENTS[name](seed=seed)


def _init(args):
    global ARGS
    ARGS = args


def complete(prompt: str) -> str:
    """One chat completion (thinking off), 600 s timeout, up to 5 retries with backoff."""
    body = {'model': ARGS.model, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': ARGS.temperature, 'max_tokens': ARGS.max_tokens,
            'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(
        ARGS.base_url.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json',
                 'Authorization': 'Bearer ' + os.environ.get('TS_EVAL_TOKEN', 'EMPTY')})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)['choices'][0]['message'].get('content') or ''
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == 5 or (isinstance(e, urllib.error.HTTPError) and e.code < 500 and e.code != 429):
                raise
            time.sleep(min(60, 2 ** attempt))


def one_game(job):
    seed, llm_side = job
    llm_side = Side.from_label(llm_side)
    opp_side = Side.USA if llm_side is Side.USSR else Side.USSR
    llm = LanguageModelAgent(complete, name=ARGS.model)
    if ARGS.rule_mask:
        llm = RuleMaskedAgent(llm)
    agents = {llm_side: llm, opp_side: make_opponent(ARGS.opponent, seed + 9999)}
    t0 = time.perf_counter()
    # own loop with a step cap: on ts-env before the realignment fix a handful of games looped inside an event
    from twilight import Game
    from twilight.record import call_agent
    game = Game(seed=seed); steps = 0
    while game.decision is not None and steps < ARGS.max_steps:
        d = game.decision
        game.step(call_agent(agents[d.player], game, d)[0]); steps += 1
    aborted = game.decision is not None
    w = game.state.winner
    class rec:  # minimal stand-in for the GameRecord fields used below
        winner = None if aborted else ('draw' if w is None else w.label)
        win_reason = 'aborted' if aborted else (game.state.win_reason.value if game.state.win_reason else 'draw')
        final_turn = game.state.turn; final_vp = game.state.vp
    rec = rec()
    n = sum(llm.retries.values())
    out = {'seed': seed, 'llm_side': llm_side.label, 'winner': rec.winner, 'win_reason': rec.win_reason,
           'final_turn': rec.final_turn, 'final_vp': rec.final_vp, 'steps': steps, 'aborted': aborted,
           'llm_decisions': n, 'first_try_legal': llm.retries[0], 'fallbacks': llm.retries[llm.max_retries + 1],
           'seconds': round(time.perf_counter() - t0, 1), 'won': rec.winner == llm_side.label,
           'draw': rec.winner == 'draw'}
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summarize(rows):
    print(f"\n{rows[0]['seed'] if rows else ''}")
    for side in ('USSR', 'USA', 'both'):
        rs = rows if side == 'both' else [r for r in rows if r['llm_side'] == side]
        if not rs:
            continue
        w = sum(r['won'] for r in rs); d = sum(r['draw'] for r in rs); n = len(rs)
        lo, hi = wilson(w, n)
        legal = sum(r['first_try_legal'] for r in rs) / max(1, sum(r['llm_decisions'] for r in rs))
        reasons = {}
        for r in rs:
            reasons[r['win_reason']] = reasons.get(r['win_reason'], 0) + 1
        print(f"{side:<5} n={n} wins {w} ({w/n:.1%}, 95% CI {lo:.1%}-{hi:.1%}) draws {d} | mean end turn "
              f"{statistics.fmean(r['final_turn'] for r in rs):.1f} | first-try legal {legal:.1%} | endings {reasons}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--games', type=int, default=100, help='per side')
    ap.add_argument('--seed', type=int, default=5000)
    ap.add_argument('--opponent', default='greedy', help="greedy | safe_random | random | value | value_rules")
    ap.add_argument('--value', default=str(ROOT / 'oracle' / 'value.pt'), help='value net for --opponent value / value_rules')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--base-url', default='http://localhost:8000/v1', help='OpenAI-compatible endpoint')
    ap.add_argument('--temperature', type=float, default=0.6)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-steps', type=int, default=2500)
    ap.add_argument('--rule-mask', action='store_true', help='filter the menu with the expert hard rules B1/B2 before the model sees it')
    args = ap.parse_args()
    tsenv.require('games')
    jobs = [(args.seed + i, s) for i in range(args.games) for s in ('USSR', 'USA')]
    done = set()
    try:
        for l in open(args.out):
            r = json.loads(l); done.add((r['seed'], r['llm_side']))
    except FileNotFoundError:
        pass
    jobs = [j for j in jobs if j not in done]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f'{args.model} vs {args.opponent}: {len(jobs)} games to play ({len(done)} done)', flush=True)
    t0 = time.perf_counter()
    import multiprocessing as mp
    with open(args.out, 'a') as f, mp.get_context('spawn').Pool(args.workers, initializer=_init, initargs=(args,)) as pool:
        for i, r in enumerate(pool.imap_unordered(one_game, jobs), 1):
            f.write(json.dumps(r) + '\n'); f.flush()
            if i % 10 == 0:
                print(f'  {i}/{len(jobs)} games, {(time.perf_counter()-t0)/60:.1f} min', flush=True)
    rows = [json.loads(l) for l in open(args.out)]
    summarize(rows)


if __name__ == '__main__':
    main()
