"""Shared featurisation for the value network: ts-env's encoding plus hand-crafted auxiliary
features distilled from the human strategy notes (a set of human strategy notes).

    featurize(game, side, aux=True) -> float32 vector of length 3081 (+ AUX_DIM)

The aux block is computed from the Observation and GameState, never from hidden information
the observer could not see (hand sizes, discard pile and the observer's own hand are public or
own knowledge). Keep AUX_NAMES in sync with the vector; the trainer stores AUX_DIM in the checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import tsenv  # noqa: E402,F401  (puts the ts-env submodule on sys.path)
from baselines import degrades_defcon  # noqa: E402
from twilight.data import CARDS, COUNTRIES  # noqa: E402
from twilight.encode import encode, flatten  # noqa: E402
from twilight.engine import HAND_SIZE  # noqa: E402
from twilight.enums import Region, Side  # noqa: E402
from twilight.observe import observe  # noqa: E402

SCORING_REGIONS = [r for r in Region if any(c.scoring_region is r for c in CARDS.values())]
_DEADLY = {n for n, c in CARDS.items() if degrades_defcon(n)}

AUX_NAMES = (
    'scoring_in_hand', 'ars_left', 'scoring_minus_ars', 'scoring_danger', 'scoring_last_ar',
    'held_scoring_net_vp_sum', 'held_scoring_net_vp_min',
    'overcontrol_total', 'overcontrol_countries', 'isolated_countries',
    'milops_gap', 'milops_gap_last_ar', 'milops_gap_no_ar',
    'deadly_opp_cards_in_hand', 'deadly_neutral_cards_in_hand', 'deadly_x_defcon_low',
    'reshuffle_next_turn', 'coupable_bg_opp_controlled', 'coupable_bg_mine_exposed',
) + tuple(f'scoring_played_{r.value}' for r in SCORING_REGIONS)
AUX_DIM = len(AUX_NAMES)


def aux_features(obs, state) -> np.ndarray:
    me, opp = obs.player, obs.opponent
    out = []
    # scoring cards in hand vs action rounds left this turn (headline counts as round 0)
    held_scoring = [n for n in obs.hand if CARDS[n].is_scoring]
    s = len(held_scoring)
    ars_left = max(0, state.action_rounds_this_turn - obs.action_round)
    out += [s / 3.0, ars_left / 8.0, (s - ars_left) / 3.0, 1.0 if s > ars_left else 0.0,
            1.0 if (s >= 1 and ars_left <= 1) else 0.0]
    nets = [obs.region(CARDS[n].scoring_region).net_vp_for_observer for n in held_scoring if CARDS[n].scoring_region]
    nets = [max(-12, min(12, v)) for v in nets]
    out += [sum(nets) / 12.0 if nets else 0.0, min(nets) / 12.0 if nets else 0.0]
    # over-control (influence beyond what control needs) and isolated countries
    over_total = over_n = isolated = 0
    mine_by = {v.name: v.mine(me) for v in obs.countries}
    coupable_bg_opp = coupable_bg_mine = 0
    for v in obs.countries:
        mine, theirs = v.mine(me), v.theirs(me)
        extra = mine - theirs - v.stability
        if extra > 0:
            over_total += extra; over_n += 1
        if mine > 0 and v.can_coup and all(mine_by.get(a, 0) == 0 for a in COUNTRIES[v.name].adjacent):
            isolated += 1
        if v.battleground and v.can_coup:
            if v.controller is opp:
                coupable_bg_opp += 1
            if v.controller is me and mine - theirs <= v.stability:
                coupable_bg_mine += 1
    out += [over_total / 20.0, over_n / 10.0, isolated / 10.0]
    # military operations shortfall
    gap = max(0, obs.military_ops_required - obs.military_ops[me])
    out += [gap / 5.0, 1.0 if (gap > 0 and ars_left <= 1) else 0.0, 1.0 if (gap > 0 and ars_left == 0) else 0.0]
    # cards in hand whose event can lower DEFCON: the opponent's fire when played for ops
    deadly_opp = sum(1 for n in obs.hand if n in _DEADLY and CARDS[n].side is opp)
    deadly_neutral = sum(1 for n in obs.hand if n in _DEADLY and CARDS[n].side is None)
    out += [deadly_opp / 4.0, deadly_neutral / 4.0, 1.0 if (deadly_opp > 0 and obs.defcon <= 2) else 0.0]
    # reshuffle before next turn's deal
    need = 2 * (HAND_SIZE[max(state.stages_in_deck, key=lambda st: st.value)] if state.stages_in_deck else 8)
    out += [1.0 if obs.deck_size < need else 0.0, coupable_bg_opp / 6.0, coupable_bg_mine / 6.0]
    # which regions' scoring cards are already in the discard pile (public)
    discard = set(obs.discard)
    for r in SCORING_REGIONS:
        out.append(1.0 if any(CARDS[n].scoring_region is r for n in discard if CARDS[n].is_scoring) else 0.0)
    return np.asarray(out, dtype=np.float32)


def featurize(game, side: Side, aux: bool = True) -> np.ndarray:
    obs = observe(game.state, side, game.decision)
    base = flatten(encode(obs))
    if not aux:
        return base
    return np.concatenate([base, aux_features(obs, game.state)])


if __name__ == '__main__':
    from twilight import Game
    g = Game(seed=3)
    for _ in range(60):
        g.step(g.decision.options[0])
    x = featurize(g, Side.USSR)
    print('dim', x.shape, 'aux', AUX_DIM, dict(zip(AUX_NAMES, np.round(x[-AUX_DIM:], 3))))
