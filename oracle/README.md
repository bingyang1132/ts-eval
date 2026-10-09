# oracle: the learned value network V and the reference agents

**V(s) = P(observer wins | s)**, a 2-layer MLP over ts-env's 3,081-dimensional observation encoding
(`twilight.encode.flatten(encode(observe(state, side)))`), trained on the outcomes of generated games. It does
two jobs here:

1. **Oracle of the single-step task.** Every legal option of every decision in `data/` is labelled with V
   (`relabel_with_value.py`), and the task's reward is the chosen option's normalised V. The leaderboard uses
   the public oracle defined below.
2. **Reference opponents for full games.** `value_agent.py` plays one-ply value-greedy (pick the option with
   the highest V after it), optionally behind two expert hard rules (B1, B2 below). Value-greedy + rules is the
   second reference line of the leaderboard, far above the greedy baseline.

Shipped checkpoint: `value.pt` (round 2, 7.4 MB, load with `torch.load(..., weights_only=False)` or
`value_model.Value`) and its training report `report.json`.

## The public oracle (exact definition)

The label of option *a* at decision *d* (mover *m*, game seed *g*, step *t*) in
`data/oracle_heldout_value_r2_public.jsonl`:

1. Clone the game (the clone replays `(g, history)`), reseed the clone's game RNG with
   `blake2b(repr(("ts-eval-v1", "dice", g, t, a, j)))` (8-byte digest) and the greedy completer's RNG with the same
   hash over `"completer"`, apply *a*, then let ts-env's greedy baseline answer the mover's remaining consecutive
   sub-decisions (at most 12).
2. If the game ended, take the exact outcome (1 / 0.5 / 0); else V round 2's probability that *m* wins
   (`oracle/value.pt`, read from *m*'s observation).
3. If the evaluation for j = 0 consumed the game RNG (a die roll or a reshuffle) or the completer broke a tie,
   repeat for j = 1..31 and average the 32 values; otherwise the single value is the label.

This is `ValueGreedyAgent(dice='expect', k=32, completer='hashed', salt='ts-eval-v1').option_values`, the
default of `relabel_with_value.py`. It is a pure function of (position, option, salt): two runs with different
worker scheduling gave 847 of 847 identical records. Cost: about 4.5 minutes for the 847 held-out decisions on 32
CPU workers. Rebuilding the positions needs ts-env `515622c`, the `third_party/ts-env-legacy` submodule, which
`relabel_with_value.py` selects itself (`tsenv.py`).

## Caveats

- **Training labels use an earlier oracle definition.** The training files (`data/oracle_train_value_r2.jsonl`,
  the big training set) and `data/oracle_heldout_value_r2_legacy.jsonl` are labelled with the same V round 2 and
  the same completion, but the look-ahead clone replays the real seed, so it sees the real dice, and the greedy
  completer's tie-breaks come from one unseeded random stream per labelling worker. These labels are not
  reproducible bit for bit. The published LLM adapters were trained on them
  (`relabel_with_value.py --dice peek --completer shared --salt ''`).
- **The public held-out labels** (`data/oracle_heldout_value_r2_public.jsonl`, the leaderboard) have no dice peek
  and are reproducible bit for bit (definition above).
- **Headline look-ahead.** The engine asks the two headlines one after the other, and the second chooser's
  observation does not reveal the first card, so a language model cannot see it. The look-ahead clones the full
  game state, however, so when the second chooser's options are evaluated, the first chooser's real card resolves
  with them. This is not handled: it affects the labels of 90 of the 847 held-out decisions (second-chooser
  headlines) and every headline the reference agent picks as second chooser in its games.

## Two random sources, quantified (847 held-out decisions)

*agree* = the other labelling's argmax is an argmax of the reference; *ceiling* = `oracle_value` under the
reference of always picking the other's argmax.

| comparison | isolates | agree | ceiling |
|---|---|---|---|
| training-label definition vs a rerun of the same command | completer ties shared across decisions | 0.828 | 0.938 |
| peek vs dice-averaged (k = 8), completer fixed | the dice peek, an upper bound | 0.872 | 0.967 |
| public labels vs training-label definition (`_legacy` file) | both | 0.747 | 0.927 |
| public labels, salt `ts-eval-v1` vs two other salts | remaining Monte Carlo noise of k = 32 | 0.917 / 0.923 | 0.989 / 0.988 |
| public labels, run 1 vs run 2 (different scheduling) | reproducibility | 1.000 | 1.000 |

The dice leak concentrates on coup_target (agree 0.67), the completer ties on place_influence and play_card
(0.67 to 0.70); headline is barely affected by either. Labels are win probabilities, and the expected cost of
either source is small (regret 0.002 to 0.005 in win-probability units), but the argmax is fragile on near-ties,
which is why `oracle_value`, not `top1`, is the headline metric. The headline look-ahead (Caveats) is not one of
these two sources and remains in the public labels.

## Is V a valid oracle? (the validity test)

A value function that ranks moves well should win games when played greedily. 200 games against ts-env's
greedy baseline (100 per side, seeds 7000 to 7099, top-10 pruning, `value_agent.py --opponent greedy
--max-seconds 3600`), Wilson 95% intervals, on ts-env `6791b34` (0 of 1,000 games aborted):

| agent vs greedy | **V round 2 (shipped)** | V round 3 (not shipped) |
|---|---|---|
| value-greedy | **76.0% [69.6, 81.4]** | 89.5% [84.5, 93.0] |
| value-greedy + rules B1/B2 (the reference agent), `--dice expect --k 4 --salt a` (no dice peek) | **87.5% [82.2, 91.4]** (USSR 90%, USA 85%) | not run |
| value-greedy + rules B1/B2, `--dice peek` (the peeking variant) | 88.0% [82.8, 91.8] (USSR 91%, USA 85%) | 93.0% [88.6, 95.8] |

V round 1 (measured on ts-env 515622c): 74.5% [68.0, 80.0] without rules, 96.5% with a first,
broader version of B2 that also forbade every coup at DEFCON 2. That version, folded into training labels, taught
the LLM to concede the military-operations VP every turn, so it was narrowed to the B2 below; the round-1 rule
figure is not comparable to the other rule rows.

The reference agent is quoted with `--dice expect --k 4 --salt a`: every option whose evaluation consumes the
game RNG is evaluated on 4 reseeded clones and V is averaged, the dice treatment of the public oracle with fewer
samples (the completer keeps its shared stream). `--dice peek`, the agent's default, lets the look-ahead see the
real die roll; it is within noise of the non-peeking agent (table above; paired 22 : 18 games, p = 0.64) and is
kept as the comparison row, an upper bound of a fair one-ply agent. The value-greedy rows without rules, the
V round 3 rows and the opponent of the second reference line (`eval/play_vs_greedy.py --opponent value_rules`) use
the peeking agent.

**V round 3 and why the labels stay on V round 2.** V round 3 adds 10,000 games in which V round 2 + rules held
half the seats (3.33 M positions from 118,865 games; held-out AUC 0.774 / log-loss 0.574, but 0.774 vs 0.770 for
V round 2 on the same held-out positions, the gain sitting on the round-3 games). It is
clearly the better player without rules (89.5% vs 76.0%, paired 41 : 14 discordant games, p = 0.0004) and
better but not significantly with rules (both peeking, 5 points apart, 23 : 13, p = 0.13). The public oracle nevertheless stays
on V round 2: every published LLM was trained on V round 2 labels, the reference agent that defines the second
reference line is V round 2 + rules, and with rules, the configuration that matters for both roles, the two
rounds are not separated at 200 games. A V round 3 oracle would be a new, versioned label set (a new salt and
file name), not a silent replacement.

The test is load-sensitive: games have a wall-clock cap (`--max-seconds`) and an aborted game counts as a
non-win. Use `--max-seconds 3600`.

## How V round 2 was trained

| stage | games | policy mixture | positions recorded | command |
|---|---|---|---|---|
| round 1b | 100,000 (10 shards, seeds 200000 + 10000 i) | greedy 40%, eps-greedy 0.1 25%, eps-greedy 0.3 20%, safe-random 15% | 4% of decision states, both perspectives | `gen_positions.py --games 10000 --seed <s> --p 0.04 --out positions/round1b_shard<i>` |
| V round 1 | | | | `train_value.py --shards 'positions/round1b_shard*.npz' --out checkpoints/value_r1_all --features all --hidden 512 --depth 2 --dropout 0.3 --wd 1e-3 --epochs 4` |
| round 2 | 10,000 (seed 400000) | V round 1 value-greedy (top-10 pruning) in half the seats, the mixture above in the rest | 8% | `gen_positions.py --games 10000 --seed 400000 --p 0.08 --value checkpoints/value_r1_all/value.pt --out positions/round2_shard0` |
| **V round 2** | 109,040 games, 2,824,738 positions | | | `train_value.py --shards 'positions/round1b_shard*.npz' 'positions/round2_shard*.npz' --out checkpoints/value_r2 --features all --hidden 512 --depth 2 --dropout 0.3 --wd 1e-3 --epochs 3` |

Training details (`train_value.py`): terminal states are dropped (the encoding cannot tell which side caused
DEFCON 1; at use time a finished game is read from the rules instead), the held-out split is by game seed
(`seed % 10 == 0`), inputs are standardised, AdamW with one-cycle LR 1e-3, batch 2048, the epoch with the best
held-out log-loss is kept. Round 2 took 3.3 minutes on CPU. Generating the positions is the expensive part (the
value-greedy seats clone the game once per option): round 2 took 9.8 h on 56 workers. The position shards are
not shipped (about 19 GB for round 1b + round 2); the commands above regenerate them.

V round 2 held-out quality (`report.json`, 278,520 positions from about 10,900 held-out games):

| | all non-terminal | turns 1-2 | 3-4 | 5-6 | 7-10 |
|---|---|---|---|---|---|
| AUC | 0.761 | 0.716 | 0.792 | 0.794 | 0.802 |
| log-loss | 0.583 | 0.620 | 0.552 | 0.552 | 0.547 |

Calibration is good in the middle and slightly overconfident at the ends (predicted 0.06 -> observed 0.08,
0.94 -> 0.92). Hand-crafted auxiliary features (`features.py`, `aux_features`, 26 dims) did not change AUC
(0.7604 with vs 0.7607 without on the same 100k games) and are not used by the shipped checkpoint (`aux: false`).

## Using V on a position: option values

One-ply V alone misreads multi-step actions: after `use:coup` at DEFCON 2 the state still looks fine to V,
although every coup target left is a battleground and the next click loses. `ValueGreedyAgent.option_values`
therefore applies the option on a clone, completes the mover's remaining consecutive sub-decisions with the
greedy baseline (at most 12 steps), and only then reads V; if the game ended on the way, the exact outcome
(1 / 0.5 / 0) replaces V. The two random sources inside this evaluation (dice, completer ties) are handled as
described under "The public oracle".

## Expert rules B1 and B2 (exact definitions)

Applied by `ValueGreedyAgent.apply_rules` (and identically by `eval/play_vs_greedy.py --rule-mask` for an LLM)
as a filter on the legal menu before choosing. `ars_left = action_rounds_this_turn - action_round` (headline
phase counts as round 0). `degrades_defcon(card)` is ts-env's `examples/baselines.py` check: the card's
effect specification contains `DegradeDEFCONLevel` or `SetDEFCONLevel`.

| decision type | condition | allowed options |
|---|---|---|
| **B1** `play_card` | the menu offers scoring cards and their number >= `ars_left` | only the scoring cards (no room left to delay one; holding a scoring card at turn end loses the game) |
| **B2** `play_card` | DEFCON <= 2 (and B1 did not fire) | every option except opponent-side cards whose event can lower DEFCON (playing them for operations fires the event, and the phasing player is blamed) |
| **B2** `headline` | DEFCON <= 2 | every option except cards (of either side or neutral) whose event can lower DEFCON |
| **B2** `card_use` | DEFCON <= 2 and the card being played can lower DEFCON and is not the mover's own card | only the space-race use, if it is legal |
| **B2** `coup_target` | DEFCON <= 2 | every option except battleground countries (a battleground coup lowers DEFCON by one) |

Coups as such stay allowed at DEFCON 2: a non-battleground coup is legal and often the only way to meet the
military-operations requirement. If a filter would leave nothing, the full menu is kept.

## Files

| file | role |
|---|---|
| `value.pt`, `report.json` | V round 2 checkpoint (state dict, input mean/std, architecture, feature indices) and training report |
| `value_model.py` | `Value(path)`: batch inference on feature vectors; `of_game(game, side)` for a live game |
| `train_value.py` | `ValueMLP` and the trainer (above) |
| `features.py` | `featurize(game, side, aux)`: ts-env's encoding, optionally plus the aux block |
| `value_agent.py` | `ValueGreedyAgent` (option values, rules B1/B2, top-k pruning) and a full-game runner against ts-env's baselines, eps-greedy or another value agent |
| `gen_positions.py` | generates (position, outcome) shards from mixed-policy games |
| `label_oracle.py` | candidate-decision sampler (`gen`), state replay (`rebuild`), `EpsGreedy`; its rollout `label` phase is the superseded oracle (two independent k=64 labellings correlated at r = 0.22 within a decision) |
| `relabel_with_value.py` | labels candidates with V: the producer of `data/*.jsonl` |
| `tsenv.py` | puts the ts-env submodule of the task (games or data commit) and its `examples/` folder first on `sys.path`, checks the commit |

## Commands

```bash
# from the ts-eval root; torch (CPU is enough); ts-env comes from the submodules (top-level README)
python -c "import sys; sys.path.insert(0, 'oracle'); from value_model import Value; Value('oracle/value.pt'); print('ok')"

# validity test on the games commit (200 games, a few minutes on 32 workers)
python oracle/value_agent.py --value oracle/value.pt --games 100 --opponent greedy --workers 32 --prune 10 --max-seconds 3600 --out results/vgreedy_vs_greedy.jsonl
python oracle/value_agent.py --value oracle/value.pt --games 100 --opponent greedy --workers 32 --prune 10 --rules --max-seconds 3600 --out results/vgreedy_rules_vs_greedy.jsonl
python oracle/value_agent.py --value oracle/value.pt --games 100 --opponent greedy --workers 32 --prune 10 --rules --dice expect --k 4 --salt a --max-seconds 3600 --out results/vgreedy_rules_exp4_vs_greedy.jsonl   # the reference agent

# the public held-out labels (uses the data-commit submodule; about 5 min on 32 workers)
python oracle/relabel_with_value.py --value oracle/value.pt --candidates data/oracle_heldout_value_r2_public.jsonl --out results/labels.jsonl --workers 32
# new candidate decisions instead, on the games commit (the shipped ones came from 515622c)
TS_ENV_TASK=games python oracle/label_oracle.py gen --games 300 --seed 1000 --out results/cands.jsonl
```
