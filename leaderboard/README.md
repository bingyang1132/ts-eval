# Leaderboard

Status 2026-10-08. Single-step scores are on the 847 held-out decisions (58 games) of
`data/oracle_heldout_value_r2_public.jsonl`, scored against the **public oracle** (V round 2, dice averaged over
32 reseeded look-aheads, hashed completer seeds, salt `ts-eval-v1`; `oracle/README.md`); 95% intervals are
game-cluster bootstrap (2000 resamples). Full-game win rates are over 200 games (100 per side), 95% Wilson
intervals, on ts-env `6791b34` unless a table note says otherwise. Protocols and commands:
`eval/README.md`. Trained adapters are not distributed with this repo; the recipes are in `training/README.md`.

**Ceiling.** The labels are fixed numbers in the data file, so the ceiling of `oracle_value`, `top1` and
`near_best` is 1.0 and of `regret` 0 (always pick a labelled argmax). The oracle's own Monte Carlo noise is
measured by relabelling with two other salts: their argmax scores 0.988 to 0.989 on the public labels, so
differences below about 0.01 are not meaningful at the top.

## Single-step, main table

| rank | row | training | oracle_value [95% CI] | top1 | near_best | regret | legal |
|---|---|---|---|---|---|---|---|
| | ceiling (labelled argmax) | | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| 1 | Qwen3.5-4B SFT + GRPO, big data, seed 2 | SFT on 47,894 V r2 labels (12 types), GRPO 200 steps | **0.804** [0.784, 0.823] | 0.496 | 0.934 | 0.014 | 0.996 |
| 2 | Qwen3.5-4B SFT + GRPO, big data, seed 42 (`grpo_sft_v2big`) | same | **0.800** [0.782, 0.818] | 0.499 | 0.932 | 0.017 | 1.000 |
| 3 | Qwen3.5-4B SFT + GRPO, big data, seed 3 | same | **0.788** [0.766, 0.809] | 0.469 | 0.932 | 0.017 | 1.000 |
| 4 | Qwen3.5-4B SFT, big data, seed 42 | SFT on 47,894 V r2 labels, 1 epoch | 0.755 [0.734, 0.775] | 0.433 | 0.910 | 0.021 | 1.000 |
| 5 | Qwen3.5-4B SFT + GRPO, small data (`grpo_sft_v2`) | SFT on 3,292 V r2 labels (5 types), GRPO 200 steps | 0.741 [0.717, 0.765] | 0.411 | 0.906 | 0.019 | 1.000 |
| 6 | Qwen3.5-4B SFT, big data, seed 2 | SFT on 47,894 V r2 labels, 1 epoch | 0.741 [0.720, 0.762] | 0.424 | 0.893 | 0.023 | 1.000 |
| 7 | Qwen3.5-4B SFT, big data, seed 3 | same | 0.739 [0.716, 0.761] | 0.417 | 0.901 | 0.024 | 1.000 |
| 8 | Qwen3.8-27B SFT, small data (`sft_27b`) | LoRA SFT on 3,292 V r2 labels, 2 epochs | 0.734 [0.710, 0.756] | 0.416 | 0.901 | 0.023 | 1.000 |
| 9 | Qwen3.5-4B SFT + GRPO, V round-1 labels (`grpo_sft_v1`) | SFT on 3,292 V r1 labels, GRPO 200 steps | 0.729 [0.707, 0.750] | 0.403 | 0.903 | 0.022 | 1.000 |
| 10 | Qwen3.5-4B SFT, small data (`sft_v2`) | SFT on 3,292 V r2 labels, 2 epochs | 0.717 [0.695, 0.739] | 0.375 | 0.874 | 0.026 | 1.000 |
| 11 | Qwen3.5-4B SFT, V round-1 labels (`sft_v1`) | SFT on 3,292 V r1 labels | 0.673 [0.645, 0.700] | 0.322 | 0.868 | 0.030 | 0.999 |
| 12 | Gemini 3.1 Pro preview, reasoning low | none | 0.610 [0.592, 0.627] | 0.234 | 0.804 | 0.031 | 1.000 |
| 13 | Gemini 3.5 Flash, reasoning low | none | 0.609 [0.588, 0.631] | 0.235 | 0.815 | 0.029 | 0.992 |
| 14 | Qwen3.8-27B zero-shot, no thinking | none | 0.592 [0.565, 0.616] | 0.224 | 0.803 | 0.029 | 0.993 |
| 15 | greedy baseline agent (ts-env) | hand-written heuristic | 0.590 [0.564, 0.616] | 0.210 | 0.776 | 0.031 | 1.000 |
| 16 | Gemini 3.5 Flash, reasoning off | none | 0.583 [0.560, 0.606] | 0.204 | 0.802 | 0.032 | 0.999 |
| 17 | Qwen3.5-4B zero-shot, no thinking | none | 0.547 [0.517, 0.577] | 0.189 | 0.758 | 0.036 | 0.986 |
| 18 | uniform random legal (expected) | none | 0.495 [0.486, 0.503] | 0.162 | 0.732 | 0.041 | 1.000 |
| 19 | Qwen3.5-4B zero-shot, thinking, 12k tokens | none | 0.251 [0.224, 0.280] | 0.093 | 0.339 | 0.060 | 0.430 |
| 20 | Qwen3.8-27B zero-shot, thinking, 12k tokens | none | 0.233 [0.203, 0.263] | 0.110 | 0.308 | 0.064 | 0.365 |

How the rows were produced. Every model row is one sample per decision at temperature 0.6 (`eval/README.md`).
The replies do not depend on the oracle, so the saved replies of each run were re-parsed and scored against the
public labels; under the legacy labels the same procedure reproduces every stored score exactly. Rows 9, 11, 12,
13, 16, 17, 19 and 20 were collected during an earlier evaluation on the same 847 decisions with the same prompt
(under the superseded turn-1 oracle) and are rescored the same way. The thinking rows run out of their 12,000-token
budget inside the reasoning block on most decisions (`legal` 0.37 to 0.43), so they measure the budget, not the
model. The greedy row breaks score ties with a per-decision seed taken from the record id,
`GreedyAgent(seed=int(blake2b(id, digest_size=8).hexdigest(), 16))` (`eval/baseline_rows.py`), so its choices do not
depend on the order of the records in the file (a shuffled copy gives the same 847 picks).

Reading the table:

- The three seeds of the big-data recipe span 0.788 to 0.804 (intervals overlap); GRPO adds +0.045 to +0.063 over
  each seed's own SFT-only start (paired, all intervals above 0).
- The 27B LoRA SFT and the 4B SFT on the same labels are level (paired +0.016 [-0.008, +0.040]); they pick
  options of equal value on only 48% of decisions.
- The best adapter is +0.191 [+0.165, +0.216] above the best zero-shot model. The frontier models sit at the
  greedy baseline's level, between 0.58 and 0.61.

**Training labels.** The adapters were trained on labels of the earlier oracle definition (dice peek, unseeded
completer ties; `oracle/README.md`, "Caveats"). Scored against that definition (`data/oracle_heldout_value_r2_legacy.jsonl`)
every row is 0.003 to 0.034 lower (the public labels are smoother, so the per-decision normalisation is kinder) and
the top four keep their order.

## Single-step, oracle_value by decision type (public labels)

| row | card_use (174) | coup_target (158) | headline (170) | place_influence (171) | play_card (174) |
|---|---|---|---|---|---|
| 4B SFT + GRPO, big data, seed 2 | 0.848 | 0.805 | 0.864 | 0.743 | 0.762 |
| 4B SFT + GRPO, big data, seed 42 | 0.837 | 0.772 | 0.884 | 0.746 | 0.760 |
| 4B SFT + GRPO, big data, seed 3 | 0.835 | 0.779 | 0.845 | 0.747 | 0.735 |
| 4B SFT, big data, seed 42 | 0.781 | 0.758 | 0.856 | 0.692 | 0.689 |
| 4B SFT + GRPO, small data | 0.817 | 0.729 | 0.811 | 0.700 | 0.649 |
| 4B SFT, big data, seed 2 | 0.754 | 0.759 | 0.833 | 0.701 | 0.662 |
| 4B SFT, big data, seed 3 | 0.782 | 0.736 | 0.824 | 0.686 | 0.668 |
| 27B SFT, small data | 0.765 | 0.695 | 0.803 | 0.679 | 0.724 |
| 4B SFT + GRPO, V r1 labels | 0.825 | 0.751 | 0.731 | 0.700 | 0.641 |
| 4B SFT, small data | 0.754 | 0.724 | 0.784 | 0.682 | 0.645 |
| 4B SFT, V r1 labels | 0.765 | 0.642 | 0.702 | 0.615 | 0.636 |
| Gemini 3.1 Pro, reasoning low | 0.777 | 0.649 | 0.501 | 0.632 | 0.489 |
| Gemini 3.5 Flash, reasoning low | 0.740 | 0.687 | 0.483 | 0.593 | 0.548 |
| 27B zero-shot | 0.775 | 0.502 | 0.556 | 0.547 | 0.567 |
| greedy agent | 0.713 | 0.659 | 0.428 | 0.629 | 0.525 |
| Gemini 3.5 Flash, reasoning off | 0.738 | 0.641 | 0.454 | 0.589 | 0.495 |
| 4B zero-shot | 0.657 | 0.445 | 0.561 | 0.554 | 0.506 |
| uniform random (expected) | 0.561 | 0.481 | 0.422 | 0.510 | 0.497 |

Headline is where training gains most over greedy and over the zero-shot models (both near or below random
there); place_influence and play_card are the two lowest-scoring types of every SFT + GRPO adapter.

## Full games against the greedy baseline

200 games, 100 per side, seeds 5000 to 5099; thinking off, temperature 0.6. "Rule mask" = the expert rules
B1/B2 (`oracle/README.md`) filter the menu before the model sees it (`eval/play_vs_greedy.py --rule-mask`).
Losses are split by cause (own DEFCON 1 / scoring card held at turn end / everything else), wins into greedy's
own DEFCON 1 and wins on the board.

### Best recipe, three training seeds

| seed | no mask | USSR / USA | losses DEFCON / held card / other | wins greedy DEFCON / board | rule mask | USSR / USA | losses DEFCON / held card / other |
|---|---|---|---|---|---|---|---|
| 42 (`grpo_sft_v2big`) | 66.5% [59.7, 72.7] | 79% / 54% | 8 / 28 / 31 | 46 / 87 | 80.5% [74.5, 85.4] | 86% / 75% | 5 / 1 / 31 |
| 2 | 72.5% [65.9, 78.2] | 86% / 59% | 20 / 22 / 11 | 74 / 71 | 86.5% [81.1, 90.6] | 88% / 85% | 5 / 2 / 20 |
| 3 | 70.0% [63.3, 75.9] | 79% / 61% | 23 / 8 / 27 | 69 / 71 | 76.5% [70.2, 81.8] | 83% / 70% | 5 / 0 / 42 |
| **pooled, 600 games** | **69.7% [65.9, 73.2]** | 81.3% / 58.0% [52.3, 63.4] | 51 / 58 / 69 | 189 / 229 | **81.2% [77.8, 84.1]** | 85.7% / 76.7% | 15 / 3 / 93 |

Every seed's lower bound is above 50%, with and without the mask. Unmasked, the three rates are consistent with
one underlying rate (seed differences are game noise); masked, there is a weaker seed effect (chi-square 6.6,
p = 0.036; seed 3 converts fewer rescued games). As USA alone the unmasked policy is borderline (58.0% pooled,
seed 42 54%). The pooled interval treats the 600 games as one sample and ignores seed variance; the per-seed lower
bounds are the criterion. **What replicates is the win rate, not the way it loses:** the own-DEFCON losses are 8,
20 and 23 for the three seeds, the held-card losses 28, 22 and 8.

### All rows

| player | win rate [95% Wilson] | USSR / USA | losses: own DEFCON 1 / held scoring card / other |
|---|---|---|---|
| value-greedy + rules B1/B2 + V r3 (not shipped) | 93.0% [88.6, 95.8] | 94% / 92% | 8 / 0 / 6 |
| value-greedy (V r3, not shipped) | 89.5% [84.5, 93.0] | 93% / 86% | 17 / 1 / 3 |
| **value-greedy + rules B1/B2 (V r2), reference agent** (`--dice expect --k 4`) | **87.5% [82.2, 91.4]** | 90% / 85% | 20 / 0 / 5 |
| value-greedy (V r2) | 76.0% [69.6, 81.4] | 84% / 68% | 44 / 3 / 0 |
| **4B SFT + GRPO, big data, rule mask, 3 seeds** | **81.2% [77.8, 84.1]** | 85.7% / 76.7% | 15 / 3 / 93 |
| **4B SFT + GRPO, big data, 3 seeds** | **69.7% [65.9, 73.2]** | 81.3% / 58.0% | 51 / 58 / 69 |
| 4B SFT + GRPO, V round-1 labels, rule mask (`grpo_sft_v1`) † | 53.0% [46.1, 59.8] | 62% / 44% | 17 / 1 / 63 |
| 4B SFT + GRPO, V round-1 labels (`grpo_sft_v1`) † | 51.0% [44.1, 57.8] | 62% / 40% | 26 / 13 / 50 |
| 4B SFT + GRPO, small data (`grpo_sft_v2`) † | 42.0% [35.4, 48.9] | 47% / 37% | 46 / 0 / 56 |
| 4B SFT only, big data (`sft_v2big`) † | 21.0% [15.9, 27.2] | | 140 DEFCON of 158 losses |
| 4B SFT only, small data (`sft_v2`) † | 17.5% [12.9, 23.4] | | 155 DEFCON |
| 27B SFT only, small data (`sft_27b`) † | 15.0% [10.7, 20.6] | 14% / 16% | 165 / 2 / 3 |
| Qwen3.5-4B zero-shot † | 7.0% [4.2, 11.4] | | 131 / 0 / 53 |

The reference agent's look-ahead averages the dice over 4 reseeded samples (`--dice expect --k 4 --salt a`), the
dice treatment of the public oracle. The other value-greedy rows let the look-ahead see the real dice
(`--dice peek`, the agent's default), an upper bound for a fair one-ply agent; for the reference agent the peeking
variant is within a point of the non-peeking one (`oracle/README.md`).

† measured on ts-env 515622c.

The greedy baseline itself beats ts-env's safe_random agent 81.5% [75.5, 86.3] (seeds 7000 to 7099, both sides).

Reading the table:

- SFT alone does not transfer to full games: the SFT-only models lose most games by lowering DEFCON to 1
  themselves (argmax imitation learns "coup a battleground" without the DEFCON 2 exception). GRPO after SFT cuts
  that loss class to 8 to 23 games per 200, by a seed-dependent amount.
- The mask removes the held-scoring-card losses (58 to 3 pooled) and most of the remaining own-DEFCON losses
  (51 to 15).
- Many wins are greedy's own mistakes: 189 of the 418 pooled unmasked wins are greedy lowering DEFCON to 1 itself.
- The small-data V r1 and V r2 adapters (51.0% and 42.0%) differ by about what two runs of one recipe differ;
  read both as parity with greedy.
- Earlier adapters trained on the superseded turn-1 oracle reached 12.5% (SFT) and 20.5% (SFT + GRPO), measured on ts-env 515622c.

## Full games against the reference agent (second reference line)

Same protocol, opponent = the reference agent's dice-peeking variant instead of greedy: value-greedy + rules
B1/B2, V round 2, top-10 pruning, `--dice peek` (`eval/play_vs_greedy.py --opponent value_rules`). Network, rules
and pruning are those of the reference agent; only the look-ahead's dice differ, and against greedy the two are
within a point of each other (`oracle/README.md`).

| player | win rate [95% Wilson] | USSR / USA | losses: own DEFCON / held card / VP and other |
|---|---|---|---|
| 4B SFT + GRPO, big data, seed 42 (`grpo_sft_v2big`) | **7.0% [4.2, 11.4]** | 3% / 11% | 28 / 39 / 119 |
| 4B SFT + GRPO, V round-1 labels (`grpo_sft_v1`) † | 5.9% [3.4, 10.3] over 185 games | | |
| 4B SFT + GRPO, small data (`grpo_sft_v2`) † | 2.0% [0.8, 5.0] (27 games cut at the step cap) | 2% / 2% | |

† measured on ts-env 515622c.

The reference agent beats greedy 87.5% (the peeking variant played here is within a point of it); the best LLM,
which beats greedy 66.5% to 72.5% depending on the seed, wins 7.0% against it, and 13 of its 14 wins are the opponent's own DEFCON drops. That gap is what is left between
the LLM and the oracle it was trained on.

## Agent ladder

A Bradley-Terry fit over the scripted and value-net agents (11 pairings, 8 agents, greedy = 0, Elo scale
400 log10 of the odds, bootstrap 95%, deviance 4.8 on 4 degrees of freedom; value-greedy agents with the peeking
look-ahead). Measured on 515622c; refit pending. The last column converts the direct rate against greedy of the
tables above for comparison.

| agent | Elo vs greedy (ladder fit) | from the rate vs greedy above |
|---|---|---|
| V r3 + rules | +403 [+352, +460] | +449 (93.0%) |
| V r3 | +372 [+298, +454] | +372 (89.5%) |
| V r2 + rules (reference agent) | +357 [+318, +404] | +338 (87.5%); +346 for the peeking variant the ladder used |
| V r2 | +281 [+235, +332] | +200 (76.0%) |
| V r1 | +169 [+125, +222] | not measured |
| greedy | 0 (anchor) | 0 |
| eps-greedy (0.10) | -7 [-49, +33] | not measured |
| safe_random | -172 [-226, -125] | -258 (greedy wins 81.5%) |

The LLMs do not fit on this scale: the best adapter (seed 42) converts to +119 from its games against greedy
(66.5%) but about -103 from its games against the reference agent's peeking variant (7.0%, at +346), about 220
Elo apart; in the ladder's own games, adding the LLMs raises the deviance from 4.8 to 39.8 on 6 degrees of freedom. Quote LLM strength against named opponents, not as one Elo number. No human
player anchors this pool.
