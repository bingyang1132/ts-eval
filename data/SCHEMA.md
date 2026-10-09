# Record schema

One JSON object per line. All files share the schema written by `oracle/relabel_with_value.py`
(which keeps the schema of the earlier rollout labeller, `oracle/label_oracle.py label`, so some fields
only make sense for rollout labels and carry fixed values here).

## Identification and position

| field | type | meaning |
|---|---|---|
| `id` | str | `g<game_seed>_s<step_index>`, unique per file |
| `split` | str | `heldout` (`game_seed % 5 == 0`) or `train` |
| `game_seed` | int | seed of the ts-env game the decision comes from: `Game(seed=game_seed)` |
| `step_index` | int | number of atomic actions played before this decision (= `len(history)`) |
| `pairing` | [str, str] | policies that played the source game, `[USSR, USA]`, each one of `greedy`, `safe_random`, `eps_greedy` |
| `side` | str | the player to move, `USSR` or `USA` |
| `turn` | int | game turn, 2 to 10 (turn 1 is not sampled) |
| `action_round` | int | action round within the turn (0 = headline phase) |
| `decision_type` | str | ts-env `DecisionType` value: `headline`, `play_card`, `card_use`, `place_influence`, `coup_target` (all files); the big train file adds `realign_target`, `remove_influence`, `discard_card`, `choose_country`, `choose_card`, `choose_option`, `choose_region` |
| `history` | [str] | the exact action keys from the start of the game; `Game(seed=game_seed)` then `game.step(key)` for each key rebuilds the state (`oracle/label_oracle.py rebuild`). Replays are exact with ts-env at the data commit `515622c` (the `third_party/ts-env-legacy` submodule); later commits do not replay most of them (`data/README.md`) |

## What the model sees

| field | type | meaning |
|---|---|---|
| `prompt_text` | str | the decision's own prompt line, e.g. `How will you use Red Scare/Purge? (4 ops)` |
| `view` | str | `twilight.render.render(observe(state, side, decision))`: the full text view from the mover's side, ending in the numbered menu of legal options. This is the user message of the task |
| `legal_keys` | [str] | the legal action keys in menu order (menu number = zero-based position) |

## Labels

| field | type | meaning |
|---|---|---|
| `options` | [obj] | one per legal option, in menu order: `key` (action key), `label` (menu text), `value` (float, the mover's win probability after this option, see below), `wins` / `draws` (always 0 here; rollout counts in the old labeller); public file only: `rng` (the option's first evaluation consumed the game RNG: a die roll or a reshuffle) and `tie` (the greedy completer broke a tie) |
| `best_keys` | [str] | keys whose value is within 1e-9 of the best (the argmax set; ties possible) |
| `value_max`, `value_min` | float | maximum and minimum of `options[].value` |
| `v_before` | float | V's win probability for the mover in the position itself, before choosing |
| `value_kind` | str | `value_r2`: the checkpoint that produced the labels (`oracle/value.pt`) |
| `rollout_policy` | str | `value_net` (fixed) |
| `horizon` | str | `value_net` (fixed; `value_net+rules` for rule-floored labels, not shipped) |
| `k` | int | 1 (fixed; number of rollouts in the old labeller) |
| `rollout_eps` | float | 0.0 (fixed) |
| `label_seconds` | float | wall time spent labelling this decision (the only field that differs between two runs of the public labelling) |
| `dice`, `dice_k` | str, int | public file: `expect`, 32 (legacy files: field absent, behaviour `peek`) |
| `completer`, `completer_seed`, `expect_completer` | str, int, bool | public file: `hashed`, 0 (unused), true (the k samples also vary the completer seed) |
| `salt` | str | public file: `ts-eval-v1` |

An option's `value`: apply the option on a clone of the game; while the same player is still to move (the
sub-decisions of one action: target, amount, confirm), let the greedy baseline answer, at most 12 steps;
then read V's probability that the mover wins from that state. If the game ended inside those steps, the
value is the exact outcome (1 win, 0.5 draw, 0 loss). In the public file, an option flagged `rng` or `tie` is
the mean of 32 such evaluations with seeds `blake2b(repr((salt, stream, game_seed, step_index, key, j)))`,
`stream` in {`dice`, `completer`}, j = 0..31; other options are evaluated once (with j = 0 seeds). Values are win probabilities, so `regret` in the
environment is in win-probability units.

## How the environment scores a reply

`env/ts_single_step/ts_single_step.py`: the reply is resolved to a legal key (exact reply, then the
`ACTION:` line, then the longest legal key contained in the reply, then a menu number). Then
`oracle_value = (value[key] - value_min) / (value_max - value_min)`, 0 when no legal key is found, 1 when all
options tie. `top1` = key in `best_keys`; `near_best` = within 0.05 of `value_max`; `regret` =
`value_max - value[key]` (`value_max - value_min` when illegal).
