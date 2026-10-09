# ts-single-step

### Overview
- **Environment ID**: `ts-single-step`
- **Short description**: One Twilight Struggle decision per example, taken from a real mid-game position of [ts-env](https://github.com/bingyang1132/ts-env); the reply is scored against a learned value oracle that labelled every legal option.
- **Tags**: games, strategy, single-turn, train, eval

### Datasets
- **Positions**: sampled from games between mixed baseline agents (the 9 pairings of greedy / safe-random / epsilon-greedy 0.2), turn >= 2, 3 to 40 legal options, five decision types (headline, play_card, card_use, place_influence, coup_target), at most 3 per type per game (`oracle/label_oracle.py gen`).
- **Labels**: for every legal option, the option is applied on a copy of the game, the mover's remaining consecutive sub-decisions are completed by the greedy baseline (at most 12 steps), and the value is V's probability that the mover wins from there (the exact outcome if the game ended). V is the round-2 value network in `oracle/` (`oracle/relabel_with_value.py`). The default eval file carries the public labels: options whose outcome depends on the dice or on a completer tie are averaged over 32 reseeded copies with hashed seeds (salt `ts-eval-v1`), so a relabelling reproduces the file bit for bit; the remaining Monte Carlo noise is measured in `oracle/README.md` (another salt's argmax scores 0.988 to 0.989).
- **Split sizes**: eval 847 decisions from 58 held-out games (`game_seed % 5 == 0`), the leaderboard split; train 3292 decisions from 233 other games. Splits are by game, so no eval position shares a game with train. A larger train set (47,894 decisions, all 12 sampled decision types) is a release download, see `data/README.md`.
- **Schema**: `data/SCHEMA.md`.

### Task
- **Type**: single-turn
- **Prompt**: the same system prompt and rendered board view as ts-env's full-game LLM harness (`examples/llm_agent.py`), ending in the numbered menu of legal action keys.
- **Output format**: `REASON: <sentence>` then `ACTION: <action key>`; a bare key or the menu number is also accepted (same resolution rules as the harness).
- **Rubric**: `oracle_value` is the reward; `legal`, `top1`, `near_best`, `regret` are metrics with weight 0.

### Quickstart
From the ts-eval root, against any OpenAI-compatible endpoint (a local vLLM server here):

```bash
export TS_EVAL_TOKEN=EMPTY        # bearer token for the endpoint; local vLLM ignores it
vf-eval ts_single_step --env-dir-path env \
  -a '{"data_dir": "data", "train_file": "none.jsonl", "eval_file": "oracle_heldout_value_r2_public.jsonl"}' \
  -m Qwen/Qwen3.5-4B -b http://localhost:8000/v1 -k TS_EVAL_TOKEN \
  -n -1 -r 1 -t 256 -T 0.6 -S '{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}'
```

`eval/run_zero_shot.sh` wraps this command; `eval/README.md` has the full leaderboard protocol.

### Env args
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `data_dir` | str | `$TS_SINGLE_STEP_DATA`, else `data/` next to this file, else ts-eval's `data/` | folder holding the JSONL files |
| `train_file` | str | `oracle_train_value_r2.jsonl` | train split (optional; pass a missing name to skip) |
| `eval_file` | str | `oracle_heldout_value_r2_public.jsonl` | eval split |
| `decision_types` | list[str] | all | restrict to some decision types |
| `max_examples` | int | -1 | cap per split |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `oracle_value` (reward) | `(v_chosen - v_min) / (v_max - v_min)`; 0 if the reply is illegal or unparseable; 1 if every option ties and the reply is legal |
| `legal` | reply resolved to a legal action key |
| `top1` | chosen option is in the oracle argmax set (ties count) |
| `near_best` | chosen value within 0.05 of the best |
| `regret` | `v_max - v_chosen` in win-probability units; for an illegal reply the full spread `v_max - v_min` |

Reference rows on the eval split are in `leaderboard/README.md`. `top1` is strict (exact argmax);
`oracle_value` and `regret` are the metrics to compare models on.
