# eval: reproducing the leaderboard

Two protocols, both against any OpenAI-compatible chat endpoint.

| protocol | script | what is measured |
|---|---|---|
| single-step (the leaderboard) | `run_zero_shot.sh` (vf-eval), `baseline_rows.py`, `summarize.py` | `oracle_value`, `top1`, `near_best`, `regret`, `legal` on the 847 held-out decisions, game-cluster bootstrap 95% CIs |
| full games | `play_vs_greedy.py` | win rate over 200 games (100 per side) against ts-env's greedy baseline or the value-greedy + rules reference agent, Wilson 95% intervals |

All commands run from the ts-eval root. Outputs go to `results/` (gitignored).

## One leaderboard row, end to end

Example: Qwen3.5-4B zero-shot, served locally with vLLM (any OpenAI-compatible endpoint works the same way).

```bash
# 0. environments: verifiers for the model rows; ts-env comes as two submodules (games commit for full games,
#    data commit 515622c for the model-free rows, which rebuild the positions); oracle/tsenv.py picks one per script
git submodule update --init          # not needed after git clone --recursive
pip install -r requirements.txt "verifiers>=0.3.1" datasets

# 1. serve the model (separate shell); thinking is switched off per request
vllm serve Qwen/Qwen3.5-4B --port 8000 --max-model-len 16384 --reasoning-parser qwen3

# 2. the model row: 847 decisions, temperature 0.6, 256 tokens, thinking off
bash eval/run_zero_shot.sh qwen3.5-4b_nothink Qwen/Qwen3.5-4B http://localhost:8000/v1 off 256

# 3. the model-free rows (uniform-random expectation and the greedy agent's choice), about 30 s on CPU
python eval/baseline_rows.py --data data/oracle_heldout_value_r2_public.jsonl

# 4. the table (main, by decision type, by side)
python eval/summarize.py
```

`run_zero_shot.sh <row> <model> <base_url> <thinking on|off> [max_tokens] [n]` wraps

```bash
vf-eval ts_single_step --env-dir-path env \
  -a '{"data_dir": "data", "train_file": "none.jsonl", "eval_file": "oracle_heldout_value_r2_public.jsonl"}' \
  -m <model> -b <base_url> -k TS_EVAL_TOKEN -n -1 -r 1 -c 32 -t 256 -T 0.6 \
  -S '{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}' \
  --disable-tui --save-results -o results/<row>
```

For a hosted endpoint, put the bearer token in `TS_EVAL_TOKEN` and replace the sampling arguments, since
`chat_template_kwargs` is vLLM-specific. Reasoning models count reasoning tokens against `max_tokens`:

```bash
export TS_EVAL_TOKEN=...            # the provider's token
SAMPLING='{"reasoning_effort": "low"}' CONCURRENCY=8 \
  bash eval/run_zero_shot.sh <row> <model> <provider_openai_compatible_url> off 4096
```

Protocol details: one sample per decision (`-r 1`), temperature 0.6, the prompt is the system prompt plus the
rendered view in `data/`, the reply is resolved to a legal key exactly as ts-env's LLM harness resolves it.
`summarize.py` resamples whole games (decisions of one game are correlated), 2000 resamples. The two
model-free rows reproduce exactly (on the public labels, positions rebuilt on ts-env `515622c`, the
`third_party/ts-env-legacy` submodule): random_legal
0.495 [0.486, 0.503], greedy_agent 0.590 [0.564, 0.616]. Greedy breaks score ties with a seed derived from the record id,
`GreedyAgent(seed=int(hashlib.blake2b(id.encode(), digest_size=8).hexdigest(), 16))`, so its choices do not depend on
the record order of the file.

## Full games

```bash
# 200 games vs greedy (seeds 5000 to 5099, each played once per side), 12 parallel games
python eval/play_vs_greedy.py --model <served name> --base-url http://localhost:8000/v1 \
  --games 100 --workers 12 --out results/games/<row>.jsonl
# the same with the expert rules B1/B2 filtering the menu before the model sees it
python eval/play_vs_greedy.py --model <served name> --rule-mask --games 100 --workers 12 --out results/games/<row>_mask.jsonl
# second reference line: against value-greedy + rules (V round 2, top-10 pruning, dice-peeking look-ahead; needs torch)
python eval/play_vs_greedy.py --model <served name> --opponent value_rules --games 100 --workers 12 --out results/games/<row>_vs_value_rules.jsonl
```

The model plays through ts-env's `LanguageModelAgent` (same prompt as the single-step task, thinking off,
temperature 0.6, 256 tokens, up to 3 retries on an illegal reply, then the first legal option); the opponent is
`GreedyAgent(seed=game_seed + 9999)`. A game is cut after 2500 steps (`--max-steps`) and counts as a non-win;
no published game on ts-env `6791b34` reached it. Full games run on the games commit, the `third_party/ts-env`
submodule, which also supplies the LLM harness (`examples/`). Rows are appended as games finish; rerunning the same command
resumes. A 4B model takes about 250 s per game.

Reference agents without a model:

```bash
# the reference line is --rules --dice expect --k 4 --salt a; without the dice flags the look-ahead peeks
python oracle/value_agent.py --value oracle/value.pt --games 100 --opponent greedy --workers 32 --prune 10 --max-seconds 3600 [--rules] [--dice expect --k 4 --salt a]
```

## Files

| file | role |
|---|---|
| `run_zero_shot.sh` | one vf-eval row on the held-out split |
| `baseline_rows.py` | random_legal (expected) and greedy_agent rows; `--limit N` for a smoke test |
| `summarize.py` | tables with game-cluster bootstrap CIs from `results/` |
| `play_vs_greedy.py` | full games of a served model vs greedy / safe_random / random / value / value_rules; `--rule-mask` |
