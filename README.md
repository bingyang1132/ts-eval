# ts-eval

Evaluation artifacts for Twilight Struggle decision-making, built on the
[ts-env](https://github.com/bingyang1132/ts-env) game environment:

- **a single-step decision task**: 847 held-out positions from real games (plus 3,292 for training), each with
  every legal option labelled, packaged as a [verifiers](https://github.com/PrimeIntellect-ai/verifiers)
  environment (`data/`, `env/`);
- **a learned oracle**: a value network V(s) = P(win) trained on 109k generated games, which labels the task
  and is validated by playing (`oracle/`);
- **a leaderboard** of LLMs and baselines on the task and in full games (`leaderboard/`);
- **reference agents**: ts-env's greedy baseline and the stronger value-greedy + rules agent built on V
  (`oracle/value_agent.py`), plus the scripts that produce every number (`eval/`) and the training recipes of
  the leaderboard adapters (`training/`).

## Three numbers

Full games, 200 per pairing and seed (100 per side), 95% Wilson intervals, on ts-env `6791b34`;
details in `leaderboard/README.md`.

| | win rate |
|---|---|
| best LLM vs the greedy baseline (Qwen3.5-4B, SFT + GRPO on 47.9k V labels; **three training seeds, 600 games pooled**) | **69.7%** [65.9, 73.2]; every seed's lower bound above 50% (59.7 / 65.9 / 63.3); 81.2% [77.8, 84.1] with the expert rules filtering its menu |
| reference agent vs the greedy baseline (value-greedy + rules B1/B2, V round 2; the look-ahead's dice averaged over 4 reseeded samples, the dice treatment of the public oracle) | **87.5%** [82.2, 91.4] |
| best LLM vs the reference agent (seed 42, no mask; played against the agent's dice-peeking variant, within a point of it against greedy) | **7.0%** [4.2, 11.4] |

On the single-step task the best adapters score `oracle_value` 0.788 to 0.804 (three seeds) against greedy's 0.590
and uniform random's 0.495 (ceiling 1.0; public labels).

## Quick start: reproduce one leaderboard row

```bash
git clone --recursive https://github.com/bingyang1132/ts-eval && cd ts-eval   # brings both ts-env submodules
pip install -r requirements.txt                      # numpy (the engine itself needs nothing else)
pip install "verifiers>=0.3.1" datasets torch        # model rows and the oracle

# random and greedy rows (they rebuild positions, so oracle/tsenv.py selects the data-commit submodule)
python eval/baseline_rows.py --data data/oracle_heldout_value_r2_public.jsonl
vllm serve Qwen/Qwen3.5-4B --port 8000 --reasoning-parser qwen3 &           # or any OpenAI-compatible endpoint
bash eval/run_zero_shot.sh qwen3.5-4b_nothink Qwen/Qwen3.5-4B http://localhost:8000/v1 off 256
python eval/summarize.py
```

`eval/README.md` has the full protocol, hosted endpoints and full games. Already cloned without
`--recursive`: `git submodule update --init`. To use the engine outside these scripts, `pip install -e
third_party/ts-env` (the games commit).

## Relation to ts-env

ts-env is the game: rules engine, observation encoding, text rendering, the greedy baseline and the LLM
harness. It is a separate repository ([bingyang1132/ts-env](https://github.com/bingyang1132/ts-env)); this repo
copies none of it and references two of its commits as git submodules:

| submodule | ts-env commit | used for |
|---|---|---|
| `third_party/ts-env` | `6791b349ce175eed8761404ffeb3ac721c8575b9` (games commit) | full games, the reference agents, generating new data |
| `third_party/ts-env-legacy` | `515622c1944c705066e2f8fe422026be5c87731e` (data commit) | rebuilding dataset positions from `(game_seed, history)`: relabelling, `eval/baseline_rows.py` |

`oracle/tsenv.py` puts the right submodule first on `sys.path` for each script (data commit for
`label_oracle.py`, `relabel_with_value.py` and `eval/baseline_rows.py`, games commit otherwise), checks its commit
and prints the engine in use. `TS_ENV_TASK=data|games` overrides the choice; `TS_ENV_DIR=<path>` points at another
checkout. The single-step environment itself (`env/ts_single_step/`) needs no game engine: views are
pre-rendered.

## Layout

| path | contents |
|---|---|
| `data/` | the labelled decisions: held-out 847 with the public labels (the leaderboard) and with the training-label definition, train 3,292 (training labels; the 47,894-decision train set is a release download), `README.md` (label definitions, provenance, checksums), `SCHEMA.md` (fields) |
| `env/ts_single_step/` | verifiers environment; reward `oracle_value`, metrics `legal`, `top1`, `near_best`, `regret` |
| `oracle/` | V round 2 checkpoint (`value.pt`, 7.4 MB) and report, model / features / trainer, value-greedy agent with rules B1/B2, position generator, labellers, `README.md` (validity test, rule definitions) |
| `eval/` | leaderboard and full-game scripts, `README.md` (commands) |
| `leaderboard/` | the tables |
| `training/` | SFT and GRPO scripts and the recipes |
| `report/` | technical report (`ts_eval_report.md`, `ts_eval_report.pdf`) |
| `third_party/` | ts-env as two git submodules (games commit, data commit) |

## Report

Technical report: [`report/ts_eval_report.md`](report/ts_eval_report.md) (PDF alongside).

## Citation

```bibtex
@misc{ts-eval-2026,
  title  = {ts-eval: a learned-oracle benchmark for Twilight Struggle decisions},
  author = {Bingyang Liu},
  year   = {2026},
  note   = {https://github.com/bingyang1132/ts-eval}
}
```

(Placeholder until the report is out.)

## License

MIT, see `LICENSE`.
