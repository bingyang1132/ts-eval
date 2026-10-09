#!/usr/bin/env bash
# One leaderboard row: the single-step task on the 847 held-out decisions (V round-2 labels), via vf-eval.
# usage: bash eval/run_zero_shot.sh <row_name> <model> <base_url> <thinking:on|off> [max_tokens] [n]
#   env: EVAL_FILE   label file under data/ (default oracle_heldout_value_r2_public.jsonl)
#        TS_EVAL_TOKEN  bearer token for the endpoint (default EMPTY; local vLLM ignores it)
#        CONCURRENCY    parallel requests (default 32)
#        SAMPLING       full -S JSON, overrides the thinking switch (for endpoints without
#                       chat_template_kwargs, e.g. '{"reasoning_effort": "low"}')
# Needs a Python environment with verifiers >= 0.3.1 and datasets (vf-eval on PATH).
# Output: results/<row_name>/ (vf-eval results.jsonl) and results/<row_name>.log; then python eval/summarize.py.
set -eo pipefail
ROW=$1; MODEL=$2; URL=$3; THINK=$4; MAXTOK=${5:-256}; N=${6:--1}
if [ -z "$THINK" ]; then sed -n '2,10p' "$0"; exit 2; fi
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export TS_EVAL_TOKEN=${TS_EVAL_TOKEN:-EMPTY}
if [ "$THINK" = on ]; then EN=true; else EN=false; fi
SAMPLING=${SAMPLING:-"{\"extra_body\": {\"chat_template_kwargs\": {\"enable_thinking\": $EN}}}"}
EVAL_FILE=${EVAL_FILE:-oracle_heldout_value_r2_public.jsonl}
mkdir -p "$ROOT/results"
vf-eval ts_single_step --env-dir-path "$ROOT/env" \
  -a "{\"data_dir\": \"$ROOT/data\", \"train_file\": \"none.jsonl\", \"eval_file\": \"$EVAL_FILE\"}" \
  -m "$MODEL" -b "$URL" -k TS_EVAL_TOKEN -n "$N" -r 1 -c "${CONCURRENCY:-32}" -t "$MAXTOK" -T 0.6 \
  -S "$SAMPLING" \
  --disable-tui --save-results -o "$ROOT/results/$ROW" 2>&1 | tail -25 | tee "$ROOT/results/$ROW.log"
