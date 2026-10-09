# training: SFT and GRPO recipes for the leaderboard adapters

LoRA post-training of Qwen3.5-4B on the single-step task: SFT on the oracle's argmax, then GRPO with the
task reward (`oracle_value`). Training products (adapters, checkpoints, logs) are not part of this repo;
`.gitignore` excludes `checkpoints/` and `*.log`.

## Recipes

| | SFT | GRPO (from the SFT adapter) |
|---|---|---|
| base model | `Qwen/Qwen3.5-4B` (bf16 compute, thinking disabled in the chat template) | same |
| adapter | LoRA r 32, alpha 64, dropout 0.05, all linear projections (q/k/v/o, gate/up/down) | same shape; initialised from the SFT adapter (`--init-adapter`) |
| target / reward | completion `ACTION: <first key of best_keys>`, loss on the completion only | reward `oracle_value` (weight 1); `top1`, `legal` logged at weight 0 |
| optimiser | lr 1e-4, cosine, 20 warmup steps | lr 1e-5, constant after 10 warmup steps |
| batch | small data: 2 epochs, batch 8 x grad-accum 2; big data: 1 epoch, batch 16 x 1 | 200 steps, 8 prompts x 8 generations per step (batch 8, grad-accum 8, 8 generations) |
| loss | cross-entropy, max length 4096 | DAPO, beta 0 (no KL term), group-scaled rewards, truncated completions masked |
| sampling | | temperature 1.0, max completion 96 tokens, colocated vLLM (0.35 of GPU memory) |
| held-out curve | eval loss every 50 steps on 200 held-out decisions | `oracle_value` every 25 steps on 300 held-out decisions |
| time on one A100 80 GB | small: about 1.7 h; big: about 11.4 h | about 4 h |

Data: small = `data/oracle_train_value_r2.jsonl` (3,292 decisions, 5 types); big =
`data/oracle_train_value_r2_big.jsonl` (47,894 decisions, 12 types, release download, `data/README.md`).
Held-out for curves = `data/oracle_heldout_value_r2_public.jsonl` (the published adapters' curves were measured on
`data/oracle_heldout_value_r2_legacy.jsonl`, the held-out file under their training-label definition,
`data/README.md`). GRPO trains on prompts from the same file the SFT
used. Seeds: 42 (default), 2 and 3 for the three-seed repeat (`--seed`).

## Commands (big-data recipe, the best leaderboard row)

```bash
# from the ts-eval root
python training/train_sft.py --out checkpoints/sft_4b_lora_v2big \
  --train data/oracle_train_value_r2_big.jsonl --eval data/oracle_heldout_value_r2_public.jsonl \
  --epochs 1 --bs 16 --grad-accum 1 --lr 1e-4 [--seed 42]
python training/train_grpo.py --out checkpoints/grpo_4b_lora_from_sft_v2big \
  --init-adapter checkpoints/sft_4b_lora_v2big/final \
  --train data/oracle_train_value_r2_big.jsonl --eval data/oracle_heldout_value_r2_public.jsonl \
  --lr 1e-5 --bs 8 --grad-accum 8 --num-generations 8 --max-steps 200 --eval-steps 25 --eval-limit 300 [--seed 42]
python training/curve.py grpo.log        # held-out curve from the GRPO log (stdout redirected to grpo.log)
```

Small-data recipe: the same two commands with `--train data/oracle_train_value_r2.jsonl` and, for SFT,
`--epochs 2 --bs 8 --grad-accum 2`. 27B: `train_sft.py --model Qwen/Qwen3.8-27B --dtype bfloat16` (the default
fp32 load does not fit one A100 and gets offloaded to CPU).

Serve the result for evaluation by mounting the adapter on vLLM, then follow `eval/README.md`:

```bash
vllm serve Qwen/Qwen3.5-4B --port 8000 --max-model-len 16384 --reasoning-parser qwen3 \
  --enable-lora --max-lora-rank 32 --lora-modules grpo_sft_v2big=checkpoints/grpo_4b_lora_from_sft_v2big/final
```

## Software used for the published rows

| package | version |
|---|---|
| torch | 2.13.0 (CUDA 12.9 build, cu129) |
| vllm | 0.29.0 (cu129) |
| trl | 1.14.1 |
| transformers | 5.18.0 |
| peft | 0.21.2 |
| verifiers | 0.3.1 |
| datasets | 5.0.1 |
| flash-linear-attention / causal-conv1d | 0.5.2 / 1.7.0 (fast kernels for the Qwen3.5 linear-attention layers; without them SFT runs at about 0.5 samples/s) |

Gotchas:

- To start GRPO from an adapter, let `GRPOTrainer` build the PEFT model from the model id and load the adapter
  weights into it afterwards (what `--init-adapter` does). Wrapping a hand-loaded model gives parameter names
  that the vLLM weight sync cannot match.
- The SFT prompt is pre-rendered with `enable_thinking=False`, so the training target matches what the model
  sees at evaluation time.

## Files

| file | role |
|---|---|
| `common.py` | dataset loading from `data/`, TRL-convention wrappers of the environment's reward functions, the LoRA config |
| `train_sft.py` | SFT on the oracle argmax |
| `train_grpo.py` | GRPO with colocated vLLM, held-out reward curve |
| `curve.py` | extracts the held-out curve from a GRPO log |
