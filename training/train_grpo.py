"""GRPO with LoRA on the single-step task; reward = oracle_value, metrics top1/legal logged.

    python training/train_grpo.py --out checkpoints/grpo_4b_lora [--init-adapter checkpoints/sft_4b_lora/final]
Colocated vLLM on the same GPU. Held-out reward is evaluated every --eval-steps (the curve).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from trl import GRPOConfig, GRPOTrainer

from common import DATA, MODEL, grpo_dataset, legal, lora_config, oracle_value, top1

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=42, help="TRL/transformers seed (data order, LoRA init, sampling)")
ap.add_argument("--train", default=str(DATA / "oracle_train_value_r2.jsonl"))
ap.add_argument("--eval", default=str(DATA / "oracle_heldout_value_r2_public.jsonl"))
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--eval-limit", type=int, default=200)
ap.add_argument("--init-adapter", default=None)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--max-steps", type=int, default=-1)
ap.add_argument("--num-generations", type=int, default=8)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--grad-accum", type=int, default=4)
ap.add_argument("--eval-steps", type=int, default=25)
ap.add_argument("--max-completion", type=int, default=96)
ap.add_argument("--vllm-mem", type=float, default=0.35)
args = ap.parse_args()

train = grpo_dataset(Path(args.train), args.limit)
evald = grpo_dataset(Path(args.eval), args.eval_limit)
print("train", len(train), "eval", len(evald))

cfg = GRPOConfig(
    output_dir=args.out,
    seed=args.seed,
    num_train_epochs=args.epochs,
    max_steps=args.max_steps,
    learning_rate=args.lr,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=10,
    per_device_train_batch_size=args.bs,
    per_device_eval_batch_size=args.bs,
    gradient_accumulation_steps=args.grad_accum,
    gradient_checkpointing=True,
    bf16=True,
    num_generations=args.num_generations,
    num_generations_eval=1,
    max_completion_length=args.max_completion,
    temperature=1.0,
    beta=0.0,
    loss_type="dapo",
    scale_rewards="group",
    mask_truncated_completions=True,
    reward_weights=[1.0, 0.0, 0.0],
    chat_template_kwargs={"enable_thinking": False},
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=args.vllm_mem,
    vllm_max_model_length=4096 + args.max_completion,
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=args.eval_steps,
    save_strategy="steps",
    save_steps=args.eval_steps,
    save_total_limit=3,
    log_completions=True,
    num_completions_to_print=2,
    report_to="none",
)
trainer = GRPOTrainer(
    model=MODEL,
    reward_funcs=[oracle_value, top1, legal],
    args=cfg,
    train_dataset=train,
    eval_dataset=evald,
    peft_config=lora_config(),
)
if args.init_adapter:
    # Let TRL build the model and the LoRA wrapper exactly as it does for the base run (the
    # vLLM weight sync depends on those parameter names), then load the SFT adapter into it.
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    sd = load_file(f"{args.init_adapter}/adapter_model.safetensors")
    result = set_peft_model_state_dict(trainer.model, sd)
    unexpected = list(getattr(result, "unexpected_keys", []))
    print(f"init adapter: {len(sd)} tensors loaded, {len(unexpected)} unexpected", unexpected[:3])
    assert not unexpected, "adapter keys do not match the model"
trainer.train()
trainer.save_model(args.out + "/final")
print("saved", args.out + "/final")
