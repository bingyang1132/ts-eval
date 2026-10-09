"""SFT on the oracle argmax: completion = "ACTION: <best key>", loss on the completion only.

    python training/train_sft.py --out checkpoints/sft_4b_lora [--limit 48 --max-steps 5]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from common import DATA, MODEL, load_rows, lora_config

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--model", default=MODEL, help="HF model id (default: the 4B in common.py)")
ap.add_argument("--dtype", default="", help="base-weight load dtype (e.g. bfloat16); empty = TRL default float32. "
                "27B needs bfloat16: fp32 (~108 GB) does not fit one A100 and gets CPU-offloaded")
ap.add_argument("--seed", type=int, default=42, help="TRL/transformers seed (data order, LoRA init, sampling)")
ap.add_argument("--train", default=str(DATA / "oracle_train_value_r2.jsonl"))
ap.add_argument("--eval", default=str(DATA / "oracle_heldout_value_r2_public.jsonl"))
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--max-steps", type=int, default=-1)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--grad-accum", type=int, default=4)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)


def to_text(rows):
    # Pre-render the prompt with thinking disabled so the SFT target matches what the
    # model sees at eval time (enable_thinking=false inserts an empty think block).
    out = []
    for r in rows:
        prompt = tok.apply_chat_template(
            r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        out.append({"prompt": prompt, "completion": f"ACTION: {r['best_key']}{tok.eos_token}"})
    return out


train = Dataset.from_list(to_text(load_rows(Path(args.train), args.limit)))
evald = Dataset.from_list(to_text(load_rows(Path(args.eval), min(args.limit, 200) if args.limit else 200)))
print("train", len(train), "eval", len(evald))
print(train[0]["prompt"][-200:], "||", train[0]["completion"])

cfg = SFTConfig(
    output_dir=args.out,
    seed=args.seed,
    num_train_epochs=args.epochs,
    max_steps=args.max_steps,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    per_device_train_batch_size=args.bs,
    per_device_eval_batch_size=args.bs,
    gradient_accumulation_steps=args.grad_accum,
    gradient_checkpointing=True,
    bf16=True,
    max_length=4096,
    completion_only_loss=True,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="epoch",
    report_to="none",
    model_init_kwargs={"dtype": args.dtype} if args.dtype else None,
)
trainer = SFTTrainer(
    model=args.model,
    args=cfg,
    train_dataset=train,
    eval_dataset=evald,
    processing_class=tok,
    peft_config=lora_config(),
)
trainer.train()
trainer.save_model(args.out + "/final")
print("saved", args.out + "/final")
