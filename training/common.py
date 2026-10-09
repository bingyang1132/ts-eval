"""Shared pieces for SFT / GRPO on the single-step task (TRL + PEFT environment, see training/README.md)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "env" / "ts_single_step"
sys.path.insert(0, str(ENV_DIR))
import ts_single_step as tss  # noqa: E402  (prompts + reward functions, same as the verifiers env)

DATA = ROOT / "data"
MODEL = "Qwen/Qwen3.5-4B"


def load_rows(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    for line in path.open():
        r = json.loads(line)
        values = {o["key"]: o["value"] for o in r["options"]}
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": tss.SYSTEM_PROMPT.format(side=r["side"])},
                    {"role": "user", "content": tss.USER_PROMPT.format(view=r["view"])},
                ],
                "best_key": r["best_keys"][0],
                "info": json.dumps(
                    {
                        "id": r["id"],
                        "decision_type": r["decision_type"],
                        "legal_keys": list(r["legal_keys"]),
                        "values": values,
                        "value_max": r["value_max"],
                        "value_min": r["value_min"],
                    }
                ),
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def grpo_dataset(path: Path, limit: int = 0) -> Dataset:
    return Dataset.from_list([{"prompt": r["prompt"], "info": r["info"]} for r in load_rows(path, limit)])


# -- reward functions in TRL's calling convention: lists in, list of floats out ---------- #

def _per_sample(fn, completions, info, **kwargs):
    out = []
    for comp, inf in zip(completions, info):
        inf = json.loads(inf) if isinstance(inf, str) else inf
        text = comp[-1]["content"] if isinstance(comp, list) else str(comp)
        out.append(fn(completion=[{"role": "assistant", "content": text}], info=inf))
    return out


def oracle_value(completions, info, **kwargs):
    return _per_sample(tss.oracle_value, completions, info)


def top1(completions, info, **kwargs):
    return _per_sample(tss.top1, completions, info)


def legal(completions, info, **kwargs):
    return _per_sample(tss.legal, completions, info)


def lora_config(r: int = 32):
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=2 * r,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
