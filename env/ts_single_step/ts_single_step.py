"""ts-single-step: one Twilight Struggle decision, scored against a learned value oracle.

Each example is a real mid-game position from ts-env (https://github.com/bingyang1132/ts-env),
rendered exactly as the full-game language-model harness renders it, plus the numbered menu of
legal atomic actions. The model answers with one action key. Every legal option was labelled
offline with a value network V (round 2, see oracle/README.md): the option is applied on a copy of
the game, the mover's remaining consecutive sub-decisions are completed by the greedy baseline,
and the option's value is V's probability that the mover wins from the resulting state (the exact
outcome if the game ended). The default eval file holds the public labels (dice averaged over 32 hashed
reseeds, salt ts-eval-v1): they are reproducible bit for bit (see oracle/README.md).

Reward (weight 1.0)
    oracle_value   (v_chosen - v_min) / (v_max - v_min)   in [0, 1]; 0 for an illegal or
                   unparseable reply; 1 when every option ties and the reply is legal.
Metrics (weight 0)
    legal          reply resolved to a legal action key
    top1           chosen option is in the oracle's argmax set
    near_best      chosen option within NEAR_BEST_TOL of the best value
    regret         v_max - v_chosen, in win-probability units; v_max - v_min when illegal

The dataset is self-contained (views are pre-rendered), so evaluating needs no game engine.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import verifiers.legacy as vf
from datasets import Dataset

NEAR_BEST_TOL = 0.05

SYSTEM_PROMPT = """\
You are playing Twilight Struggle, the Cold War board game, as {side}.

You win by reaching +20 victory points, by controlling Europe when Europe Scoring is
played, or by holding the victory point lead after turn 10.

You lose immediately if you:
  - lower DEFCON to 1 (so never coup, and never fire a DEFCON-lowering event, when
    DEFCON is already low -- note that playing an opponent's card for operations still
    triggers their event, and you carry the blame);
  - are still holding a scoring card when the turn ends.

Play to the position you are shown. Prefer taking control of battleground countries,
watch the military operations requirement, and time scoring cards for when a region
pays you.

Reply in exactly this format and nothing else:

REASON: <one short sentence on why>
ACTION: <one action key from the list you are given>"""

USER_PROMPT = """\
{view}

Reply in exactly this format:

REASON: <one short sentence on why>
ACTION: <one action key from the list above>"""

_ACTION_INDEX = re.compile(r"ACTION\s*:\s*\[?\s*(\d+)\s*\]?[.。]?\s*$", re.IGNORECASE | re.MULTILINE)
_ACTION_LINE = re.compile(r"ACTION\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def extract_key(reply: str, legal_keys: list[str]) -> str | None:
    """Same resolution rules as ts-env's examples/llm_agent.py.

    Prefer an exact reply, then the ACTION line, then the longest legal key the reply
    contains, then a menu number (zero-based position in the legal list).
    """
    cleaned = reply.strip().strip(".,`\"'")
    if cleaned in legal_keys:
        return cleaned
    m = _ACTION_LINE.search(reply)
    if m:
        line = m.group(1).strip().strip(".,`\"'")
        if line in legal_keys:
            return line
        for key in sorted(legal_keys, key=len, reverse=True):
            if key in line:
                return key
    for key in sorted(legal_keys, key=len, reverse=True):
        if key in reply:
            return key
    m = _ACTION_INDEX.search(reply)
    if m:
        i = int(m.group(1))
        if 0 <= i < len(legal_keys):
            return legal_keys[i]
    return None


class ActionParser(vf.Parser):
    """Returns the last assistant message verbatim; key resolution needs the legal set,
    which lives in ``info``, so it happens in the reward functions."""

    def parse(self, text: str) -> str:
        return text


def _chosen_value(completion, info: dict) -> tuple[str | None, float | None]:
    parser = ActionParser()
    reply = parser.parse_answer(completion) or ""
    key = extract_key(reply, list(info["legal_keys"]))
    if key is None:
        return None, None
    return key, float(info["values"][key])


def oracle_value(completion, info, **kwargs) -> float:
    key, v = _chosen_value(completion, info)
    if key is None:
        return 0.0
    vmax, vmin = float(info["value_max"]), float(info["value_min"])
    if vmax - vmin < 1e-12:
        return 1.0
    return (v - vmin) / (vmax - vmin)


def legal(completion, info, **kwargs) -> float:
    key, _ = _chosen_value(completion, info)
    return float(key is not None)


def top1(completion, info, **kwargs) -> float:
    key, v = _chosen_value(completion, info)
    return float(key is not None and v >= float(info["value_max"]) - 1e-12)


def near_best(completion, info, **kwargs) -> float:
    key, v = _chosen_value(completion, info)
    return float(key is not None and v >= float(info["value_max"]) - NEAR_BEST_TOL)


def regret(completion, info, **kwargs) -> float:
    key, v = _chosen_value(completion, info)
    if key is None:
        return float(info["value_max"]) - float(info["value_min"])
    return float(info["value_max"]) - v


def _rows(path: Path, decision_types: list[str] | None, max_examples: int):
    rows = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if decision_types and r["decision_type"] not in decision_types:
                continue
            values = {o["key"]: o["value"] for o in r["options"]}
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT.format(side=r["side"])},
                        {"role": "user", "content": USER_PROMPT.format(view=r["view"])},
                    ],
                    "answer": r["best_keys"][0],
                    # JSON string, not a dict: per-row dicts with different keys would be
                    # unified into one Arrow struct with None gaps. init_state parses it.
                    "info": json.dumps({
                        "id": r["id"],
                        "side": r["side"],
                        "turn": r["turn"],
                        "decision_type": r["decision_type"],
                        "legal_keys": list(r["legal_keys"]),
                        "values": values,
                        "best_keys": list(r["best_keys"]),
                        "value_max": r["value_max"],
                        "value_min": r["value_min"],
                        "k": r["k"],
                        "horizon": r.get("horizon", "game"),
                    }),
                    "task": r["decision_type"],
                }
            )
            if max_examples > 0 and len(rows) >= max_examples:
                break
    return rows


def _default_data_dir() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here / "data", here.parents[1] / "data"):  # packaged copy, then ts-eval/data
        if cand.is_dir():
            return cand
    return here / "data"


def load_environment(
    data_dir: str | None = None,
    train_file: str = "oracle_train_value_r2.jsonl",
    eval_file: str = "oracle_heldout_value_r2_public.jsonl",
    decision_types: list[str] | None = None,
    max_examples: int = -1,
    **kwargs,
) -> vf.Environment:
    """Build the single-step environment from the oracle-labelled JSONL files.

    ``data_dir`` defaults to ``$TS_SINGLE_STEP_DATA``, then a ``data/`` folder next to this
    file, then ts-eval's top-level ``data/``. Default files are the V round-2 labels (847
    held-out decisions, the leaderboard split; 3292 train decisions). Pass a missing
    ``train_file`` name (e.g. ``none.jsonl``) to skip loading the train split.
    """
    base = Path(data_dir or os.environ.get("TS_SINGLE_STEP_DATA") or _default_data_dir())
    eval_rows = _rows(base / eval_file, decision_types, max_examples)
    train_path = base / train_file
    train_rows = _rows(train_path, decision_types, max_examples) if train_path.exists() else []
    parser = ActionParser()
    rubric = vf.Rubric(
        funcs=[oracle_value, legal, top1, near_best, regret],
        weights=[1.0, 0.0, 0.0, 0.0, 0.0],
        parser=parser,
    )
    return vf.SingleTurnEnv(
        dataset=Dataset.from_list(train_rows) if train_rows else None,
        eval_dataset=Dataset.from_list(eval_rows),
        parser=parser,
        rubric=rubric,
        **kwargs,
    )
