"""Put the right ts-env engine first on sys.path and report which one is in use.

ts-env (https://github.com/bingyang1132/ts-env) is referenced as two git submodules of this repo:

- `third_party/ts-env`, the **games commit** 6791b34: full games, the value-greedy agents, new data;
- `third_party/ts-env-legacy`, the **data commit** 515622c: the dataset's games were played on it, so scripts
  that rebuild a position from `(game_seed, history)` (`label_oracle.py`, `relabel_with_value.py`,
  `eval/baseline_rows.py`) need it.

The task is chosen when this module is imported, before any `twilight` import: 'data' when the entry script is
one of the position-rebuilding scripts above, otherwise 'games'; `TS_ENV_TASK=data|games` overrides the choice.
The chosen checkout's root and its `examples/` folder (baseline agents, LLM harness) are put first on
`sys.path`, so its `twilight` wins over a pip-installed one. `TS_ENV_DIR=<path>` replaces the submodule with an
explicit checkout. `require(kind)` prints the engine in use and warns when the checkout's commit is not the one
`kind` expects.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DATA_COMMIT = "515622c1944c705066e2f8fe422026be5c87731e"
GAMES_COMMIT = "6791b349ce175eed8761404ffeb3ac721c8575b9"
# commits whose engine is identical to the games commit (the later change moves the baseline agents into the package)
_SAME_ENGINE = {"games": {GAMES_COMMIT, "3f4da075bdde7349c77ebf5d65aaf7bcc75298c0",
                          "05cd8defd2631b394842fabfaffc27c350f2100a"},
                "data": {DATA_COMMIT}}
_REPO = Path(__file__).resolve().parents[1]
SUBMODULES = {"games": _REPO / "third_party" / "ts-env", "data": _REPO / "third_party" / "ts-env-legacy"}
_DATA_SCRIPTS = {"label_oracle.py", "relabel_with_value.py", "baseline_rows.py"}

TASK = os.environ.get("TS_ENV_TASK") or ("data" if Path(sys.argv[0]).name in _DATA_SCRIPTS else "games")
if TASK not in SUBMODULES:
    raise ValueError(f"TS_ENV_TASK must be 'games' or 'data', got {TASK!r}")
CHECKOUT = Path(os.environ["TS_ENV_DIR"]).expanduser().resolve() if os.environ.get("TS_ENV_DIR") else SUBMODULES[TASK]

if not (CHECKOUT / "twilight" / "__init__.py").exists():
    raise ImportError(
        f"ts-env not found at {CHECKOUT}: run `git submodule update --init` in the ts-eval root "
        "(or clone with --recursive), or set TS_ENV_DIR to a ts-env checkout")
if "twilight" in sys.modules and Path(sys.modules["twilight"].__file__).resolve().parents[1] != CHECKOUT:
    raise ImportError("twilight was imported before tsenv; import tsenv first")
for _p in (CHECKOUT / "examples", CHECKOUT):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))


def head(path: Path = CHECKOUT) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def require(kind: str = TASK) -> None:
    """Print the engine in use; warn if its commit is not the one `kind` expects."""
    import twilight
    want = {"data": DATA_COMMIT, "games": GAMES_COMMIT}[kind]
    h = head()
    print(f"ts-env engine ({TASK}): {Path(twilight.__file__).resolve().parents[1]} "
          f"(commit {h[:7] if h else 'unknown'})", file=sys.stderr, flush=True)
    if h not in _SAME_ENGINE[kind]:
        what = ("rebuilding dataset positions needs the data commit" if kind == "data"
                else "full games run on the games commit")
        print(f"WARNING: {what} {want[:7]}, found {h[:7] if h else 'unknown'}", file=sys.stderr, flush=True)
