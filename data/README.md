# data

Single-step decisions of Twilight Struggle, each with every legal option labelled by the round-2 value
network (`oracle/value.pt`). Field definitions are in [`SCHEMA.md`](SCHEMA.md).

| file | decisions | games | decision types | labels | size | in this repo | sha256 |
|---|---|---|---|---|---|---|---|
| **`oracle_heldout_value_r2_public.jsonl`** | 847 | 58 | 5 | **public oracle** (below) | 11.0 MB | yes (**the leaderboard split**) | `7d21290f5d28e722bb92df760c75d02eebad432cd6a6bbd2ff1bb346a2e37d8a` |
| `oracle_heldout_value_r2_legacy.jsonl` | 847 | 58 | 5 | legacy (below) | 10.6 MB | yes (the held-out file under the training-label definition; the training curves of `training/README.md` were measured on it) | `08548a3994628604e002b113ad169599586e9e73844e9838871c629b0eaf1f39` |
| `oracle_train_value_r2.jsonl` | 3,292 | 233 | 5 | legacy | 41.0 MB | yes | `1656027a8294d7c70e589e3444a1e988292c7662cd41321a6964fcb198c715fd` |
| `oracle_train_value_r2_big.jsonl` | 47,894 | 2,400 | 12 | legacy | 570.3 MB | no: release download | `2116127e0a4bed899511ce22f4c8bd6f4e4b3c34e075703105e0638b4ceb8f84` |

Same V, two label definitions (`oracle/README.md`, "Caveats"):

| | public oracle (held-out leaderboard file) | legacy (training files, legacy held-out) |
|---|---|---|
| value network | V round 2 (`oracle/value.pt`) | V round 2 |
| dice in the look-ahead | `dice=expect, k=32`: every option whose evaluation uses the game RNG is evaluated on 32 independently reseeded clones and V is averaged | `peek`: the clone replays the real seed and sees the real roll |
| greedy completer tie-breaks | `completer=hashed`: reseeded per (salt, game seed, step, option, sample) with blake2b | one RNG shared across options and decisions of a labelling worker |
| salt | **`ts-eval-v1`** (public, fixed) | none |
| reproducible | bit for bit (two runs with different scheduling: 847 of 847 records identical) | no (a rerun reproduces 82.8% of argmaxes) |
| command | `python oracle/relabel_with_value.py --value oracle/value.pt --candidates <cands.jsonl> --out <labels.jsonl>` (these are the defaults) | `--dice peek --completer shared --salt ''` |

The two agree on 74.7% of held-out argmaxes (ceiling 0.927). The published LLM adapters were trained on the
legacy training labels; the training sets were not relabelled (about 4.5 h of CPU at k = 32 for the big set), and
their held-out ranking does not change under the public labels (`leaderboard/README.md`).

**Which ts-env commit.** Rebuilding a position from `(game_seed, history)` (relabelling, `eval/baseline_rows.py`,
`oracle/label_oracle.py rebuild`) must use ts-env `515622c`, the `third_party/ts-env-legacy` submodule: the
dataset's games were played on it, and their histories include realignment rolls that this commit allows and
later commits do not, so most of them do not replay elsewhere. Full games use `6791b34`, the `third_party/ts-env`
submodule. `oracle/tsenv.py` selects the right one per script. Scoring a model needs neither: the views and
option values are stored in the files.

Splits are by game: held-out games are those with `game_seed % 5 == 0`; no train file contains a held-out game.
Per-type counts:

| file | headline | play_card | card_use | place_influence | coup_target | other 7 types |
|---|---|---|---|---|---|---|
| held-out | 170 | 174 | 174 | 171 | 158 | 0 |
| train | 660 | 695 | 693 | 677 | 567 | 0 |
| train (big) | 6,000 | 6,000 | 6,000 | 6,000 | 6,000 | realign_target 5,351, remove_influence 5,299, discard_card 2,020, choose_country 1,933, choose_card 1,732, choose_option 1,541, choose_region 18 |

## How the files were made

1. Candidates (`oracle/label_oracle.py gen`): games between the 9 pairings of greedy, safe-random and
   epsilon-greedy (eps 0.2), decisions sampled by reservoir with turn >= 2 and 3 to 40 legal options, at most
   3 per decision type per game. Held-out and train: 300 games (seeds 1000 to 1299, of which 291 yielded
   candidates: 58 held-out, 233 train), the five main decision types. Big train set: 3,000 games (seeds 50000 to 52999, the 600 held-out-seed games dropped), all sampled
   decision types.
2. Labels (`oracle/relabel_with_value.py --value oracle/value.pt`): each option is applied on a copy of the
   game, the mover's remaining consecutive sub-decisions are completed by the greedy baseline (at most 12
   steps), and V reads the mover's win probability (the exact outcome if the game ended); for the public labels
   averaged over 32 reseeded copies where the option's outcome depends on the dice or on a completer tie.
   The public file lists the records in candidate order.
3. Big train set only: 76,405 labelled decisions filtered to a value spread (`value_max - value_min`) of at
   least 0.03 and capped at 6,000 per decision type, giving 47,894.

Legacy labels are not reproducible bit for bit (completer tie-breaks, `oracle/README.md`); the shipped files
are the reference labels and scores are always computed against them. The public labels are a pure function of
(position, option, salt): rerunning the command above (on `515622c`) reproduces the file except the wall-time field
`label_seconds`.

## The big train set

`oracle_train_value_r2_big.jsonl` is over GitHub's file-size limit, so it is attached to a GitHub release of
this repo instead of being committed:

```bash
# from the ts-eval root, once the release is published
gh release download --repo bingyang1132/ts-eval --pattern 'oracle_train_value_r2_big.jsonl*' --dir data
# or: curl -L -o data/oracle_train_value_r2_big.jsonl https://github.com/bingyang1132/ts-eval/releases/download/<tag>/oracle_train_value_r2_big.jsonl
# (if the asset is gzipped: gunzip data/oracle_train_value_r2_big.jsonl.gz)
echo "2116127e0a4bed899511ce22f4c8bd6f4e4b3c34e075703105e0638b4ceb8f84  data/oracle_train_value_r2_big.jsonl" | sha256sum -c
```

It is only needed for the large-data training recipe (`training/README.md`); evaluation uses the held-out file.

Release status: the release has not been published yet.
