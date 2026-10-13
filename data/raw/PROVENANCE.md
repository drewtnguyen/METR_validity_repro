# Raw data provenance

`pipeline/download_data.py` (`make download`) fetches these files, and
`MANIFEST.json` records where each came from and its SHA-256.

## METR's GitHub repo, pinned

From [METR/eval-analysis-public](https://github.com/METR/eval-analysis-public)
at commit `571a941b2336c417731001229281a5d2202c1e0e` (2026-02-13). The
download refuses any file whose SHA-256 differs from the pin.

| File | Path in the repo | Bytes |
| --- | --- | --- |
| `th1_1_runs.jsonl` | `reports/time-horizon-1-1/data/raw/runs.jsonl` | 13,105,640 |
| `th1_0_runs.jsonl` | `reports/time-horizon-1-0/data/raw/runs.jsonl` | 24,022,538 |
| `release_dates.yaml` | `data/external/release_dates.yaml` | 2,374 |

The upstream repo has moved on since that commit: its later TH 1.1
runs and release dates add newer AIs. That newer data reaches this
archive through the live file below, not through the runs.

## METR's live results, not pinned

`task_results_1_1.yaml` is <https://metr.org/assets/task_results_1_1.yaml>,
METR's published TH 1.1 results. The archived copy was retrieved on
2026-10-06 and has 23 AIs. Because TH 1.1 is being retired in favour of
this file, every download refreshes it, and the data step uses it in
preference to the TH 1.1 runs. The download warns when the refreshed
file has AIs the previous copy lacked, because everything downstream
then needs rerunning. If the fetch fails, the existing copy is kept.

## How the dataset uses them

- **canonical**: cells from `task_results_1_1.yaml`, plus
  GPT-2, davinci-002 (GPT-3) and gpt-3.5-turbo-instruct grafted from
  `th1_0_runs.jsonl`, since TH 1.1 starts at GPT-4 0314. The YAML has
  no family field, so a task's family is its task-id prefix. On TH 1.1
  that prefix matches the runs.jsonl families exactly. If the YAML is
  missing, `th1_1_runs.jsonl` stands in for it.

On the 18 AIs both TH 1.1 sources share, every cell's success and run
counts agree exactly, and release dates and families agree too. Task
lengths agree to the YAML's three-decimal rounding. The live file adds
five newer AIs: Claude Mythos Preview (early), Claude Opus 4.6
(Inspect), GPT-5.3-Codex, GPT-5.4 and Gemini 3.1 Pro.

Grafted rows take TH 1.1's task metadata wherever it disagrees with
TH 1.0. The disagreements are listed in
`data/processed/canonical_info.json`; all of them are task lengths.
