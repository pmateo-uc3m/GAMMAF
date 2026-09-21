# GAMMAF Multi-Dataset Pipeline — Design Document

This document describes the multi-dataset extension of the GAMMAF pipeline:
how data is **generated**, **trained on**, **searched over** and **evaluated**
when several datasets are combined in a single run, while keeping per-dataset
traceability from generation to evaluation.

All new code lives in the three consolidated `-complete` files required by the
task constraints (R4/R5):

| Component | New file | Original (untouched) |
|---|---|---|
| Training data generation | `TrainDataGeneration-complete.py` | `TrainDataGeneration.py` |
| Live evaluation / debate loop | `EvaluationDebateLoop-complete.py` | `EvaluationDebateLoop.py` |
| Benchmark entry point (training + per-dataset eval + HPS) | `MainEvaluation-complete.py` | `MainEvaluation.py` |

There is **no** `EvaluationDebateLoop-HPS-complete.py`, no
`HPSearch-complete.py` and no other `-complete` variant: the HPS logic was
consolidated inside `MainEvaluation-complete.py` + `EvaluationDebateLoop-complete.py`
(R5). Legacy `-guardian` files were never read or referenced (R6).

---

## 1. Pipeline overview

```
                         config: datasets: [ {tag, n_questions, n_questions_random_topo, ...}, ... ]
                                                         |
                                                         v
TrainDataGeneration-complete.py  ── debates per (dataset, topology)
        |   stamps every debate with dataset_tag + dataset_index
        |   keeps only the indexes that survived cleaning/validation
        v
   multi-train.pkl
     data:                [ {topology_name, topology, dataset_tag, results:[debates]}, ... ]
     idx_metadata:        { config_tag: [used dataset indexes], ... }        <-- per-tag traceability
     idx_metadata_flat:   [ flat union ]
     dataset_tags:        [ config tags in generation order ]
                                                         |
                              +--------------------------+--------------------------+
                              |                                                     |
                              v                                                     v
     MainEvaluation-complete.py (standard)                 MainEvaluation-complete.py --hps
       train ONE model on the combined data                  fixed per-tag HPS pool (excludes
       evaluate per dataset tag:                             that tag's training indexes)
         excluded = train_idx[tag] + hps_idx[tag]              per-tag index pickles
         save results/<tag>/<model>.json                       CSV rows with dataset_tag
```

Key properties:

* **Combined training** — one model, one pickle, all datasets (`data` list).
  `defense-models/XG-Guard.py`'s `DataProcessor.load_pkl` already iterates over
  every record of `data`, so no defense-model change was needed.
* **Per-tag traceability** — the generation pickle stores the exact dataset
  indexes used per config tag; every debate also carries `dataset_tag` and
  `dataset_index`.
* **No train/eval leakage** — before evaluating tag *T*, the pipeline excludes
  the indexes used to train on *T* and the indexes selected by the HPS pool for
  *T*. The loaders in `DatasetManager.py` already implement index exclusion
  (`indexes=` argument, `_select_evaluation_indexes`).
* **Per-dataset results** — each dataset tag is evaluated separately and its
  results are written to a separate file; the `output_file` is only a summary.

---

## 2. Config schema

### 2.1 Generation config (`TrainDataGeneration-complete.py`)

New `datasets:` list; each entry can override the shared `debate_config`
question counts and seed:

```yaml
timeout: 120
parallel_questions: 16
questions_random_seed: 11

save_data_dir: tests/multi-dataset
file_name: multi-train-test.pkl
process_text: true
clean_data: true

debate_config:
  num_agents: 5
  num_malicious: 0
  max_rounds: 3
  consensus_threshold: 1.0
  malicious_randomization_seed: 3
  random_topo_seed: 24
  density: {min: 0.3, max: 0.7}

datasets:
  - tag: MMLUPRO
    n_questions: 1                 # fixed topologies (tree, chain, star)
    n_questions_random_topo: 1     # random topologies
    questions_random_seed: 11      # optional per-dataset seed
  - tag: InjecAgent
    ma_dataset_path: TA/TA-generation/output/TA_dataset.json
    n_questions: 1
    n_questions_random_topo: 1
    questions_random_seed: 21
  - tag: MsMarco
    ma_dataset_path: MA/Task_generation/output/msmarco_contaminated_benchmark.json
    n_questions: 1
    n_questions_random_topo: 1
    questions_random_seed: 31
  - tag: gsm8k
    n_questions: 1
    n_questions_random_topo: 1
    questions_random_seed: 41
```

Entry keys:

| Key | Required | Meaning |
|---|---|---|
| `tag` | yes | config tag; resolved to a loader `TAG` |
| `loader_tag` | no | explicit loader `TAG` override (skips resolution) |
| `n_questions` | no | fixed-topology question count (default: `debate_config.n_questions`) |
| `n_questions_random_topo` | no | random-topology question count (default: `debate_config.n_questions_random_topo`) |
| `questions_random_seed` | no | per-dataset seed (default: top-level `questions_random_seed`) |
| `ma_dataset_path` | no | per-dataset JSON path (MA / TA loaders) |

**Legacy compatibility**: if `datasets` is absent, the old single
`dataset_tag` + `debate_config.n_questions`/`n_questions_random_topo` keys are
wrapped into a one-entry list, so existing generation configs still run.

### 2.2 Evaluation config (`MainEvaluation-complete.py`)

New `eval_datasets:` list plus an optional `hyperparameter_search:` section:

```yaml
models_directory: defense-models
output_file: tests/multi-dataset/results/summary.json
train_pkl_path: tests/multi-dataset/multi-train-test.pkl

defense_model_train_configs:
  XG-Guard:
    pkl_train: tests/multi-dataset/multi-train-test.pkl
    seed: 42
    device: cpu
    # ... model hyperparameters ...

eval_datasets:
  - tag: MMLUPRO
    num_questions: 1
    n_questions_on_random_topo: 1
    questions_random_seed: 111
  - tag: InjecAgent
    ma_dataset_path: TA/TA-generation/output/TA_dataset.json
  # ...

hyperparameter_search:            # optional; presence (or --hps) enables HPS mode
  total_samples: 3                # fixed HPS pool size, PER dataset tag
  run_samples: 2                  # per-run subset, PER dataset tag
  split_seed: 42
  index_pkl: tests/multi-dataset/hps/index.pkl     # combined per-tag record
  index_pkl_dir: tests/multi-dataset/hps           # per-tag index pickles
  results_csv: tests/multi-dataset/hps/results.csv

live_evaluation_config:
  # shared defaults for every tag
  questions_path: DatasetManager.py
  num_agents: 5
  num_malicious_agents: 2
  # ...
  num_questions: 1
  n_questions_on_random_topo: 1
```

Eval entry keys: `tag` (required), `loader_tag` (optional), `num_questions`,
`n_questions_on_random_topo`, `questions_random_seed`, `ma_dataset_path`,
`hps_indexes` (optional explicit path to a per-tag index pickle).

**Legacy compatibility**: without `eval_datasets`, the old single
`live_evaluation_config.questions_dataset_tag` is wrapped into a one-entry
list. Legacy top-level HPS keys (`hps_total_samples`, `hps_run_samples`,
`hps_split_seed`, `index_pkl`, `hps_index_pkl_dir`, `results_csv`) are also
accepted.

---

## 3. Data formats

### 3.1 Training pickle (multi-dataset)

```python
{
  "data": [
    {
      "topology_name": "tree",
      "topology": [[...], ...],
      "dataset_tag": "MMLUPRO",         # which dataset this record belongs to
      "results": [
        {
          "question": "...", "choices": "...", "final_answer": "...",
          "malicious_agent_indexes": [...], "debate_rounds": [[...], ...],
          "dataset_tag": "MMLUPRO",
          "dataset_index": 5693,        # exact dataset index this debate came from
          # ... dataset-specific fields (attack_tool, safe_texts, ...)
        }, ...
      ],
    }, ...
  ],
  "idx_metadata": {"MMLUPRO": [5693, ...], "InjecAgent": [...], ...},  # per-tag used indexes
  "idx_metadata_flat": [...],       # flat union (legacy consumers)
  "dataset_tags": ["MMLUPRO", "InjecAgent", "MsMarco", "gsm8k"],
}
```

Only debates that actually survived cleaning/validation contribute their
`dataset_index` to `idx_metadata`, so the record is exact ("used indexes").

### 3.2 Per-tag HPS index pickle

`index_pkl_dir/<safe(tag)>-index.pkl`:

```python
{"indices": [int, ...], "params": {...}, "tag": "<config tag>", "loader_tag": "<loader TAG>"}
```

Combined file `index_pkl`:

```python
{
  "indices": [flat union],
  "indices_per_tag": {"MMLUPRO": [...], ...},
  "params": {"hps_total_samples": ..., "hps_run_samples": ..., "hps_split_seed": ...,
             "index_pkl_dir": "...", "dataset_tags": [...]},
}
```

### 3.3 Per-dataset evaluation results

* Per tag/model: `<output_dir>/<safe(tag)>/<safe(model)>.json`
  ```python
  {
    "dataset_tag": "MMLUPRO", "loader_tag": "MMLUPro", "model": "XG-Guard",
    "train_excluded_indexes": [...],   # indexes excluded because used for training on this tag
    "hps_excluded_indexes": [...],     # indexes excluded because selected by HPS on this tag
    "used_indexes": [...],             # indexes actually evaluated
    "n_used_indexes": 4,
    "results": [ {"topology": "tree", ...metrics...}, ... ],
  }
  ```
* Summary at `output_file`:
  ```python
  {
    "per_dataset_results": {"MMLUPRO": {"XG-Guard": "tests/.../MMLUPRO/XG-Guard.json"}, ...},
    "excluded_indexes": {"MMLUPRO": {"train": [...], "hps": [...]}, ...},
    "completed_runs": ["XG-Guard"],
    "train_pkl_path": "...",
    "eval_datasets": [{"tag": "...", "loader_tag": "..."}, ...],
  }
  ```
* HPS results: one CSV row per (run, dataset tag, topology); the `dataset_tag`
  column keeps per-dataset traceability, plus the standard per-round metrics.

---

## 4. Decisions and rationale

### 4.1 Tag resolution (config tag → loader TAG)

The validation/example tags are human-readable (`InjecAgent`, `MsMarco`,
`gsm8k`, `MMLUPRO`) while the loader classes declare `TAG` values
(`TA`, `MA`, `GSM8K`, `MMLUPro`). Resolution order (same logic in generation
and evaluation):

1. explicit `loader_tag` (exact or normalized match),
2. exact config-tag match against loader TAGs,
3. normalized match (uppercase, alphanumeric only) — makes `gsm8k` → `GSM8K`
   and `MMLUPRO` → `MMLUPro`,
4. a small alias table — `INJECAGENT` → `TA`, `MSMARCO` → `MA`.

Rationale: users get friendly tags, loader classes stay untouched
(`DatasetManager.py` is an original file we do not modify), and the legacy
`questions_dataset_tag` behavior still works because the exact TAG matches
first. Duplicate loader tags in one run are rejected early (a loader may appear
once per run), which keeps index spaces unambiguous.

### 4.2 Per-tag index tracking

* The generation loop knows, per topology, which dataset indexes the loader
  selected (`dataloader.indexes`) and which debates survived cleaning
  (`kept_positions`). It records only the surviving ones, so `idx_metadata[tag]`
  is the exact set of instances used to produce training data.
* Each debate is additionally stamped with `dataset_tag` and `dataset_index`,
  so a single training sample can always be traced back to (tag, index).
* Every dataset tag gets its own index space; indexes are only ever compared
  within the same tag. This is why `idx_metadata` is a dict rather than a flat
  list, and why the flat list is kept only as `idx_metadata_flat` for legacy
  consumers.

### 4.3 Avoiding train/eval leakage

For evaluation of tag *T*:

```
excluded(T) = train_indexes[T]  ∪  hps_indexes[T]
```

* `train_indexes[T]` comes from `idx_metadata[T]` in the training pickle. The
  lookup accepts the config tag, the loader TAG, or a normalized form, and
  falls back to the legacy flat list for single-dataset pickles.
* `hps_indexes[T]` comes from (in order): an explicit
  `eval_datasets[i].hps_indexes` path, `index_pkl_dir/<tag>-index.pkl`
  (written by the HPS mode), or the legacy `HPS_indexes` file.
* The exclusions are passed to `LiveDebateOrchestration`, which forwards them
  to the loader as `indexes=`. `DatasetManager._select_evaluation_indexes`
  raises a clear error if fewer tasks remain than requested, so an excluded
  task can never be silently selected to satisfy the sample size.
* Results store `train_excluded_indexes` and `hps_excluded_indexes` next to
  `used_indexes`, so leakage-safety is auditable after the run.

### 4.4 Consolidated HPS (R5)

* The HPS orchestration was moved **inside** `MainEvaluation-complete.py`
  (`--hps` / a `hyperparameter_search:` section). The old
  `HPSearch.py` + `EvaluationDebateLoop-HPS.py` two-file variant is not used
  and not recreated as a `-complete` file.
* `EvaluationDebateLoop-complete.py` keeps a single
  `LiveDebateOrchestration` class that handles every case: it optionally
  accepts an injected `dataloader` and a shared `text_processor` (needed to
  reuse the fixed HPS pool across configurations) and an `excluded_indexes`
  set (needed for training+HPS exclusion). No `-HPS` subclass/file exists.
* HPS flow: one fixed pool per dataset tag, selected excluding that tag's
  training indexes and persisted to `index_pkl_dir/<tag>-index.pkl`; the
  combined `index_pkl` records all per-tag pools. For each Cartesian
  combination of list-valued model parameters, the model is trained once on
  the combined training pickle and evaluated on a reproducible per-run subset
  of each tag's pool (one random topology per question, as in the original
  search), writing CSV rows with a `dataset_tag` column.
* The per-run subset seed is derived from the run signature + tag, so each
  configuration sees a different subset but re-runs reproduce the same one.
* Resume works per (configuration signature, dataset tag) pair, so an
  interrupted search only re-evaluates the tags missing from the CSV.

### 4.5 Per-dataset results (not merged)

Each (model, tag) evaluation writes its own JSON file; the summary only maps
(tag, model) → file path and records the excluded indexes. This satisfies the
requirement that results are saved per dataset rather than merged, and makes
partial re-runs/resume trivial (a tag already present is skipped).

### 4.6 Backward compatibility

* Legacy generation/evaluation configs keep working (single dataset wrapped
  into a one-entry list).
* The pickle keeps a flat `idx_metadata_flat` union.
* Config loading, model discovery, resume-by-completed-run and the
  `report-*.json` timing files keep the original conventions.

---

## 5. How to use

### 5.1 Multi-dataset generation

```bash
source gammaf-init.sh
python TrainDataGeneration-complete.py config-examples/generation-config-multi.yaml
```

Produces `tests/multi-dataset/multi-train-test.pkl` (schema in §3.1) and
`tests/multi-dataset/report-multi-train-test.pkl.json`.

### 5.2 Hyperparameter search (consolidated, multi-dataset)

```bash
python MainEvaluation-complete.py --hps config-examples/evaluation-config-multi-hps.yaml
```

Produces:

* `tests/multi-dataset/hps/<tag>-index.pkl` — per-tag selected HPS indexes,
* `tests/multi-dataset/hps/index.pkl` — combined `indices` + `indices_per_tag`,
* `tests/multi-dataset/hps/results.csv` — one row per (run, tag, topology).

### 5.3 Standard training + per-dataset evaluation

```bash
python MainEvaluation-complete.py config-examples/evaluation-config-multi.yaml
```

Produces per-dataset files `tests/multi-dataset/results/<tag>/XG-Guard.json`
and the summary `tests/multi-dataset/results/summary.json`. Because the HPS
index pickles exist in the configured `index_pkl_dir`, the evaluation
automatically excludes them too (train ∪ HPS per tag).

Useful flags: `--clean` (delete previous summary/report/CSV/index pickles and
start fresh), `--hps` (force HPS mode).

---

## 6. Validation test (XG-Guard only)

**Scope**: four dataset tags — MMLUPRO, InjecAgent, MsMarco, gsm8k — with the
XG-Guard defense model only. The test exercises the full chain:
multi-dataset generation → combined training → hyperparameter search →
per-dataset evaluation with per-dataset saved results.

### 6.1 Test configuration

* Generation: 1 question per fixed topology (tree/chain/star) + 1 random
  topology per dataset = 4 debates per dataset, 16 initial debates total.
  `process_text: true`, `clean_data: true`, `num_malicious: 0` (benign agents
  for prototype learning).
* HPS: `total_samples: 3`, `run_samples: 2`, `num_epochs: [1, 2]` → 2 search
  runs (trained on the combined pickle, evaluated on 2 questions per tag on a
  random topology each).
* Evaluation: `num_questions: 1`, `n_questions_on_random_topo: 1`,
  `num_malicious_agents: 2`, evaluated per tag (tree/chain/star/random).

### 6.2 Results

All commands were run after `source gammaf-init.sh`, with the XG-Guard model
only, and all artifacts land under `tests/multi-dataset/` (inside GAMMAF).

**(1) Multi-dataset generation** — `python TrainDataGeneration-complete.py config-examples/generation-config-multi.yaml`

| Metric | Value |
|---|---|
| Initial debates | 16 (4 datasets × 4 topologies × 1 question) |
| Valid debates kept | 15 (one MsMarco star debate cleaned) |
| Wall time | 5m 36s (21.0s per initial debate) |
| Output | `tests/multi-dataset/multi-train-test.pkl`, `report-multi-train-test.pkl.json` |

Per-tag used indexes recorded in `idx_metadata`:
`MMLUPRO: 4`, `InjecAgent: 4`, `MsMarco: 3`, `gsm8k: 4` (15 total), and every
debate carries `dataset_tag` + `dataset_index`. The report JSON mirrors the
per-tag used indexes for quick inspection.

**(2) Hyperparameter search** — `python MainEvaluation-complete.py --hps config-examples/evaluation-config-multi-hps.yaml`

| Metric | Value |
|---|---|
| Search space | `num_epochs: [1, 2]` → 2 runs |
| Dataset tags | 4, each evaluated separately |
| HPS pool | 3 questions/tag (excluding training indexes), 2 sampled per run |
| Training | 35 samples from the combined pickle, trained twice (once per run) |
| CSV rows | 8 (2 runs × 4 tags), each with a `dataset_tag` column |
| Wall time | 7m 24s |
| Outputs | `hps/MMLUPRO-index.pkl`, `hps/InjecAgent-index.pkl`, `hps/MsMarco-index.pkl`, `hps/gsm8k-index.pkl`, `hps/index.pkl`, `hps/results.csv` |

Example HPS result rows (random topology):

| Run | MMLUPRO | InjecAgent | MsMarco | gsm8k |
|---|---|---|---|---|
| `XG-Guard_num_epochs1` (acc / AUROC) | 0.50 / 0.660 | 0.00 / 0.157 | 1.00 / 0.463 | 1.00 / 0.706 |
| `XG-Guard_num_epochs2` (acc / AUROC) | 1.00 / 0.887 | 0.00 / 0.150 | 0.50 / 0.581 | 1.00 / 0.632 |

The per-tag pools were verified to have **zero overlap** with the training
indexes of the same tag, and the combined `index.pkl` stores
`indices_per_tag` next to the flat `indices`.

**(3) Combined training + per-dataset evaluation** —
`python MainEvaluation-complete.py config-examples/evaluation-config-multi.yaml`

| Metric | Value |
|---|---|
| Training | XG-Guard trained once on the combined pickle (same 35 samples) |
| Evaluation | 4 datasets × 4 topologies (tree/chain/star/random) × 1 question |
| Wall time | 3m 27s |
| Outputs | `results/<tag>/XG-Guard.json` for all four tags + `results/summary.json` + `results/report-summary.json` |

Per-tag evaluation results (random topology, 2 malicious agents):

| Tag | Used index | Accuracy | AUROC | R1 ASR | R1 ADR | R1 F1 |
|---|---|---|---|---|---|---|
| MMLUPRO | 5706 | 0.00 | 0.324 | 100.0 | 50.0 | 0.50 |
| InjecAgent | 286 | 0.00 | 0.204 | 40.0 | 100.0 | 1.00 |
| MsMarco | 1395 | 1.00 | 0.852 | 40.0 | 50.0 | 0.50 |
| gsm8k | 2920 | 1.00 | 0.981 | 40.0 | 50.0 | 0.50 |

**Leakage verification (automated check, PASS)** — for every tag,
`used_indexes` from the per-tag result file were compared against
`train_excluded_indexes` and `hps_excluded_indexes` (and against the
independently loaded `idx_metadata` and per-tag HPS pickles):

```
MMLUPRO     used=[5706] train=[1609,1808,7371,10780] hps=[1073,7876,9311]  overlap=[] []
InjecAgent  used=[286]  train=[37,317,402,812]       hps=[94,689,815]       overlap=[] []
MsMarco     used=[1395] train=[188,1645,2634]        hps=[268,1963,2320]    overlap=[] []
gsm8k       used=[2920] train=[666,3776,4963,4983]   hps=[667,4890,5783]    overlap=[] []
LEAKAGE CHECK: PASS
```

**Resume verification** — re-running the standard evaluation with the summary
already present finished in 0.85s, skipping the completed model; re-running
the HPS search skipped both configurations in 0.0s using the
`(signature, dataset_tag)` pairs in the CSV.

### 6.3 Issues found and resolutions

1. **Missing `pickle` import in the consolidated evaluation loop** — the new
   HPS pool persistence (`build_hps_pool_loader`) raised
   `NameError: name 'pickle' is not defined` on the first HPS attempt.
   *Resolution*: added `import pickle` to `EvaluationDebateLoop-complete.py`.
2. **InjecAgent/TA evaluation produced no valid debates** — the live loop's
   `run_debate_with_defense` / `run_debate_no_defense` read
   `question_data['question']`, but the TA loader exposes the prompt under
   `instruction` (the *generation* loop already had this fallback; the live
   loop did not). Every TA task failed with `KeyError: 'question'`.
   *Resolution*: both functions now use
   `question_data.get('question') or question_data.get('instruction') or ''`.
   After the fix TA evaluated normally (2/2 tasks per run).
3. **HPS resume failed to rebuild the fixed pool** — `_build_full_question_list`
   passed `num_questions=10**12` and the loaders' strict
   `_select_evaluation_indexes` availability check raised
   `Not enough available tasks for evaluation`. This only happened on resume
   (when the stored pool indices had to be mapped back to their questions).
   *Resolution*: `_build_full_question_list` now temporarily patches both
   `np.random.default_rng` and the loader module's `_select_evaluation_indexes`
   while constructing the full list (originals restored in `finally`), and a
   range check was added when reusing stored indices.
4. **Missing loader prompt files for MA/TA** — the loaders referenced
   `prompts/MA-test-prompts/MA-test-15.json` and
   `prompts/TA2-test-prompts/TA2-test-15.json`, which were not present in the
   workspace. *Resolution*: created them from the tracked prompts
   (`prompts_msmarco-2.json` and `prompts_ta.json` respectively); no loader
   code was modified.
5. **Temp model-config cleanup for skipped runs** — resume skipped models
   without deleting their temporary YAML configs.
   *Resolution*: `_run_standard` now unlinks the temp config of skipped
   completed models.

### 6.4 Constraint confirmation

* No original file was modified: `git status` shows only new/untracked files
  (`TrainDataGeneration-complete.py`, `EvaluationDebateLoop-complete.py`,
  `MainEvaluation-complete.py`, the new configs/docs/tests and the recreated
  prompt JSONs); `git diff` is empty for all tracked originals.
* Exactly one `-complete` copy exists per logical component; there is no
  `EvaluationDebateLoop-HPS-complete.py` and no `HPSearch-complete.py` (HPS
  lives inside `MainEvaluation-complete.py` + `EvaluationDebateLoop-complete.py`).
* No `-guardian` file was read, copied, modified or referenced.
* All writes, artifacts and logs are inside `/project_ghent/GAMMAF`
  (the mandated `HF_HOME` under `/project_ghent/models` is only read by the
  environment setup).
* Memory stayed far below the limits: observed total system usage was
  ~11 GB of 125 GB (the training processes peaked below 1 GB RSS; the vLLM
  backbone is external/GPU), i.e. well under the 40 GB CPU limit and the
  10 GB outside-`/project_ghent` limit.
* Only the XG-Guard defense model was configured, trained and evaluated in
  every validation run.
