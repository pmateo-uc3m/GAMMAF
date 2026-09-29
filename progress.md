# GAMMAF progress

## Current task: scalability sweep instrumentation (agents = 10..100)

### Configuration (current run)
- Pipeline order per agent count: **generation -> HPS -> evaluation**, where the
  evaluation config is derived from the HPS best trials
  (`benchmarks/apply_hps_best.py` -> `exp2/N<N>/evaluation-config-hpsbest.yaml`),
  so MainEvaluation trains/evaluates with the best-scoring hyperparameters.
- Agent counts: 10, 20, 40, 60, 80, 100 (one MODEL_NAME per sweep: `openai/gpt-oss-20b`).
- Dataset: MMLUPRO only.
  - Generation: `num_questions: 0`, `num_questions_on_random_topo: 300` (random topologies only).
  - Evaluation: `num_questions: 0`, `num_questions_on_random_topo: 100`.
- Random-topology density re-centred for N>20 to keep average node degree = 4.
- Defense models: CASPIAN + PREM (both consume only `st_embedding`).
- Text processor: `SentenceOnlyRoundProcessor` (new class in `TextProcessingManager.py`,
  user-authorized) — stores only pooled sentence embeddings, no tk embeddings.
- Malicious agents: `round(0.30*N)`; defense `top_k` = number of ground-truth anomalies.
- HPS: 10 Optuna trials + early stopping (`early_stopping_min_improvement_pct: 1`),
  training `lr_reduce_improvement_pct` / `early_stop_improvement_pct` = 0.5.
- Devices: auto (`cuda` when available).

### Harness (additive; original `.py` untouched except the authorized processor class)
- `benchmarks/metrics/prom.py` — Prometheus parser, vLLM scraper/sampler, stage deltas.
- `benchmarks/metrics/proc.py` — RSS/CPU/threads sampler + torch CUDA memory tracker.
- `benchmarks/metrics/stage.py` — `StageCollector` JSONL records, `attempt` ids, stage suffixes.
- `benchmarks/generation_resume.py` — shard-level checkpoint/resume for generation
  (`--generation-shard-size 25`).
- `benchmarks/instrument.py` — runtime patches: generation, defense train, baseline/live eval,
  HPS train/scoring, per-call defense/embedding/LLM counters.
- `benchmarks/run_instrumented.py` — instrumented stage entry point.
- `benchmarks/run_scalability.py` — orchestrator: subprocess per stage, idle gate, sweep state,
  memory watchdog (`--memory-limit-gb 36`), storage watchdog disabled (`--storage-limit-gb 0`).
- `benchmarks/aggregate.py` — stage records -> CSVs + summary, latest-attempt lineage.

### Resume / checkpointing
- `exp2/runs/sweep-state.json` (atomic) + `runs/sweep-log.jsonl`; `running` -> `interrupted` on restart.
- Generation shards in `<output_dir>/.shards-<TAG>-<file>/` (raw pre-embedding), `.complete` marker.
- Evaluation resumes per model+dataset combo; HPS per model (framework `output_file` based).
- Aggregation keeps only the newest attempt per logical stage (`--include-superseded` to count retries).
- 2026-09-28: previous run (N=10 with XG-Guard/tk, 250/100 questions) was stopped and wiped;
  restarted fresh with the configuration above.

### Status
- 2026-09-29: prompt placeholders extended for evaluation
  (`flags`, `flags_string`, `anomaly_scores`, `anomaly_scores_string`, plus the
  existing `malicious_indexes`/`malicious_agents_string`), wired through
  `EvaluationDebateLoop.py` and kept symmetric (empty) in the generation loop.
  Documented in `/project_antwerp/repos/placeholders.md`.
- 2026-09-29 12:17: N=100 shards 1-2 regenerated with relaxed cleaning (all
  three shards 100/100 valid; old shards had 42 and 39). N=100 generation
  completed at 14:32 (pkl 129.5MB, 300/300 valid).
- 2026-09-29 20:01: **N=100 fully completed** (generation 300/300 valid,
  HPS PREM+CASPIAN 10 trials each, evaluation with best params). Sweep stopped
  intentionally right after, before N=40, at the user's request (guard
  `stop_after_n100.py`). Resume N=40 -> N=60 when instructed; N=80 is skipped
  (`stop_after_n60.py` guard remains armed to prevent it).
- N=40/N=60 outputs and old stage records were archived as `superseded-*`;
  aggregation keeps only the newest attempt per stage.
- 2026-09-29 11:14: cleaning relaxed (see `/project_antwerp/repos/cleaningchanges.md`):
  a debate is dropped only when an agent response has neither a non-empty `message`
  nor a non-empty `answer`; empty single fields no longer remove debates, and
  `TextProcessingManager` tolerates empty/missing messages. N=100 generation
  resumed for the final shard with the new rules after shards 0-1 were
  checkpointed (valid: 42/100 and 39/100 under the old rules).
- 2026-09-29 10:00: N=100 generation paused deliberately after shard 1/3
  checkpoint so a second GPU could be added.
- 2026-09-29 10:16: resumed on 2 GPUs (vLLM `--data-parallel-size 2`) from the
  shard-0 checkpoint. Applied: generation window 25 debates, generation
  `llm.timeout` 900s; evaluation concurrency 900, eval `llm.timeout` 600s.
- Sweep running. 2026-09-28: N=10 and N=20 fully completed; N=40 generation in progress.
- `llm.max_concurrent_inference` raised 150 -> 300 -> 600 -> 900 -> 1500 for stages
  launched after 2026-09-28 21:35 (vLLM kept draining the queue, KV cache <35%,
  0 timeouts; monitor waiting-requests/timeouts each cycle and lower if needed).
- Generation shard size: 25 -> 75 (4 shards) was tried; at N=100, 2 shards of 150
  created ~15k client threads (150 questions x 100 agents) and GIL starvation
  (0/150 completions in 4.7h, vLLM running ~11). Reverted to 25-question shards.
- Shard-tail idle time is addressed with the framework's own worker pool: set the
  generation `llm.max_concurrent_inference` below the shard size so
  `run_debate`'s `ThreadPoolExecutor` acts as a sliding window (start a new
  debate as soon as one finishes). Final setting (2026-09-29 08:36, one GPU
  available): N=80/N=100 generation window = 15 concurrent debates, evaluation
  `max_concurrent_inference` = 900, 3 shards of 100 questions per N
  (`--generation-shard-size 100`). N=80 runs last, after N=100.
- Aborted attempts: stage records of an aborted/re-run stage are archived as
  `stages.jsonl.superseded-<ts>` when the run is reset manually, and the
  aggregator additionally keeps only the newest attempt per logical stage
  (`benchmarks.aggregate`), so aborted work does not pollute the metrics.
- 2026-09-28 20:00: server kill interrupted only N=40 evaluation; sweep restarted and
  resumed (N=10/20 done, N=40 gen+HPS done, N=40 evaluation rerunning).
- Metrics attribution fix: vLLM convenience totals are delta-based; `benchmarks.aggregate`
  recomputes them from raw per-key deltas, so HPS/defense-training stages correctly show
  zero vLLM usage.

### Notes
- HPS performs no vLLM calls (precomputed pkls); generation/baseline/live-eval consume tokens.
- `setup.sh` (AGENTS.md §3) does not exist on this machine; venv activation is used instead.
- Server served model must match `MODEL_NAME`; vLLM is running `openai/gpt-oss-20b`.
