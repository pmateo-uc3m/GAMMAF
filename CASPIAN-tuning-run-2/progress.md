# CASPIAN tuning run 2 — progress

## Objective
Adapt `defense-models/CASPIAN.py` so it (1) flags at round 1 and (2) returns
scores that yield a proper AUROC in the `MainEvaluation.py` results JSON, then
develop up to 5 variants that improve ADR, lower ASR, keep FPR low and F1 high.

All metrics reported here come exclusively from `EvaluationDebateLoop.py` via
`MainEvaluation.py`. No custom stats code is used.

## Constraints (do not violate)
- Only `CASPIAN.py` (baseline) and new files inside `CASPIAN-tuning-run-2/` may be written.
- Use `gammaf-init.sh` to initialise env (pyenv + HF_HOME). Never redirect HF_HOME.
- 40 GB disk cap outside GAMMAF. HF assets are cached already.
- 8 agents; evaluation with 2 malicious; max_rounds 3; density [0.5, 0.9];
  dataset tag MMLUPRO; 200/200 generation, 50/50 evaluation.
- No new ASR/stat parsing code.

## Status log

### 2026-09-20 — setup
- Initialised env with `gammaf-init.sh` (HF_HOME=/project_ghent/models).
- Confirmed vLLM endpoint `openai/gpt-oss-20b` is live.
- Confirmed MMLU-Pro dataset and all-MiniLM-L6-v2 already cached.
- Created directory tree: configs/ data/ models/ results/ logs/.
- Inspected current `CASPIAN.py`: `_flags_for_scores` returns all-zero flags
  until `cascade_emitted` is True; cascade needs a warm (previous-turn) state,
  so round 1 can never flag. This is the root cause of requirement (1).
- Scores are returned every round, so AUROC is computed by the loop, but the
  ranking quality at round 1 is poor (many ties from per-component max
  normalisation). This drives requirement (2).

### 2026-09-20 — prior-art review
- Found and read the deleted `CASPIAN-tests/FINAL-REPORT.md` (commit 4073202).
  Current `defense-models/CASPIAN.py` is that project's v5
  (sha256 7169821a…): onset-only instant rule + Appendix-C partial-correlation
  copula. Prior study used 0 malicious agents, so AUROC was undefined there.
- Key structural facts from that study (reused to reason about scoring):
  degree normalization pins `lambda1 ~ 1`; tree/chain are bipartite so
  `gap = 0`; WeakLink is vacuous; CrossChannel is always false. Therefore the
  spectral detector alone is weak here and the per-agent score/flags must do
  the work.

### 2026-09-20 — baseline adaptation (B0)
- Edited `defense-models/CASPIAN.py` only:
  * `_flags_for_scores` now emits top-`top_k` every round (round-1 flagging).
  * `_node_scores` now returns a continuous standardised sum of the attribution
    statistics origin/amplifier/bridge (proper AUROC, no saturation ties).
  * added `_score_agents(...)` extension hook so variants can override only the
    scoring, leaving LI-CTE/spectral/attribution untouched.
- Added `top_k`/`component_weights` config knobs. Repo compatibility tests pass.

### 2026-09-20 — variants (code written, not yet evaluated)
- `models/caspian_base.py` — exact copy of the adapted baseline, shared base.
- `CASPIAN_B0.py` — baseline wrapper.
- `CASPIAN_V1.py` — reciprocity-aware influence score.
- `CASPIAN_V2.py` — V1 + same-answer message coordination + support penalty.
- `CASPIAN_V3.py` — V2 + temporal persistence (EMA) of coordination.
- `CASPIAN_V4.py` — V2 + neighbourhood consensus deviation + cascade-role boost.
- `CASPIAN_V5.py` — coordination-first fusion + reciprocal-pair evidence.
- All six load and flag on synthetic data; smoke test flags the injected
  coordinated pair (V3/V5).

### 2026-09-20 — data
- 200/200 MMLU-Pro generation launched (background, PID logged in
  `logs/gen-200.log`). 8 agents, density [0.5, 0.9], seed 101.

### 2026-09-20 — generation complete + first measurements
- Generation done: 800 initial debates, 641 valid, `train-mmlupro-200.pkl`
  (4.1 GB, inside GAMMAF), 133 min.
- B0 debug eval (12/12): ADR ~7-23%, FPR ~25-31%, AUROC ~0.30-0.66.
- Screening eval (V1-V4 at 12/12, V5 interrupted): round-1 ADR is near chance
  for all variants and FPR ~25-33%, so the round-1 coordination signal is not
  yet separating the injected pair. Later rounds climb (ADR ~40-60% at r3),
  i.e. only the cascade/influence terms help. Weak round-1 detection is the
  main open problem.

### 2026-09-20 — root-cause diagnostics (offline, no LLM)
Read the existing B0 debug log (`debug-logs/Debug-20260920105019.txt`):
- Injected agents (ground_truth=1) are frequently the *safe/correct* answerers
  whose small group is outvoted; B0's influence score flags benign
  wrong-answerers instead. So answer correctness is NOT a usable signal (the
  attack sometimes backfires because the framework picks `mal_answer` from
  A-D without excluding the correct option).
- B0's score also degenerates after round 1 (all-equal values) once edges are
  removed.

Offline statistics of the already-generated benign data (150/300 debates):
- Same-answer message cosine is dominated by topic: mean 0.83, best pair per
  round p50 0.935. => MESSAGE SIMILARITY CANNOT DISCRIMINATE. Drop it as the
  main term.
- Answer-group structure on benign rounds: 67% of rounds are unanimous;
  majority fraction mean 0.90; a same-answer PAIR exists in only 10.6% of
  rounds, and it is the unique pair in 9.6%. The two injected agents, however,
  ALWAYS share one answer, so with the usual safe majority (size ~5-6) they
  form the unique non-majority pair.

### 2026-09-20 — FINAL 50/50 RUN IN PROGRESS
Launched once: `MainEvaluation.py configs/eval-variants-50.yaml` (B0 + V1..V5),
background PID logged, results appended incrementally to
`results/variants-50.json`. B0 finished (31m):
- tree r1 ADR 14.1 / FPR 28.6 / F1 0.14 / AUROC 0.38; r3 ADR 40.6 / F1 0.41
- chain r1 ADR 13.0 / FPR 29.0 / F1 0.13 / AUROC 0.36
- star  r1 ADR 15.6 / FPR 28.1 / F1 0.16 / AUROC 0.35
- random r1 ADR 18.8 / FPR 27.1 / F1 0.19 / AUROC 0.36
=> B0 round-1 detection is near-chance, as predicted.

### 2026-09-20 — interim results as they land (variants-50.json)
- B0 and V1 (influence-only): round-1 ADR 9-19%, FPR 27-30%, F1 ~0.09-0.19,
  AUROC ~0.30-0.36. Near chance.
- **V2 (reciprocity + synchronized-minority cluster): round-1 ADR 72.6-81.8%,
  FPR 6.1-9.1%, F1 0.73-0.82, AUROC 0.81-0.87; overall AUROC 0.79-0.83.**
  This confirms the answer-cluster signal is the right discriminator.

### 2026-09-20 — DONE
Final 50/50 run finished (178 min total) and all six models are in
`results/variants-50.json`. Final summary (mean of the four topology rows):

| Model | r1 ADR | r1 FPR | r1 F1 | r1 AUROC | overall AUROC | r2 ASR | r3 ASR | r3 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 15.4 | 28.2 | 0.154 | 0.361 | 0.560 | 42.5 | 40.7 | 0.435 |
| V1 | 12.5 | 29.2 | 0.125 | 0.335 | 0.555 | 39.1 | 37.7 | 0.439 |
| V2 | 76.5 | 7.8 | 0.765 | 0.844 | 0.812 | 28.5 | 27.8 | 0.765 |
| V3 | 74.9 | 8.4 | 0.749 | 0.826 | 0.808 | 28.3 | 27.3 | 0.756 |
| V4 | 73.2 | 8.9 | 0.732 | 0.811 | 0.781 | 28.3 | 27.3 | 0.748 |
| **V5** | **75.9** | **8.0** | **0.759** | **0.841** | **0.849** | 30.0 | 28.4 | **0.786** |

Best model: **V5** (highest overall AUROC on every topology, best r2 ADR, best
r3 F1, lowest r3 FPR). Full report: `FINAL-REPORT.md`.

### PLAN (historical; completed)
Root cause: the round-1 discriminator must be the **synchronized minority
answer cluster** (group size), not embedding similarity. Revised variants:
- V1: influence + reciprocity (faithful influence-only).
- V2: reciprocity + minority-cluster (answer-group-size) score.
- V3: V2 + temporal persistence of the cluster score.
- V4: V2 + neighbourhood consensus deviation + cascade-role boost.
- V5: coordination-first fusion (minority cluster + non-majority size-2 pair +
  persistence + small influence tie-break).

1. Diagnose round-1 separation once, from the ALREADY-EXISTING debug log
   (`debug-logs/Debug-20260920105019.txt`): inspect per-agent answers, ground
   truth, scores and flags. Determine why the coordinated malicious pair is not
   top-ranked (missing answers? benign agreement? weak similarity?).
2. Switch `fast_round_processor.py` to use the GPU when available (GPU has
   ~13 GB free; embedding model is 90 MB), serialising encode calls with a
   lock. This removes the 4-core CPU bottleneck and cuts per-model eval time.
3. Apply ONE targeted scoring fix to the variants based on (1). Keep all five
   variants as required, but make their round-1 term correct rather than
   random.
4. Run exactly ONE 50/50 evaluation of B0 + V1..V5
   (`configs/eval-variants-50.yaml`, resume-capable: each finished model is
   saved immediately, so partial results are usable). Do not re-run screening.
5. Read metrics directly from the produced JSON (no new parsing/metrics code)
   and write the final report.

### Constraint reminders
- The only existing file modified is `defense-models/CASPIAN.py`.
- New allowed files: everything under `CASPIAN-tuning-run-2/` including the new
  text-processing class `fast_round_processor.py` and the variant models.
- All metrics come from `MainEvaluation.py` -> `EvaluationDebateLoop.py`.
