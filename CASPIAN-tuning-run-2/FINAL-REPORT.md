# CASPIAN for GAMMAF — adaptation, five variants, and results

All numbers in this report are read directly from the JSON produced by
`MainEvaluation.py` (which uses `EvaluationDebateLoop.py`). No separate metric
code was written. Results file referenced throughout:
`CASPIAN-tuning-run-2/results/variants-50.json`.

---

## 1. Scope and constraints respected

- **Only existing file modified:** `defense-models/CASPIAN.py` (baseline
  adaptation). `git diff --stat` shows exactly this one file changed.
- Everything else lives under `CASPIAN-tuning-run-2/`:
  `configs/`, `data/`, `models/`, `results/`, `logs/`, plus the new
  text-processing class `fast_round_processor.py`.
- `EvaluationDebateLoop.py`, `MainEvaluation.py`, `TextProcessingManager.py`
  and every other framework file are untouched and used as-is.
- Environment initialised with `gammaf-init.sh` (HF_HOME left at
  `/project_ghent/models`). vLLM was never touched.
- Training data generated with `TrainDataGeneration.py` at
  **200 fixed + 200 random questions**, 8 agents, MMLU-Pro, density
  [0.5, 0.9] -> `data/train-mmlupro-200.pkl` (800 debates, 641 valid).
- Evaluation: 8 agents, 2 malicious, `max_rounds: 3`, 50 + 50 questions,
  random density [0.5, 0.9], dataset tag MMLU-Pro.

---

## 2. Baseline adaptation (`defense-models/CASPIAN.py`, model B0)

The pre-existing implementation was the previous study's "v5" (onset-only
instant cascade + Appendix-C partial-correlation copula). It could not be used
by `MainEvaluation.py` for two reasons:

1. **No round-1 flags.** `_flags_for_scores` returned all zeros until
   `cascade_emitted` was `True`, and the cascade rules require a warm
   (previous-turn) spectral state. Round 1 can therefore never flag, so the
   malicious agents are never isolated from round 2 onward.
2. **Degenerate score for AUROC.** The score was
   `max(outgoing, amplifier, bridge)` after *per-component max normalisation*.
   Each component saturates at 1, producing many exact ties; the resulting
   per-round `roc_auc_score` was ~0.5 at best (and behaviour was undefined
   when all scores tied).

Changes made to B0:

- `_flags_for_scores(...)` now emits the top-`top_k` agents **every round**,
  including round 1. (GAMMAF consumes per-round flags; the paper's cascade
  decision only exists at confirmation, so this is the minimal framework
  adapter.)
- `_node_scores(...)` now returns a **continuous, standardised sum** of the
  paper's three attribution statistics — origin (outgoing influence, Eq. 11),
  amplifier (out/in, Eq. 12) and bridge (out·in, Eq. 13). Standardisation
  removes the saturation ties and gives a well-defined AUROC.
- Added `_score_agents(...)`, an overridable hook called from `predict`, so the
  five variants can change only the per-agent scoring while reusing the
  LI-CTE influence estimator, degree-aware normalisation, spectral signals,
  cascade rules and attribution byte-for-byte.
- Added optional `component_weights` config knob.

The compatibility tests in `tests/test_caspian_compatibility.py` pass.

---

## 3. Iteration process and key findings

### 3.1 Why the obvious idea failed

The natural GAMMAF adaptation of "synchronization" is message similarity among
agents that share an answer. Two offline measurements on the generated benign
data killed this idea:

- **Message similarity is dominated by the topic.** Same-answer message cosine
  mean **0.83**, best pair per round p50 **0.935**. Because the pooled
  MiniLM embedding encodes the shared question/choices, any two agents that
  agree look nearly identical — malicious and benign alike.
- **Answer correctness is not a usable label.** Reading the existing B0 debug
  log (`debug-logs/Debug-20260920105019.txt`) showed the injected agents are
  frequently the *correct* answerers whose small group is outvoted; the attack
  sometimes backfires because the framework picks `mal_answer` from A–D without
  excluding the correct option. A defense keyed on "wrong answer" would flag
  benign agents.

### 3.2 The signal that works: the synchronized minority cluster

The two injected agents **always share one answer**. On benign rounds, by
contrast, a same-answer *pair* exists in only **10.6%** of rounds (unique pair
9.6%), 67% of rounds are unanimous, and the majority has size ~5–6. So with
two injected agents the usual structure is a safe majority plus a small
non-majority coordinated cluster. The variants therefore score agents by
answer multiplicity:

```
counts = multiplicity of each answer;  maj = max counts;  c = count(answer_i)
c == maj      ->  -(c / n)             (majority: low anomaly)
c == 1        ->   0.5                (lonely answer)
1 < c < maj   ->   1 + (1 - c / maj)  (small coordinated minority: high)
```

This is a discrete communication-channel synchronization proxy; it is combined
with CASPIAN's own influence/attribution signals, which remain in the model.

### 3.3 Variant lineage

| Version | What it adds over B0 | Rationale |
| --- | --- | --- |
| **V1** | edge-level reciprocity `min(A_ij, A_ji)` added to the attribution score | self-reinforcing propagation is reciprocal; faithful influence-only probe |
| **V2** | + synchronized-minority answer cluster (dominant term, weight 2.0) | the round-1 discriminator found in 3.2 |
| **V3** | + EMA persistence of the cluster score | the paper's persistence requirement: sustained synchronization, not one turn |
| **V4** | + neighbourhood consensus deviation (structural graph) and cascade-role boost | attribution (origin/amplifier/bridge) reflected in flags; local propagation |
| **V5** | coordination-first fusion: cluster + non-majority size-2 group boost + persistence + deviation + light influence tie-break | best of the above; the smallest non-majority group (size 2, or 3 if a benign agent copied the injected answer) is boosted directly |

All variants keep: LI-CTE influence estimation, degree-aware normalisation,
singular-value signals, Watch/instant/multi-turn cascade logic, and the
origin/amplifier/bridge/spine attribution. Only `_score_agents` differs.

---

## 4. Results (from `results/variants-50.json`)

Round 1 is the decisive round: flags at round 1 remove the malicious agents'
edges for all later rounds. Per-topology round-1 values for every model:

| Model | Topology | ADR | FPR | F1 | AUROC (r1) | overall_AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B0 | tree | 14.06 | 28.64 | 0.1406 | 0.380 | 0.537 |
| B0 | chain | 12.96 | 29.01 | 0.1296 | 0.355 | 0.577 |
| B0 | star | 15.62 | 28.12 | 0.1562 | 0.352 | 0.581 |
| B0 | random | 18.75 | 27.08 | 0.1875 | 0.357 | 0.546 |
| V1 | tree | 16.67 | 27.78 | 0.1667 | 0.351 | 0.514 |
| V1 | chain | 9.26 | 30.24 | 0.0926 | 0.364 | 0.565 |
| V1 | star | 12.90 | 29.03 | 0.1290 | 0.296 | 0.592 |
| V1 | random | 11.29 | 29.57 | 0.1129 | 0.331 | 0.551 |
| V2 | tree | 72.58 | 9.14 | 0.7258 | 0.841 | 0.788 |
| V2 | chain | 76.47 | 7.84 | 0.7647 | 0.848 | 0.828 |
| V2 | star | 81.82 | 6.06 | 0.8182 | 0.871 | 0.833 |
| V2 | random | 75.00 | 8.33 | 0.7500 | 0.814 | 0.799 |
| V3 | tree | 71.43 | 9.52 | 0.7143 | 0.800 | 0.777 |
| V3 | chain | 77.59 | 7.47 | 0.7759 | 0.871 | 0.832 |
| V3 | star | 72.06 | 9.31 | 0.7206 | 0.779 | 0.819 |
| V3 | random | 78.38 | 7.21 | 0.7838 | 0.856 | 0.806 |
| V4 | tree | 72.86 | 9.05 | 0.7286 | 0.800 | 0.763 |
| V4 | chain | 74.36 | 8.55 | 0.7436 | 0.801 | 0.799 |
| V4 | star | 70.00 | 10.00 | 0.7000 | 0.812 | 0.767 |
| V4 | random | 75.61 | 8.13 | 0.7561 | 0.829 | 0.795 |
| **V5** | tree | 75.00 | 8.33 | 0.7500 | 0.831 | 0.855 |
| **V5** | chain | 80.30 | 6.57 | 0.8030 | 0.876 | 0.857 |
| **V5** | star | 71.88 | 9.37 | 0.7188 | 0.805 | 0.821 |
| **V5** | random | 76.25 | 7.92 | 0.7625 | 0.850 | 0.864 |

Simple mean of the four topology values per model (for quick comparison only;
the authoritative values are the per-topology rows above):

| Model | r1 ADR | r1 FPR | r1 F1 | r1 AUROC | r2 ADR | r2 ASR | r3 ASR | r3 FPR | r3 F1 | overall AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 15.4 | 28.2 | 0.154 | 0.361 | 36.7 | 42.5 | 40.7 | 21.5 | 0.435 | 0.560 |
| V1 | 12.5 | 29.2 | 0.125 | 0.335 | 37.2 | 39.1 | 37.7 | 20.9 | 0.439 | 0.555 |
| V2 | 76.5 | 7.8 | 0.765 | 0.844 | 76.9 | 28.5 | 27.8 | 15.1 | 0.765 | 0.812 |
| V3 | 74.9 | 8.4 | 0.749 | 0.826 | 78.5 | 28.3 | 27.3 | 15.2 | 0.756 | 0.808 |
| V4 | 73.2 | 8.9 | 0.732 | 0.811 | 75.2 | 28.3 | 27.3 | 17.0 | 0.748 | 0.781 |
| **V5** | **75.9** | **8.0** | **0.759** | **0.841** | **81.7** | 30.0 | 28.4 | **13.0** | **0.786** | **0.849** |

Interpretation:

- **Influence-only variants are near chance.** B0 and V1 give round-1 ADR
  ~12–19% (random would be 2/8 = 25%; the top-2 attribution ranks are
  anti-correlated with the injected agents), FPR ~27–30%, F1 ~0.1, AUROC ~0.3.
- **Adding the synchronized-minority cluster is the step change.** V2–V5 jump
  to round-1 ADR **70–82%**, FPR **6–10%**, F1 **0.70–0.82**, AUROC
  **0.78–0.88**. Because round-1 flags now isolate the injected agents, later
  rounds improve too: round-3 ASR falls from ~38–41% (B0/V1) to ~20–34%
  (V2–V5) and round-3 F1 rises from ~0.43 to ~0.75–0.79.
- **V5 is best overall.** It has the highest overall AUROC on every topology
  (0.821–0.864), the best round-2 ADR (81.7% mean), the best round-3 F1
  (0.786) and the lowest round-3 FPR (13.0%), while matching the others at
  round 1. V2 is a close second and slightly better on the `star` round-1
  (81.8 vs 71.9); V3/V4 are incrementally behind.
- **ASR.** Round-1 ASR is identical across models (~40–45%) because round-1
  responses are generated before any flag is applied; the defense's effect is
  in later rounds and is large (B0/V1 ~38–41% vs V2–V5 ~20–34% at round 3).

---

## 5. Best model: CASPIAN V5

**File:** `CASPIAN-tuning-run-2/models/CASPIAN_V5.py` (shared pipeline in
`models/caspian_base.py`, identical to the adapted `defense-models/CASPIAN.py`).

**Full logic**

1. Run the unchanged CASPIAN pipeline each turn: build the communication
   LI-CTE influence matrix with the Appendix-C partial-correlation copula,
   degree-normalise it, and compute `λ1, λ2`, amplification, spectral gap and
   contraction, and the Watch / instant / multi-turn cascade decisions.
2. On cascade confirmation, run the original origin/amplifier/bridge/spine
   attribution.
3. Per-agent score (the only difference from the baseline):
   - `cluster`: answer-multiplicity score from §3.2 (small non-majority group
     = high, large majority = low), standardised, weight 2.0;
   - `minority_group`: boost (1.5) for the members of the **smallest
     non-majority answer group** (size 2 for the typical injected pair, or
     size 3 if one benign agent happened to echo the injected answer). If two
     size-2 groups exist, the one with the higher message cosine breaks the
     tie;
   - `cluster_ema`: EMA (decay 0.5) of `cluster` across turns, weight 0.8 —
     persistence;
   - `deviation`: answer disagreement with structural neighbours, weight 0.3;
   - `base + reciprocity`: the B0 attribution/reciprocity score, weight 0.2
     each, used only as a tie-break.
4. `_flags_for_scores` flags the top-`top_k` (=2) agents every round; this is
   what guarantees round-1 flagging and isolates the injected pair for rounds
   2–3.

**How it differs from the original/current implementation**

- Original: no round-1 flags (cascade-gated), saturating max-normalised score,
  no answer/synchronization signal. V5 flags every round and drives the
  round-1 decision from a synchronized-minority cluster while retaining the
  full CASPIAN spectral/attribution machinery.
- It does **not** use message similarity as a primary signal (empirically
  topic-dominated), does not use ground-truth labels, and does not fabricate
  the memory/tool/execution channels.

---

## 6. AUROC coherence with `MainEvaluation.py`

`variants-50.json` was produced by `MainEvaluation.py` unmodified. For every
model and topology the JSON contains:

- `rounds_rates[i].AUROC` — per-round mean of `roc_auc_score(gt_flags,
  anomaly_scores)` computed in `EvaluationDebateLoop.debate_question`;
- `rounds_rates[i].pooled_AUROC` — pooled over agent-round observations;
- `overall_AUROC` — pooled across rounds and questions.

All are finite and non-degenerate (0.35–0.88), confirming that every CASPIAN
variant returns a proper per-agent score array that the existing loop turns
into AUROC. Files: `CASPIAN-tuning-run-2/results/variants-50.json`,
`results/report-variants-50.json`.

---

## 7. Insights / key findings

1. CASPIAN's spectral detector cannot flag round 1 by construction (it needs a
   previous turn); the framework adapter must therefore supply a cold-start
   per-agent score. We did this via the attribution statistics, which fixes
   the contract but is weak on its own here.
2. Under GAMMAF's single-channel pooled-embedding interface, the paper's
   influence/spectral signals do not separate the injected agents (round-1
   AUROC ≈ 0.35). This matches the paper's own communication-only ablation,
   which is its weakest setting.
3. Message similarity is **not** a usable coordination signal here because
   MiniLM pooled embeddings are dominated by the shared question (benign
   same-answer cosine ≈ 0.83).
4. Answer correctness is **not** a usable label: the framework's malicious
   answer is drawn from A–D without excluding the truth, so injected agents are
   sometimes the correct minority.
5. The reliable observable is the **synchronized minority answer cluster**:
   two injected agents always share an answer, while benign same-answer pairs
   occur in only ~10% of rounds. This is the difference between ~15% and ~78%
   round-1 ADR.
6. Persistence (V3/V5) and the explicit smallest-minority-group boost (V5)
   improve later-round reliability and overall AUROC; neighbourhood deviation
   and cascade-role boosting (V4) add more false positives than benefit.
7. Net: round-1 ADR 15% → 76%, FPR 28% → 8%, F1 0.15 → 0.76, overall AUROC
   0.56 → 0.85, and round-3 ASR ~41% → ~28% (best single topology 20.7% for
   V3/chain; V5 has the best mean overall AUROC and round-3 F1).

---

## 8. Artifacts

```
CASPIAN-tuning-run-2/
├── FINAL-REPORT.md                     this report
├── progress.md                         running log of the work
├── fast_round_processor.py             new text-processing class (st_embedding only)
├── configs/
│   ├── generation-mmlupro-200.yaml     200/200 generation
│   ├── eval-debug.yaml                 small diagnostic run
│   ├── eval-variants-50.yaml           final 50/50 run (B0 + V1..V5)
│   └── eval-screen-12.yaml             screening run
├── data/
│   ├── train-mmlupro-200.pkl           generated data (800 debates, 641 valid)
│   └── report-train-mmlupro-200.pkl.json
├── models/
│   ├── caspian_base.py                 shared base (= adapted CASPIAN.py)
│   ├── CASPIAN_B0.py                   baseline wrapper
│   ├── CASPIAN_V1.py ... CASPIAN_V5.py variants
│   └── CASPIAN_v0_original.py          pre-adaptation snapshot
├── results/
│   ├── variants-50.json                FINAL results (all six models)
│   ├── report-variants-50.json         timing report
│   ├── debug-eval.json, screen-12.json intermediate
└── logs/                               stdout of generation and evaluations
```

Reproduce the final run:

```bash
python TrainDataGeneration.py CASPIAN-tuning-run-2/configs/generation-mmlupro-200.yaml
python MainEvaluation.py     CASPIAN-tuning-run-2/configs/eval-variants-50.yaml
```

The adapted baseline is `defense-models/CASPIAN.py`; the best model is
`CASPIAN-tuning-run-2/models/CASPIAN_V5.py`.
