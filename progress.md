# GAMMAF Prompt Placeholders Task — Progress

**STATUS: COMPLETE** (all plan steps done; validation test passed with real runs).

Task: make three new prompt placeholders available framework-wide (system/agent/debate prompts):
`{topology_string}`, `{malicious_agents_string}`, `{flags_string}`. Purely additive.

## Overall Plan

1. **Env init** — source `gammaf-init.sh` (done before any execution). ✅
2. **progress.md** — create with full plan (this file); update after every step. ✅
3. **Codebase exploration** — find every place the prompt-filling dictionary is constructed and
   passed to prompt formatting (generation pipeline + evaluation/debate loop). Skip
   `-guardian` files entirely (R6). ✅
4. **Design** — exact format/logic for `topology_string`, `malicious_agents_string`,
   `flags_string`, derived from actual topology/state/malicious-assignment objects. ✅
5. **Implementation** — add keys in the single `-complete` copy of each affected component
   (R4/R5; reuse existing `-complete` files where present, do not fork variants). ✅
6. **Design documentation** — `PROMPT_PLACEHOLDERS_DESIGN.md` written alongside the work. ✅
7. **Validation test** — small test prompt set with all three placeholders + small debate
   (few questions, small topology, ≥2 rounds); confirm rendering, correct topology/malicious
   values, `flags_string` empty in round 1 and populated later if flags occur. ✅ **PASS**
8. **Fix issues** found during the test, staying within R1–R6. ✅ (2 issues fixed)
9. **Finalize** — design `.md` includes rendered test evidence; confirm purely additive;
   mark this file complete. ✅

## Current Step

None — task complete.

## Results of Last Completed Step (Step 9 — finalization)

- `PROMPT_PLACEHOLDERS_DESIGN.md` written and finalized: build sites, exact computation rules,
  sample outputs, usage instructions, additivity confirmation and the actual rendered test
  evidence (evaluation round 1 with `flags_string=[]`, evaluation round 2 with
  `flags_string=[1]` and the flag-modified topology, generation round 2 with malicious `[2]`).
- Final checks: `python -m py_compile` OK on all `-complete` files and the test;
  `git diff --stat` empty (no tracked/original file modified);
  exactly one `-complete` copy per logical component (4 files total: the 3 from the previous
  multi-dataset task + the new `DebateDataGenerationLoop-complete.py`);
  no `-guardian` file was read, copied or referenced; nothing written outside GAMMAF.

## Results of Step 8 — issues found during validation and fixes

1. **Flag vector vs index list mismatch (real bug).** The evaluation call site passed
   `flagged_indexes=[1]` (agent indexes) into `build_flags_string`, which expects the 0/1
   *flags vector* per agent (`[0, 1, 0]`), so the placeholder rendered `[0]` instead of `[1]`.
   Fix: `generate_debate_round_concurrent` now takes `flags: list[int] | None` and
   `debate_question` passes the live `flags` vector directly to `build_flags_string`
   (`EvaluationDebateLoop-complete.py:436`, `:459`, call site `:648`).
2. **Test-harness false positives (test bug, not product).** The recording agent eagerly
   rendered all three templates in every phase, so round-1 records reported a bogus
   `KeyError('neighbors_messages')` for the debate template. Fixed to mirror production:
   first round renders system + first-round prompts, debate rounds render the debate prompt.

After the fixes: `PLACEHOLDER VALIDATION: PASS`, exit code 0.

## Results of Step 7 — validation test (real vLLM backbone, scripted defense model)

Artifacts (all test-only, under `tests/placeholder-test/`):
- `prompts/prompts-placeholder-test.json` — test prompt set containing `{topology_string}`,
  `{malicious_agents_string}`, `{flags_string}` in system, first-round and debate prompts
  (benign + malicious variants). No production prompt file touched.
- `test_placeholders.py` — runs 3-agent / 2-round debates through **both** pipelines:
  `EvaluationDebateLoop-complete.py` (with a scripted defense model that flags the true
  malicious agents after round 1) and `DebateDataGenerationLoop-complete.py` (no defense).
- `rendered-evidence.json` — every per-agent `format_data` + rendered prompts; assertions.
- `run.log` — full run log.

Observed result (2 runs, identical outcomes):
- Evaluation loop: malicious `[1]`; round 1 `flags_string="[]"`, round 1 topology = original
  adjacency; round 2 `flags_string="[1]"`, topology = adjacency after `modify_adjacency`
  (agent 1 isolated, `Weakly connected components: [0, 2]; [1]`).
- Generation loop: malicious `[2]`; both rounds `flags_string="[]"`, topology unchanged.
- All 12 agent turns rendered without missing-key errors.
- Additivity control check: existing production prompts (`prompts_gsm8k.json`,
  `prompts_blindguard.json`) render byte-identically with and without the new keys.

## Results of Step 6 — design documentation

`PROMPT_PLACEHOLDERS_DESIGN.md` (GAMMAF root) contains: the build sites table with line
references, per-placeholder computation rules and sample outputs, how-to-use instructions,
the additivity statement, and the full validation evidence (including real rendered examples
of all three placeholders).

## Results of Step 5 — implementation (one `-complete` copy per component, originals untouched)

- **`DebateDataGenerationLoop-complete.py`** (new single copy of `DebateDataGenerationLoop.py`):
  placeholder helpers at lines 82–211 (`build_topology_string` :205,
  `build_malicious_agents_string` :104, `build_flags_string` :109, plus private
  `_index_list_to_string`, `_adjacency_key`, `_weakly_connected_components`,
  `_build_topology_string_cached`); new keys in the round-1 dict (:298–300) and the debate-round
  dict (:379–381).
- **`EvaluationDebateLoop-complete.py`** (extended, same single copy as before):
  new keys in the round-1 dict (:378–380) and debate-round dict (:457–459, after adding the
  `flags` parameter at :436). Its `generate_random_topologies` binding now comes from the
  complete generation module via `_load_placeholder_generation_module()` (:19–41) so the
  placeholder logic has a single source of truth.
- **`TrainDataGeneration-complete.py`** (extended): `DebateOrchestration` now loaded from
  `DebateDataGenerationLoop-complete.py` via `_load_placeholder_generation_module()` (:56–72);
  hyphenated filename loaded by path (same pattern already used elsewhere in the repo).
- `MainEvaluation-complete.py` needed no change: it loads `EvaluationDebateLoop-complete.py`
  (standard + HPS paths) and never builds `format_data` itself.
- Legacy (`EvaluationDebateLoop.py`, `DebateDataGenerationLoop.py`, `HPSearch.py`,
  `EvaluationDebateLoop-HPS.py`, agents, `Utils.py`) intentionally untouched (R4/R6).
- `python -m py_compile` passes on all three files; `git diff` on tracked files is empty.

## Results of Step 4 — design (the exact rules)

- `topology_string`: purely descriptive adjacency text. Header line with agent count; a legend
  line describing the `sender -> receiver` edge notation; the full edge list; one line per agent
  `Agent i: receives messages from [...]; sends messages to [...]`; and, only when the graph has
  more than one weakly connected component, a component line. No instructions/guidance.
  Derived from the live adjacency (`adj[i][j]==1` ⇔ agent `i` receives from agent `j`, matching
  the neighbour computation). `None`/non-square → neutral "not available" sentence.
- `malicious_agents_string`: `[i1, i2, ...]` (sorted, deduplicated, ints) from the run's
  malicious assignment (`malicious_indexes` argument, falling back to `agent.is_malicious`).
  `[]` when none.
- `flags_string`: `[i1, ...]` from the defense-model flags vector at that point
  (`None`/all-zero → `[]`; hence empty in round 1 by construction).

## Results of Step 3 — codebase exploration

- Evaluation/debate loop builds the dict in `EvaluationDebateLoop-complete.py`
  (`LiveDebateOrchestration.generate_round_1_concurrent`,
  `generate_debate_round_concurrent`); the defense `flags` vector is produced after each round
  by `_predict_defense_model` and applied by `modify_adjacency` before the next round.
- Generation pipeline builds it in `DebateDataGenerationLoop.py`, which was imported by
  `TrainDataGeneration-complete.py` → required the new `-complete` copy + import re-point.
- Agents (`DebateAgent.py`, `DebateAgent-TA.py`) only consume the dict; `MainEvaluation-complete.py`,
  `Utils.py` judge prompts and `MA/`/`TA/` generators do not build per-turn dicts.
- The consolidated `--hps` path in `MainEvaluation-complete.py` uses the same
  `EvaluationDebateLoop-complete.py` class, so it is covered.
