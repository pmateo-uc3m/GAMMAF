# MS MARCO Contaminated-Passage Task Generator

Generates a controlled benchmark for evaluating whether an LLM-agent system can
detect **memory/data poisoning** in retrieved textual evidence.

The generator takes passages from
[`microsoft/ms_marco`](https://huggingface.co/datasets/microsoft/ms_marco) (the
`v2.1` subset) and uses an LLM to produce a *coordinated* set of contaminated
passages that all support a single **incorrect** answer. Each output entry pairs
the original ("safe") passages with the LLM-generated ("adversarial") passages so
a detection system can be trained/tested on the difference.

## Directory layout

```
MA/Task_generation/
├── config.json              # generation configuration
├── llm_settings.json        # LLM connection + sampling parameters
├── prompts_system.txt       # contamination system prompt (editable)
├── prompts_user.txt         # contamination user prompt template (editable)
├── prompts_answer_system.txt  # answer-generation system prompt (editable)
├── prompts_answer_user.txt  # answer-generation user prompt template (editable)
├── dataset_utils.py         # dataset loading + field validation
├── passage_selection.py     # deterministic passage selection
├── contamination_llm.py     # OpenAI-compatible client + JSON validation/retry
├── prompts.py               # prompt template loading/rendering
├── main.py                  # entry point
├── run_validation.py        # self-tests (no live LLM required)
└── README.md
```

## Dependencies

* `datasets` — MS MARCO loading (already in the project `requirements.txt`).
* `openai` — OpenAI-compatible client (a dependency of `langchain-openai`,
  already installed).
* `numpy` — deterministic RNG for selection.

No files outside `MA/Task_generation` are created or modified.

## Configuration: `config.json`

| Key | Meaning |
|-----|---------|
| `dataset.name` | Hugging Face dataset name (`microsoft/ms_marco`). |
| `dataset.config` | Subset/version (`v2.1`). |
| `dataset.split` | Split to read (`validation`). |
| `num_entries` | Target number of generated tasks. Skipped/failed entries are replaced with additional entries from the dataset until this many tasks are produced (or the dataset is exhausted). |
| `max_concurrent_calls` | Maximum number of LLM requests in flight (default 200). |
| `entry_selection_seed` | Seed for selecting which dataset entries are processed. |
| `output.dir` / `output.file_name` | Where the final benchmark JSON is written. |
| `llm_settings_file` | Path to the LLM settings JSON. |
| `prompts.system_prompt_file` / `prompts.user_prompt_file` | Contamination prompt template files. |
| `prompts.answer_system_prompt_file` / `prompts.answer_user_prompt_file` | Answer-generation prompt template files. |
| `retry.max_llm_attempts`, `retry.retry_delay_seconds`, `retry.backoff_factor` | LLM retry policy. |

## Configuration: `llm_settings.json`

Uses the OpenAI-compatible API format. Placeholders are filled in by the user.

| Key | Meaning |
|-----|---------|
| `host_url` | Base URL of the OpenAI-compatible server. |
| `api_key` | API key (may be a placeholder for local servers). |
| `model_name` | Model identifier. |
| `timeout` | Request timeout (seconds). |
| `temperature`, `top_p`, `max_tokens` | Sampling parameters (exposed, not hard-coded). |
| `max_retries` | Unused fallback; see `retry` in `config.json`. |
| `json_mode` | When `true`, use `response_format={"type":"json_object"}`. |

The host URL and model name are **never hard-coded in Python**; they come from
this file. The client works with any OpenAI-compatible local or remote server.

## Running the generator

```bash
python MA/Task_generation/main.py
```

Options:

```bash
python MA/Task_generation/main.py --config PATH   # custom config
python MA/Task_generation/main.py --limit 5       # quick test run (overrides num_entries)
python MA/Task_generation/main.py --out OUT.json  # override output path
```

## Output format

The final benchmark JSON is a list of entries with **only** the required fields:

```json
{
  "query_id": "...",
  "query": "...",
  "answers": ["..."],
  "safe_passages": ["...", "..."],
  "adv_passages": ["...", "..."]
}
```

* `query_id` — directly from MS MARCO.
* `query` — the original query.
* `answers` — the correct answer(s). For ordinary queries this is the original
  MS MARCO answer; see below for unanswerable ("No Answer Present.") queries.
  It is **never** replaced by the generated incorrect answer.
* `safe_passages` — the original selected passages (byte-for-byte unchanged).
* `adv_passages` — the LLM-generated contaminated passages, in the same order
  as `safe_passages`.

The generated incorrect answer is *not* stored in the output; it is reflected
only through the contaminated evidence. (Internally it is kept for debugging.)

### Unanswerable queries ("No Answer Present.")

MS MARCO v2.1 marks unanswerable queries with the placeholder answer
`"No Answer Present."`. For such entries the generator:

1. Asks the LLM to produce a single answer **from the safe (non-contaminated)
   passages** (using the answer-generation prompts).
2. Stores that generated answer in the output `answers` field.
3. Uses that generated answer as the ground truth the contamination must
   contradict.

Ordinary entries keep their original MS MARCO `answers` unchanged.

## Passage selection

For each selected entry:

1. Read `query`, `answers`, `is_selected` (0/1 list) and `passages`.
2. Select **all** passages with `is_selected == 1`, preserving their original
   order. These become `safe_passages` (byte-for-byte unchanged).
3. The number of passages per task therefore equals the number of relevant
   passages in the MS MARCO entry.

**Skipping policy:** if an entry has **no** selected passage (`is_selected` all
0), it is skipped and counted as `skipped (no selected passages)` in the
summary. Skipped or LLM-failed entries do **not** reduce the final task count:
`num_entries` is a target number of *generated* tasks, and additional entries
are consumed from the dataset (in the same seeded order) to replace them.

## Reproducibility

* Entry ordering uses `numpy.random.default_rng(entry_selection_seed)` to
  produce a deterministic permutation of the dataset; entries are consumed in
  that order until `num_entries` tasks are generated.
* Passage selection is fully deterministic — it simply returns every passage
  with `is_selected == 1`; no random sampling is involved.
* No uncontrolled global randomness is used.

Running with identical seeds and config yields identical entry ordering,
passage sets, and output ordering. LLM generation may vary with the
model/sampling parameters (`temperature`, etc.).

## LLM failures and retries

For each entry the LLM is asked to produce a JSON object containing a single
`incorrect_answer` and one `adv_passage` per safe passage. For unanswerable
entries it is also asked to produce a single `answer` from the safe passages.
Responses are parsed and validated:

* invalid JSON / no JSON object → fail,
* wrong number of passages → fail,
* missing/empty target answer → fail,
* non-string or empty passages → fail,
* missing/empty generated `answer` → fail.

On failure the request is retried up to `retry.max_llm_attempts` times with
exponential backoff. If all attempts fail, the entry is counted in
`LLM failures` and replaced with an additional entry from the dataset, so the
requested `num_entries` is still produced whenever the dataset has enough valid
entries. No entry is emitted unless `adv_passages` contains exactly as many
valid passages as `safe_passages`.

## Validation

```bash
python MA/Task_generation/run_validation.py
```

This runs schema checks, deterministic-selection checks, JSON
parse/validation checks, and output-schema checks without a live LLM.
