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
├── prompts_system.txt       # system prompt (editable)
├── prompts_user.txt         # user prompt template (editable)
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
| `n` | Number of passages per generated task. |
| `num_entries` | Total MS MARCO entries to process. |
| `entry_selection_seed` | Seed for selecting which dataset entries are processed. |
| `passage_selection_seed` | Seed for selecting passages within each entry. |
| `output.dir` / `output.file_name` | Where the final benchmark JSON is written. |
| `llm_settings_file` | Path to the LLM settings JSON. |
| `prompts.system_prompt_file` / `prompts.user_prompt_file` | Prompt template files. |
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
* `answers` — the original correct answer(s); **never** replaced by the
  generated incorrect answer.
* `safe_passages` — the original selected passages (byte-for-byte unchanged).
* `adv_passages` — the LLM-generated contaminated passages, in the same order
  as `safe_passages`.

The generated incorrect answer is *not* stored in the output; it is reflected
only through the contaminated evidence. (Internally it is kept for debugging.)

## Passage selection

For each selected entry:

1. Read `query`, `answers`, `is_selected` (0/1 list) and `passages`.
2. Prioritize passages with `is_selected == 1`.
3. Include as many selected passages as possible within the `n`-passage budget.
4. Fill remaining slots with randomly chosen non-selected passages.
5. Return exactly `n` passages.

**Skipping policy:** if an entry has fewer than `n` available passages, it is
skipped (counted in the summary) rather than silently producing an incorrectly
sized example.

## Reproducibility

* Entry selection uses `numpy.random.default_rng(entry_selection_seed)`.
* Passage selection uses `numpy.random.default_rng(passage_selection_seed)`.
* No uncontrolled global randomness is used.

Running with identical seeds and config yields identical entry/passage
selection. LLM generation may vary with the model/sampling parameters
(`temperature`, etc.).

## LLM failures and retries

For each entry the LLM is asked to produce a JSON object containing a single
`incorrect_answer` and exactly `n` `adv_passages`. The response is parsed and
validated:

* invalid JSON / no JSON object → fail,
* wrong number of passages → fail,
* missing/empty target answer → fail,
* non-string or empty passages → fail.

On failure the request is retried up to `retry.max_llm_attempts` times with
exponential backoff. If all attempts fail, the entry is skipped and counted in
`LLM failures`; generation continues with the next entry. No entry is emitted
unless both `safe_passages` and `adv_passages` contain exactly `n` valid
passages.

## Validation

```bash
python MA/Task_generation/run_validation.py
```

This runs schema checks, deterministic-selection checks, JSON
parse/validation checks, and output-schema checks without a live LLM.
