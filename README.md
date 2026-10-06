# GAMMAF: Graph-Based Anomaly Monitoring Benchmarking for LLM Multi-Agent Systems

<div align="center">
  <img width="650" height="350" alt="logo" src="https://github.com/user-attachments/assets/b32d042a-6ec2-4d3a-af6b-fbfe09c5e4f8" />
</div>

<br>

This repository contains the source code for our [paper](https://arxiv.org/abs/2604.24477) introducing the **GAMMAF** framework.

## Overview

**GAMMAF** is an evaluation framework (not a defense by itself) to:

1. **Generate synthetic multi-agent communication data** (debates) over different graph topologies.
2. **Benchmark topology-guided defenses** that detect and isolate malicious agents during live inference.

Main components: a data-generation pipeline, a defense-benchmarking pipeline, an unsupervised hyperparameter search, pluggable task datasets, pluggable text-to-embedding processors, and pluggable defense models.

<img width="5711" height="6096" alt="functionDiagram-major" src="https://github.com/user-attachments/assets/fdfc80a0-02bb-4f18-8096-c4656f214a2a" />

### Project structure

| Path | Purpose |
| --- | --- |
| `TrainDataGeneration.py` | Training-data generation entry point (multi-dataset). |
| `DebateDataGenerationLoop.py` | Debate orchestration used during generation. |
| `MainEvaluation.py` | Defense-benchmarking entry point (multi-dataset, crash-safe resume). |
| `EvaluationDebateLoop.py` | Live debate orchestration and metric computation. |
| `HyperParameterSearch.py` | Unsupervised (AutoUAD/NPD) hyperparameter search. |
| `DatasetManager.py` | Task dataset loaders (`MMLU`, `CSQA`, `GSM8K`, `MMLUPro`, `MA`, `TA`). |
| `TextProcessingManager.py` | Text-to-embedding processors. |
| `GenerationConfigCheck.py`, `EvaluationConfigCheck.py` | YAML config loading/validation. |
| `defense-models/` | Defense model implementations evaluated by the framework. |
| `config-examples/` | Example YAML configurations. |
| `prompts/` | Prompt templates (JSON). |
| `experiments-Information-Sciences/` | Configs and result summaries for the *Information Sciences* submission. |
| `auxiliary/` | Result analysis/display helpers. |

## Requirements

- Python 3.11 (recommended)
- An OpenAI-compatible API endpoint (remote or local)

Notes:

- On Windows, `torch-geometric` may require a matching PyTorch + CUDA setup depending on your hardware.
- The generation pipeline can run on CPU, but embeddings (SentenceTransformers) are much faster on GPU.

## Installation

Create an environment and install dependencies:

```bash
conda create -n gammaf-env python=3.11
conda activate gammaf-env
pip install -r requirements.txt
```

## Configure your LLM backend

Create a `.env` file (see [.env.example](.env.example)):

```ini
BASE_URL="http://localhost:8000/v1"
MODEL_NAME="openai/gpt-oss-20b"
API_KEY="your_api_key_here"
```

GAMMAF uses `langchain-openai`'s `ChatOpenAI`, so any **OpenAI-compatible** server works. For local inference, a typical setup is `vLLM` exposing an OpenAI-compatible endpoint.

## Main scripts

### Training data generation: `TrainDataGeneration.py`

Runs multi-agent debates for one or more datasets in a single run, one debate per selected question and topology (`chain`, `star`, `tree`, plus `random`, unless disabled or replaced by topologies loaded from a file). Optionally embeds every agent message with the configured text processor and drops debates with empty responses (`clean_debates`, never applied to `TA`). It saves one pickle with the debates, a `.idx_metadata.json` sidecar recording the dataset indexes used per tag (for leak-free training/evaluation later), and a timing report.

Run:

```bash
python TrainDataGeneration.py config-examples/generation-config.yaml
```

Config schema:

| Key | Meaning |
| --- | --- |
| `llm.{timeout, llm_max_retries, max_concurrent_inference}` | Inference API settings. |
| `debate.{num_agents, num_malicious_agents, malicious_seed, max_rounds, consensus_threshold, random_topo_seed}` | Debate setup. |
| `debate.density_range_for_random_topo` or `debate.average_neighbors` | Random-topology density (one is required). |
| `datasets[]` | One run per entry: `tag` (unique), optional `loader_tag`, `num_questions`, `num_questions_on_random_topo`, `questions_random_seed`, `ma_dataset_path`, `prompts_file`; optionally `load_topology_file` + `num_questions_loaded_topo` to use custom topologies. |
| `output_dir`, `output_file` | Output location (must end in `.pkl`). |
| `process_text`, `clean_debates` | Embed messages / drop empty-response debates. |
| `text_processor_path`, `text_processor_class_name`, `text_processor_kwargs`, `text_processor_device`, `text_process_workers` | Which processor to use and how to run it (`0` = auto; non-CPU processors run sequentially). |
| `verbose` | Verbose logging. |

The pickle has the shape `{data, idx_metadata, idx_metadata_flat, dataset_tags}`, where each entry of `data` groups the debates of one topology and carries its `dataset_tag`.

### Defense benchmarking: `MainEvaluation.py`

Loads every defense model found in `models_directory`, trains each one once on the combined training pickle, then runs live debates for each configured dataset tag and topology, isolating flagged agents. Per dataset tag it excludes the indexes used for training on that tag (read from the sidecar of `train_pkl_path`) and any per-dataset `hps_indexes`, so no evaluated question was used for training/tuning. Results are written atomically to a single summary JSON: completed model+dataset combinations are resumed on re-run, and optional per-item score artifacts are saved when `save_scores_artifact: true`.

Run:

```bash
python MainEvaluation.py config-examples/evaluation-config.yaml
python MainEvaluation.py config-examples/evaluation-config.yaml --clean  # start fresh
```

Config schema:

| Key | Meaning |
| --- | --- |
| `models_directory`, `output_file` | Where defense models live and where results are written. |
| `train_pkl_path` | Global training pickle used by models without their own `pkl_train`. |
| `llm`, `debate` | Same debate setup as generation, plus `check_consensus_only_unflagged`, `no_consensus_check`, `new_random_each_question`, `clean_debates`. |
| `datasets[]` | Same entries as generation, plus optional `hps_indexes` (per-tag HPS exclusion pickle). |
| `evaluation.{questions_path, questions_class_name}` | Loader source for live questions; class name optional if `tag`/`loader_tag` matches a loader `TAG`. |
| `evaluation.{python_seed, numpy_seed, answer_seed}` | Reproducibility seeds. |
| `evaluation.top_k_defense` | Global top-k injected into every defense model as `model.config.top_k`. |
| `evaluation.no_defense_baseline` | Also run a no-defense baseline. |
| `evaluation.{save_traces, save_scores_artifact, debug_mode, static_adjacency_mode, topologies_file, topologies_from_pkl}` | Logging/storage/topology options. |
| `text_processor_*` | Same as generation. |
| `training` | Global training defaults (LR reduction / early stopping), overridable per model. |
| `defense_model_train_configs.<FileStem>` | One mapping per model file (a list of mappings runs the same model several times); `seed` is required and `pkl_train` falls back to `train_pkl_path`. |

### Hyperparameter search: `HyperParameterSearch.py`

Selects defense-model hyperparameters **without ground-truth labels**. The training debates are split at debate level (stratified by topology and dataset tag) into train/validation; a synthetic Gaussian proxy `X_gen` is sampled from the training statistics with the validation graph structures. Optuna (TPESampler, maximize) searches the configuration space using the Normalized Pseudo Discrepancy between held-out validation scores and synthetic-anomaly scores. The best configuration is then retrained on the full dataset. Re-running with the same `output_file` resumes the sweep.

Run:

```bash
python HyperParameterSearch.py config-examples/hyperparameter-search-config.yaml
python HyperParameterSearch.py config-examples/hyperparameter-search-config.yaml --clean  # start fresh
```

Config schema:

| Key | Meaning |
| --- | --- |
| `data_path`, `output_file` | Training pickle to search on and results JSON. |
| `models_directory` | Where the defense models live (default `defense-models`). |
| `algorithm.{validation_split_ratio, epsilon, split_seed, gaussian_seed, max_debates, feature_key, save_final_models_dir}` | NPD search parameters and optional final-model checkpoints. |
| `optuna.{n_trials, sampler_seed, timeout, early_stopping_patience, early_stopping_min_improvement_pct}` | Optuna budget and early stopping (`n_trials` required). |
| `training.*` | Global model-training defaults, overridable per model. |
| `models.<ModelName>` | One section per file stem. Scalar values stay fixed; list values are searched. Reserved per-model keys: `seed`, `feature_key`/`feature_keys`, `n_trials`, `split_seed`, `gaussian_seed`. |

Output JSON records each model's trials, best trial and the final retrained model (parameters, plus a checkpoint only if the model has `save_model` and `save_final_models_dir` is set).

### How the three scripts relate

`TrainDataGeneration.py` produces the embedded training pickle and its used-index sidecar. `HyperParameterSearch.py` (optional) consumes that pickle to select defense hyperparameters without labels. `MainEvaluation.py` consumes the pickle to train every defense model and then benchmarks them on the configured datasets, excluding both the training indexes and any per-dataset `hps_indexes` so that no evaluated question was used for training or tuning.

### Custom topologies from JSON

Instead of the generated `chain`/`star`/`tree` set, any dataset entry can run on pre-defined topologies loaded from a JSON file:

| Key (per dataset entry) | Meaning |
| --- | --- |
| `load_topology_file` | JSON file with the topologies to use for this dataset entry. |
| `num_questions_loaded_topo` | Questions per loaded topology (required when the file is set). |

The file is a non-empty JSON list of entries, each with a unique `name` and either `topology` or `topologies` (alias). A `topology` may be a single adjacency matrix or a list of matrices:

```json
[
  {"name": "chain", "topology": [[0,1,0],[1,0,1],[0,1,0]]},
  {"name": "variants", "topologies": [
    [[0,1,0],[1,0,1],[0,1,0]],
    [[0,1,1],[1,0,0],[1,0,0]]
  ]}
]
```

```yaml
datasets:
  - tag: MMLU
    # ... num_questions / num_questions_on_random_topo / questions_random_seed
    load_topology_file: my_topologies.json
    num_questions_loaded_topo: 20
```

Every matrix must be square, contain only 0/1 integers and match `debate.num_agents`. When an entry holds several matrices, each debate samples one uniformly using `debate.random_topo_seed`. Loaded topologies replace the generated set for that dataset entry (the standard `num_questions`, `num_questions_on_random_topo` and `questions_random_seed` keys are still required by the config validator). In evaluation, per-dataset loaded topologies take precedence over the global `evaluation.topologies_file`, and `evaluation.topologies_from_pkl` can reuse the topologies of an existing training pickle.

## Extending the framework

### Adding a new defense model to `defense-models`

1. Create `defense-models/MyDefense.py`.
2. Implement a `Master` class that loads its YAML config and returns a trained model instance.
3. The returned model must implement `predict(debate_embeddings, adjacency_matrix)` and return `(flags, anomaly_scores)`.
4. Add a matching section under `defense_model_train_configs` in the evaluation YAML, keyed by the file stem (`MyDefense`).

Minimal skeleton:

```python
# defense-models/MyDefense.py
from EvaluationConfigCheck import load_defense_model_config


class MyDefenseModel:
    def __init__(self, config):
        self.config = config

    def predict(self, debate_embeddings, adjacency_matrix):
        """debate_embeddings: list of per-agent dicts (agent_id, answer,
        st_embedding, and tk_embedding when RoundProcessor is used).
        adjacency_matrix: n_agents x n_agents matrix.

        Returns (flags, anomaly_scores): flags has one 0/1 entry per agent
        (1 = flagged/malicious) and scores are anomaly scores where higher
        means more anomalous (used for AUROC)."""
        scores = ...  # your scoring logic
        flags = ...   # e.g. top-k highest scores set to 1
        return flags, scores


class Master:
    def __init__(self, config_path):
        self.args = load_defense_model_config(config_path)

    def _run(self, train_pkl_path=None):
        # Train using train_pkl_path or self.args.pkl_train.
        # The optional metrics dict may include "computed_threshold" to rename the run.
        return {}, MyDefenseModel(self.args)
```

```yaml
defense_model_train_configs:
  MyDefense:               # must match defense-models/MyDefense.py
    pkl_train: data/train-data.pkl  # optional if top-level train_pkl_path is set
    seed: 42                # required
    device: cpu
    top_k: 2
    # ... any model-specific keys
```

Notes:

- `evaluation.top_k_defense` is injected as `model.config.top_k` before every prediction; the model instance should expose a `config` (as the built-in models do).
- `predict` is called once per debate round with the embedded round, so it must handle inputs grouped by round.
- Optional extras used by the framework: `begin_trace(trace_id, adjacency_matrix)` / `end_trace(trace_id)` (or `reset()`) to isolate per-question state, `predict(..., trace_id=None)` to receive the trace id, `save_model(path)` for HPS checkpoints, and a `threshold` attribute (on the model or `model.config`) to flag by threshold instead of top-k.
- Files without a `Master` class, or without a matching `defense_model_train_configs` section, are skipped.

### Adding a new task dataset to `DatasetManager.py`

1. Add a loader class with a unique `TAG` to `DatasetManager.py` (subclassing an existing loader, e.g. `MMLULoader`, is the easiest path).
2. Set `self.indexes`, `self.questions` and `self.formatted_questions` in the constructor, honoring the requested `num_questions` and `random_seed`.
3. Make `get_formatted_questions()` return one dict per question with at least `question` and `choices`, plus the ground truth where applicable (`answer`, or `correct_answer` for judge-based datasets).
4. Select it in the config through `datasets[].tag` (or `loader_tag`).

Minimal skeleton:

```python
# DatasetManager.py
class MyDatasetLoader(MMLULoader):
    TAG = "MYDATASET"
    PROMPTS_FILE = "prompts/prompts_MY.json"

    def load_questions(self):
        # Build self.questions and sample self.num_questions from
        # available indexes (excluding self.indexes); store the selection
        # in self.indexes and return the selected questions.
        ...

    def format_questions(self):
        return [
            {
                "question_index": i,
                "question": q["question"],
                "choices": q["choices"],  # prompt placeholder
                "answer": q["answer"],    # ground truth (excluded from prompts)
                # any extra key becomes a prompt placeholder
            }
            for i, q in enumerate(self.questions)
        ]

    def parse_model_output(self, message):
        return default_parse_model_output(message, self.RESPONSE_FORMAT)

    def agent_is_safe(self, response_data):
        return response_data["response"]["answer"] == response_data["correct_answer"]
```

```yaml
datasets:
  - tag: MYDATASET
    loader_tag: null
    num_questions: 10
    num_questions_on_random_topo: 10
    questions_random_seed: 1
```

Notes:

- In live evaluation the constructor is called with `num_questions`, `random_seed` and `indexes`; during generation only `num_questions` and `random_seed`. `dataset_path` is forwarded from `datasets[].ma_dataset_path` when the signature accepts it.
- Inherited methods the loops rely on: `get_prompts()`, `parse_model_output(...)`, `agent_is_safe(...)`, `is_answer_correct(...)` and `get_formatted_questions()`. Set `RESPONSE_FORMAT` (default `ResponseFormat`) and, for tool-calling datasets, `SUPPORTS_TOOL_CALLS = True` (see `InjecAgentLoader`/`TAResponseFormat`).
- Keys returned by `get_formatted_questions()` are merged into the prompt format dictionary, so they can be referenced as `{placeholder}` in the prompt JSON. `answer` is intentionally excluded to avoid leaking labels.
- For evaluation only, a loader can also live in an arbitrary file selected with `evaluation.questions_path` plus `evaluation.questions_class_name`.

### Adding a new text processing class to `TextProcessingManager.py`

1. Add a class with a `process_round(round_data)` method.
2. It receives the list of agent dicts of one round (including their `message`) and must return the same list with embeddings.
3. Select it in the YAML via `text_processor_path` / `text_processor_class_name` (plus optional `text_processor_kwargs` and `text_processor_device`).

Minimal skeleton:

```python
# TextProcessingManager.py
class MyRoundProcessor:
    def __init__(self, device="cpu", **kwargs):
        self.device = device

    def process_round(self, round_data):
        processed = []
        for agent in round_data:
            out = {key: value for key, value in agent.items() if key != "message"}
            out["st_embedding"] = ...  # 1-D pooled sentence embedding
            out["tk_embedding"] = ...  # 2-D per-token embeddings (optional)
            processed.append(out)
        return processed
```

```yaml
text_processor_path: TextProcessingManager.py
text_processor_class_name: MyRoundProcessor
text_processor_kwargs: {}
text_processor_device: cpu
```

Notes:

- `process_round` runs before every defense `predict` in live evaluation, and before saving each debate when `process_text: true` in generation.
- The built-in `RoundProcessor` produces `st_embedding` and `tk_embedding`; the built-in `SentenceOnlyRoundProcessor` produces only `st_embedding`.
- **`BlindGuard` and `XG-Guard` require `st_embedding`; `XG-Guard` also requires `tk_embedding`. Any custom processor used with them must produce those keys.**

## Future work

- Expand the framework to support heterogeneous agents and agents with different roles.
- Expand the framework to support more channels (such as tools and memory), not only communication.
- Increase efficiency and support process sharding for bigger multi-agent systems.

## Timeline

* 🚀 **21st Apr 2026:** First version published.
* 📤 **13th May 2026:** Submitted to *Information Sciences*.
* ✨ **29th Sep 2026:** Second version published.
* 📝 **1st Oct 2026:** Review submitted to *Information Sciences*.

## License

This project is released under the Apache 2.0 License. See [LICENSE](LICENSE).

## Citations

If you liked this repository and used our work for your publications, please cite our framework as:

```
@misc{mateotorrejón2026gammafcommonframeworkgraphbased,
      title={GAMMAF: A Common Framework for Graph-Based Anomaly Monitoring Benchmarking in LLM Multi-Agent Systems}, 
      author={Pablo Mateo-Torrejón and Alfonso Sánchez-Macián},
      year={2026},
      eprint={2604.24477},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2604.24477}, 
}
```

## Acknowledgements

This work has been supported by R&D project PID2022-136684OB-C21 (Fun4Date-Redes) funded by the Spanish Ministry of Science and Innovation MCIN/AEI/10.13039/501100011033 and TUCAN6-CM (TEC-2024/COM-460), funded by Comunidad de Madrid (ORDEN 5696/2024). The authors would like to thank SLICES-Madrid (https://slices-madrid.eu/), the main site of SLICES-Spain, part of the ESFRI SLICES-RI project, for the use of AI Training Research Infrastructure.

