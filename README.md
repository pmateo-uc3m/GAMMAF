# GAMMAF: Graph-Based Anomaly Monitoring Benchmarking for LLM Multi-Agent Systems

## Overview

**GAMMAF** is a framework to:

1. **Generate synthetic multi-agent communication data** (debates) over different graph topologies.
2. **Benchmark topology-guided defenses** that detect and isolate malicious agents during live inference.

GAMMAF is an evaluation architecture (not a new defense by itself). It provides a data-generation pipeline and a defense-benchmarking pipeline that can be extended with new datasets, new output parsing, and new defense models.

<img width="100%" alt="functionDiagram_more" src="https://github.com/user-attachments/assets/23b3b611-1c28-4f5d-ba85-04b186d78afb" />

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

GAMMAF uses `langchain-openai`’s `ChatOpenAI`, so any **OpenAI-compatible** server works. For local inference, a typical setup is `vLLM` exposing an OpenAI-compatible endpoint.

## Pipeline 1: Training data generation

Entry point: `TrainDataGeneration.py`

This stage runs debates and optionally **embeds the agent "reason" fields** into sentence-level and token-level embeddings. Output is a pickle file (default: `data/train-data.pkl`) containing per-topology debate traces.

### Minimal generation config (YAML)

Create a YAML file (e.g., `config-examples/generation-config.yaml`):

```yaml
# Generation pipeline config (TrainDataGeneration.py)

timeout: 60
parallel_questions: 20
verbose: false

# Prompt template file (JSON) used by agents
prompts: prompts/prompts_gsm8k.json

# Dataset selection (must match a class TAG in DatasetManager.py)
dataset_tag: GSM8K
questions_random_seed: 28

# Output
save_data_dir: data
file_name: train-data.pkl

# Optional post-processing
process_text: true          # add embeddings with TextProcessingManager.RoundProcessor
clean_data: true            # drop debates with empty/invalid agent outputs
text_process_workers: 0     # 0 = auto; GPU processors will be forced to sequential

# Debate parameters
debate_config:
	num_agents: 5
	num_malicious: 2
	max_rounds: 3
	consensus_threshold: 1.0
	malicious_randomization_seed: 42

	# Question counts per topology
	n_questions: 50
	n_questions_random_topo: 50

	# Random topology generation (only used for the "random" topology)
	random_topo_seed: 24
	density:
		min: 0.3
		max: 0.7
```

Run:

```bash
python TrainDataGeneration.py config-examples/generation-config.yaml
```

## Pipeline 2: Defense benchmarking (training + live evaluation)

Entry point: `MainEvaluation.py`

This stage:

1. Loads **defense model trainers** from a directory (default: `defense-models/`).
2. Trains each defense model using an embedded per-model training config.
3. Runs **live multi-agent debates**, calling `defense_model.predict(...)` each round to flag and isolate agents.

### Minimal evaluation config (YAML)

Create a YAML file (e.g., `config-examples/evaluation-config.yaml`):

```yaml
# Main evaluation config (MainEvaluation.py)

models_directory: defense-models
output_file: results/eval-results.json

# One config section per defense model file in models_directory (file stem must match)
defense_model_train_configs:
	BlindGuard:
		pkl_train: data/train-data.pkl
		seed: 42
		device: cpu
		# Data perturbation (used to simulate anomalies in training)
		anomaly_rate: 0.2
		anomaly_scale: 0.5
		anomalize_data: true
		no_balance: false
		topologies: null

		# Training hyperparameters (required)
		input_dim: 1152
		hidden_dim: 256
		emb_dim: 128
		batch_size: 256
		val_split: 0.2
		num_epochs: 20
		temperature: 0.07
		learning_rate: 0.001
		weight_decay: 0.0001
		scheduler_t_max: 20

		# Optional reproducibility seeds
		# data_seed: 42
		# split_seed: 42
		# dataloader_seed: 42

		# Optional checkpointing
		save_model: false
		save_path: ""

	XG-Guard:
		pkl_train: data/train-data.pkl
		seed: 42
		device: cpu
		topologies: null

		# Training hyperparameters (required)
		feat_dim_s: 384
		feat_dim_t: 384
		hidden_dim: 256
		batch_size: 8
		val_split: 0.2
		num_epochs: 5
		learning_rate: 0.001
		weight_decay: 0.0001
		alpha: 0.5

		# Optional reproducibility seeds
		# split_seed: 42
		# dataloader_seed: 42

		# Optional checkpointing (training loop may be extended to use these)
		save_model: false
		save_path: ""

live_evaluation_config:
	timeout: 60
	prompts_file: prompts/prompts_gsm8k.json

	# Question loader
	questions_path: DatasetManager.py
	questions_dataset_tag: GSM8K
	questions_random_seed: 28

	# Debate + attacker settings
	num_agents: 5
	num_malicious_agents: 2
	malicious_seed: 123
	max_rounds: 3
	consensus_threshold: 1.0
	no_consensus_check: false
	check_consensus_only_unflagged: true

	# Defense settings
	top_k_defense: 2
	no_defense_baseline: true

	# Concurrency
	max_concurrent_inference: 20

	# Topologies
	new_random_each_question: false
	n_questions_on_random_topo: 50
	topologies_seed: 24
	density_range_for_random_topo: [0.3, 0.7]

	# Text processing used during live evaluation (embeds each round)
	text_processor_path: TextProcessingManager.py
	text_processor_class_name: RoundProcessor

	# Debug artifacts
	save_traces: false
	clean_debates_with_empty_responses: true
```

Run:

```bash
python MainEvaluation.py config-examples/evaluation-config.yaml
```

## Customized benchmarking

### Adding new defense architectures

Defense models are discovered dynamically from `models_directory` (e.g., `defense-models/`). Each `*.py` file is imported, and if it defines a `Master` class it will be trained and then evaluated.

**What you need to implement**

1. A file `defense-models/MyDefense.py`.
2. A `class Master` with:
	 - `__init__(config_path: str)` (loads a YAML config)
	 - `_run()` that returns a tuple `(metrics_dict, defense_model_instance)`
3. The returned `defense_model_instance` must implement:

```python
def predict(self, debate_embeddings, adjacency_matrix, top_k: int = 1):
		"""Return (flags, scores).

		flags: list[int] of length num_agents, where 1 means "flagged/malicious".
		scores: list[float] or array-like anomaly scores (higher = more anomalous), or None.
		"""
```

**Wire it into the main config**

Add a matching section under `defense_model_train_configs` where the key equals the file stem:

```yaml
defense_model_train_configs:
	MyDefense:
		pkl_train: data/train-data.pkl
		seed: 42
		device: cpu
		# ... your model-specific config
```

### Adding new model output processing

In GAMMAF there are two distinct “processing” points:

1. **Parsing the LLM text into** `{reason, answer}`.
2. **Embedding the parsed text** (turning `reason` into vectors used by defenses).

**1) Parsing**

- Both the generation pipeline and the live evaluation pipeline rely on the dataset loader’s `parse_model_output(...)`.
- The built-in loaders live in `DatasetManager.py` (e.g., `MMLULoader`, `GSM8KLoader`) and parse XML-like tags such as `<reason>` and `<answer>`.

To change the required format, implement/override `parse_model_output` in your dataset loader and update the prompt JSON (see `prompts/`) accordingly.

**2) Embedding (text processor)**

- Offline generation (`TrainDataGeneration.py`) can embed rounds if `process_text: true`.
- Live evaluation (`EvaluationDebateLoop.py`) embeds each round using a configurable processor class.

To customize embeddings, create a new processor class with a `process_round(round_data)` method that returns the same structure as `TextProcessingManager.RoundProcessor` (i.e., each agent dict includes `st_embedding` and `tk_embedding`). Then point the configs to it:

```yaml
text_processor_path: path/to/my_processor.py
text_processor_class_name: MyRoundProcessor
```

### Adding new tasks datasets

There are two supported ways to plug in datasets:

**A) Add a new loader to DatasetManager.py (used by training data generation)**

`DebateDataGenerationLoop.py` discovers dataset loaders by inspecting classes in `DatasetManager.py` that define a `TAG` field.

To add a dataset for the generation pipeline, add a new class with:

- `TAG = "MYDATASET"`
- `get_formatted_questions()` returning a list of dicts with (at minimum): `question`, `choices`, `answer`
- `parse_model_output(message)` returning `ResponseFormat(reason=..., answer=...)`
- `validate_answer(model_answer, correct_answer)`

Then set in your generation YAML:

```yaml
dataset_tag: MYDATASET
```

**B) Load a questions loader from an arbitrary file (used by live evaluation)**

`EvaluationDebateLoop.py` can load a questions loader class from a Python file path using either:

- `questions_dataset_tag` (matches a class’s `TAG`), or
- `questions_class_name` (explicit class name)

Config keys:

```yaml
questions_path: path/to/my_questions.py
questions_dataset_tag: MYDATASET
# or:
# questions_class_name: MyQuestionsLoader
```
