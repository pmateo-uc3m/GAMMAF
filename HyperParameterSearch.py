"""HyperParameterSearch.py -- unsupervised AutoUAD/NPD hyperparameter search.

Selects defense-model hyperparameters **without using ground-truth labels**.
A synthetic isotropic Gaussian dataset acts as a proxy for unseen anomalies
and Normalized Pseudo Discrepancy (NPD) ranks candidate configurations by how
well the trained model separates the held-out normal validation set from the
synthetic set:

    V_NPD = (mean(s_gen) - mean(s_val))^2 / (2 * (var(s_gen) + var(s_val)) + eps)

with ``s_val = f_M(X_val | Theta)`` and ``s_gen = f_M(X_gen | Theta)``, where
higher anomaly scores mean "more anomalous" for every defense model.

Search is driven by Optuna (TPESampler, direction="maximize") for the
configured trial budget (a per-model ``n_trials`` can override the global
one).  When the budget exceeds the number of distinct search-space
combinations, TPE may repeat configurations; this is reported in the logs so
the budget can be tuned, but the search itself is not truncated.  After the
search the best configuration is retrained on the full feature matrix; that
retrained model (params + optional checkpoint) is reported in the results
JSON.  Trial NPDs are the quality metrics: no NPD is reported for the
retrained model because X_val is part of its training data, which would make
that value in-sample and not comparable to the trial NPDs.

Config schema (see ``config-examples/hyperparameter-search-config.yaml``)::

    data_path: paper-experiments/exp1/exp1-train-data.pkl
    models_directory: defense-models
    output_file: hps/hps-results.json

    algorithm:
      validation_split_ratio: 0.3
      epsilon: 1.0e-9
      split_seed: 42
      gaussian_seed: 42
      max_samples: null
      feature_key: st_embedding       # default primary key (see feature_keys)
      round_size: 10
      score_chunk_size: 10
      synthetic_topology: complete
      save_final_models_dir: null

    optuna:
      n_trials: 50
      sampler_seed: 42
      timeout: null

    models:                          # one section per defense model file stem
      BlindGuard:
        seed: 42
        hidden_dim: [64, 128, 256]   # list value -> searched
        learning_rate: [0.001]       # list value -> searched
        input_dim: 1152              # scalar -> held constant
        ...
      XG-Guard:                      # multi-key model
        feature_keys: [st_embedding, tk_embedding]
        n_trials: 10                 # optional per-model Optuna budget override
        ...

Per-model feature keys
----------------------
The first key in ``feature_keys`` is the *primary* key: its fixed-length
vectors form the tabular matrix used by the AutoUAD/NPD algorithm
(standardization, split, Gaussian proxy, NPD). Every key in the list is
attached to the per-agent dicts handed to the model, exactly as the live
evaluation pipeline does. ``feature_key: <key>`` is a shorthand for a
single-entry list; if neither is given, the global
``algorithm.feature_key`` is used. Only the primary key exists in the
synthetic Gaussian set, so extra keys are synthesized from the generated
primary vectors (matching the single-key behaviour of earlier versions).
"""

import argparse
import gc
import importlib.util
import inspect
import json
import pickle
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import time

import numpy as np
import optuna
import yaml
from sklearn.preprocessing import StandardScaler

from EvaluationConfigCheck import write_model_config
from LoggingUtils import (
    fmt_seconds,
    log_config,
    log_done,
    log_error,
    log_info,
    log_section,
    log_warn,
)
from Utils import AttrDict


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

_ROOT_KEYS = {
    "data_path",
    "models_directory",
    "output_file",
    "algorithm",
    "optuna",
    "models",
}
_ROOT_REQUIRED = {"data_path", "output_file", "models"}

_ALGORITHM_KEYS = {
    "validation_split_ratio",
    "epsilon",
    "split_seed",
    "gaussian_seed",
    "max_samples",
    "feature_key",
    "round_size",
    "score_chunk_size",
    "synthetic_topology",
    "save_final_models_dir",
}

_OPTUNA_KEYS = {"n_trials", "sampler_seed", "timeout"}

_MODEL_RESERVED_KEYS = {"feature_key", "feature_keys", "n_trials"}

_TOPOLOGIES = {"complete", "chain", "star"}


def _load_yaml(config_path):
    path = Path(config_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"Hyperparameter-search configuration does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in '{path}': {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Hyperparameter-search configuration root must be a mapping")
    return data


def _reject_unknown(mapping, allowed, location):
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown configuration key(s) at {location}: {', '.join(unknown)}")


def _require_mapping(parent, key, location):
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration field '{location}.{key}' must be a mapping")
    return value


def _require_int(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Configuration field '{name}' must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"Configuration field '{name}' must be >= {minimum}")
    return value


def _require_number(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Configuration field '{name}' must be a number")
    if minimum is not None and value < minimum:
        raise ValueError(f"Configuration field '{name}' must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"Configuration field '{name}' must be <= {maximum}")
    return float(value)


def _require_str(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration field '{name}' must be a non-empty string")
    return value


def _require_path(value, name):
    _require_str(value, name)
    if not Path(value).exists():
        raise ValueError(f"Configured path does not exist: {name}={value}")
    return value


def _validate_algorithm(raw):
    _reject_unknown(raw, _ALGORITHM_KEYS, "algorithm")
    ratio = _require_number(
        raw.get("validation_split_ratio", 0.3),
        "algorithm.validation_split_ratio",
        minimum=0.0,
        maximum=1.0,
    )
    if not 0.0 < ratio < 1.0:
        raise ValueError("Configuration field 'algorithm.validation_split_ratio' must be in (0, 1)")
    epsilon = _require_number(raw.get("epsilon", 1e-9), "algorithm.epsilon", minimum=0.0)
    if epsilon <= 0.0:
        raise ValueError("Configuration field 'algorithm.epsilon' must be > 0")
    split_seed = _require_int(raw.get("split_seed", 42), "algorithm.split_seed")
    gaussian_seed = _require_int(raw.get("gaussian_seed", split_seed), "algorithm.gaussian_seed")
    round_size = _require_int(raw.get("round_size", 10), "algorithm.round_size", minimum=2)
    score_chunk_size = _require_int(
        raw.get("score_chunk_size", round_size), "algorithm.score_chunk_size", minimum=2
    )
    max_samples = raw.get("max_samples")
    if max_samples is not None:
        max_samples = _require_int(max_samples, "algorithm.max_samples", minimum=2)
    topology = _require_str(raw.get("synthetic_topology", "complete"), "algorithm.synthetic_topology")
    if topology not in _TOPOLOGIES:
        raise ValueError(
            f"Configuration field 'algorithm.synthetic_topology' must be one of {sorted(_TOPOLOGIES)}"
        )
    save_dir = raw.get("save_final_models_dir")
    if save_dir is not None:
        _require_str(save_dir, "algorithm.save_final_models_dir")
    return AttrDict(
        validation_split_ratio=ratio,
        epsilon=epsilon,
        split_seed=split_seed,
        gaussian_seed=gaussian_seed,
        max_samples=max_samples,
        feature_key=_require_str(raw.get("feature_key", "st_embedding"), "algorithm.feature_key"),
        round_size=round_size,
        score_chunk_size=score_chunk_size,
        synthetic_topology=topology,
        save_final_models_dir=save_dir,
    )


def _validate_optuna(raw):
    _reject_unknown(raw, _OPTUNA_KEYS, "optuna")
    if "n_trials" not in raw:
        raise ValueError("Missing required field: 'optuna.n_trials'")
    n_trials = _require_int(raw["n_trials"], "optuna.n_trials", minimum=1)
    sampler_seed = _require_int(raw.get("sampler_seed", 42), "optuna.sampler_seed")
    timeout = raw.get("timeout")
    if timeout is not None:
        timeout = _require_number(timeout, "optuna.timeout", minimum=0.0)
    return AttrDict(n_trials=n_trials, sampler_seed=sampler_seed, timeout=timeout)


def _resolve_feature_keys(model_cfg, location, default_feature_key):
    has_single = "feature_key" in model_cfg
    has_multi = "feature_keys" in model_cfg
    if has_single and has_multi:
        raise ValueError(
            f"Configuration at '{location}' cannot set both 'feature_key' and 'feature_keys'"
        )
    if has_multi:
        raw = model_cfg["feature_keys"]
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"Configuration field '{location}.feature_keys' must be a non-empty list")
        keys = [
            _require_str(value, f"{location}.feature_keys[{index}]")
            for index, value in enumerate(raw)
        ]
    elif has_single:
        keys = [_require_str(model_cfg["feature_key"], f"{location}.feature_key")]
    else:
        keys = [default_feature_key]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Configuration field '{location}.feature_keys' contains duplicates: {keys}")
    return keys


def _validate_models(raw, models_directory, default_feature_key):
    if not isinstance(raw, dict) or not raw:
        raise ValueError("'models' must be a non-empty mapping of model_name -> search config")
    models = AttrDict()
    for model_name, model_cfg in raw.items():
        _require_str(model_name, "models key")
        if not isinstance(model_cfg, dict) or not model_cfg:
            raise ValueError(f"Model configuration '{model_name}' must be a non-empty mapping")
        model_file = Path(models_directory) / f"{model_name}.py"
        if not model_file.exists():
            raise ValueError(
                f"Model configuration '{model_name}' has no matching file: {model_file}"
            )

        entry = dict(model_cfg)
        feature_keys = _resolve_feature_keys(entry, f"models.{model_name}", default_feature_key)
        per_model_trials = entry.get("n_trials")
        if per_model_trials is not None:
            per_model_trials = _require_int(
                per_model_trials, f"models.{model_name}.n_trials", minimum=1
            )
        for key in _MODEL_RESERVED_KEYS:
            entry.pop(key, None)
        if "seed" not in entry:
            entry["seed"] = 42

        normalized = AttrDict(entry)
        normalized["feature_keys"] = feature_keys
        normalized["n_trials"] = per_model_trials
        models[model_name] = normalized
    return models


def load_hps_config(config_path):
    raw = _load_yaml(config_path)
    _reject_unknown(raw, _ROOT_KEYS, "root")
    for key in _ROOT_REQUIRED:
        if key not in raw:
            raise ValueError(f"Missing required top-level configuration field: '{key}'")

    models_directory = _require_path(raw.get("models_directory", "defense-models"), "models_directory")
    if not Path(models_directory).is_dir():
        raise ValueError(f"Configuration field 'models_directory' is not a directory: {models_directory}")

    algorithm = _validate_algorithm(_require_mapping(raw, "algorithm", "root"))
    return AttrDict(
        data_path=_require_path(raw["data_path"], "data_path"),
        models_directory=models_directory,
        output_file=_require_str(raw["output_file"], "output_file"),
        algorithm=algorithm,
        optuna=_validate_optuna(_require_mapping(raw, "optuna", "root")),
        models=_validate_models(raw["models"], models_directory, algorithm.feature_key),
    )


# ---------------------------------------------------------------------------
#  Data preparation
# ---------------------------------------------------------------------------

def _as_feature_vector(value):
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or array.size == 0:
        return None
    return array


def load_feature_data(data_path, feature_keys, max_samples, seed):
    """Load every key a model needs, aligned row-by-row with the primary matrix.

    ``feature_keys[0]`` is the primary key whose vectors form the tabular
    matrix used by the AutoUAD/NPD algorithm; the remaining keys are carried
    alongside (e.g. variable-length token embeddings) for the model.
    """
    primary_key = feature_keys[0]
    auxiliary_keys = list(feature_keys[1:])

    with open(data_path, "rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict):
        records = payload.get("data", [])
        dataset_tags = [str(tag) for tag in payload.get("dataset_tags", [])]
    else:
        records = payload
        dataset_tags = []

    vectors = []
    auxiliary = {key: [] for key in auxiliary_keys}
    skipped = 0
    for entry in records:
        if not isinstance(entry, dict):
            continue
        for debate in entry.get("results", []) or []:
            if not isinstance(debate, dict):
                continue
            for debate_round in debate.get("debate_rounds", []) or []:
                for agent in debate_round or []:
                    vector = _as_feature_vector(agent.get(primary_key)) if isinstance(agent, dict) else None
                    if vector is None:
                        skipped += 1
                        continue
                    payloads = {}
                    missing = False
                    for key in auxiliary_keys:
                        if agent.get(key) is None:
                            missing = True
                            break
                        payloads[key] = agent[key]
                    if missing:
                        skipped += 1
                        continue
                    vectors.append(vector)
                    for key, value in payloads.items():
                        auxiliary[key].append(value)

    if not vectors:
        raise ValueError(f"No usable '{primary_key}' samples found in {data_path}")

    dimensions = {vector.shape[0] for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError(
            f"Inconsistent '{primary_key}' dimensions in {data_path}: {sorted(dimensions)}"
        )

    matrix = np.stack(vectors)
    if max_samples is not None and len(matrix) > max_samples:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(matrix), size=max_samples, replace=False))
        matrix = matrix[chosen]
        auxiliary = {key: [values[i] for i in chosen] for key, values in auxiliary.items()}

    auxiliary_kinds = {}
    for key, values in auxiliary.items():
        first = values[0] if values else None
        auxiliary_kinds[key] = "sequence" if isinstance(first, (list, tuple)) else "value"

    return AttrDict(
        matrix=matrix,
        auxiliary=auxiliary,
        auxiliary_kinds=auxiliary_kinds,
        dataset_tags=dataset_tags,
        skipped=skipped,
    )


def split_indexes(n_samples, validation_split_ratio, split_seed):
    if n_samples < 2:
        raise ValueError("Hyperparameter search requires at least two samples")
    n_val = int(round(validation_split_ratio * n_samples))
    n_val = min(max(n_val, 1), n_samples - 1)
    rng = np.random.default_rng(split_seed)
    permutation = rng.permutation(n_samples)
    return permutation[n_val:], permutation[:n_val]


def prepare_search_data(matrix, validation_split_ratio, split_seed, gaussian_seed):
    scaler = StandardScaler().fit(matrix)
    standardized = scaler.transform(matrix).astype(np.float32)

    trn_indexes, val_indexes = split_indexes(len(standardized), validation_split_ratio, split_seed)
    x_trn = standardized[trn_indexes]
    x_val = standardized[val_indexes]

    mu_trn = x_trn.mean(axis=0)
    sigma2_trn = x_trn.var(axis=0)
    gen_rng = np.random.default_rng(gaussian_seed)
    x_gen = gen_rng.normal(
        loc=mu_trn, scale=np.sqrt(sigma2_trn), size=(len(x_val), standardized.shape[1])
    ).astype(np.float32)

    return x_trn, x_val, x_gen, trn_indexes, val_indexes


def compute_npd(s_val, s_gen, epsilon=1e-9):
    s_val = np.asarray(s_val, dtype=np.float64)
    s_gen = np.asarray(s_gen, dtype=np.float64)
    numerator = (s_gen.mean() - s_val.mean()) ** 2
    denominator = 2.0 * (s_gen.var() + s_val.var()) + epsilon
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
#  Defense-model bridge (matrix -> synthetic graphs -> model train/score)
# ---------------------------------------------------------------------------

def _synthetic_adjacency(n_nodes, topology):
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    if topology == "complete":
        adjacency[:] = 1.0
    elif topology == "chain":
        for i in range(n_nodes - 1):
            adjacency[i, i + 1] = 1.0
            adjacency[i + 1, i] = 1.0
    elif topology == "star":
        for i in range(1, n_nodes):
            adjacency[0, i] = 1.0
            adjacency[i, 0] = 1.0
    else:
        raise ValueError(f"Unknown synthetic topology: {topology}")
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def _synthetic_auxiliary_value(kind, vector):
    if kind == "sequence":
        return [np.asarray(vector, dtype=np.float32)]
    return np.asarray(vector, dtype=np.float32)


def build_rounds(matrix, chunk_size, topology, feature_keys, auxiliary=None, auxiliary_kinds=None):
    primary_key = feature_keys[0]
    auxiliary_keys = list(feature_keys[1:])
    auxiliary_kinds = auxiliary_kinds or {}
    rounds = []
    for start in range(0, len(matrix), chunk_size):
        chunk = matrix[start:start + chunk_size]
        adjacency = _synthetic_adjacency(len(chunk), topology)
        round_data = []
        for i in range(len(chunk)):
            vector = np.asarray(chunk[i], dtype=np.float32)
            agent = {
                "agent_id": i,
                "answer": "A",
                "is_malicious": 0,
                primary_key: vector,
            }
            for key in auxiliary_keys:
                if auxiliary is not None:
                    agent[key] = auxiliary[key][start + i]
                else:
                    agent[key] = _synthetic_auxiliary_value(auxiliary_kinds.get(key, "sequence"), vector)
            round_data.append(agent)
        rounds.append((round_data, adjacency))
    return rounds


def write_synthetic_pkl(path, matrix, round_size, topology, feature_keys, auxiliary=None, auxiliary_kinds=None):
    records = []
    for round_index, (round_data, adjacency) in enumerate(
        build_rounds(matrix, round_size, topology, feature_keys, auxiliary, auxiliary_kinds)
    ):
        adjacency_list = adjacency.tolist()
        records.append(
            {
                "topology_name": f"synthetic_{round_index}",
                "topology": adjacency_list,
                "dataset_tag": "combined",
                "results": [
                    {
                        "debate_rounds": [round_data],
                        "malicious_agent_indexes": [],
                        "topology": adjacency_list,
                    }
                ],
            }
        )

    payload = {
        "data": records,
        "idx_metadata": {},
        "idx_metadata_flat": [],
        "dataset_tags": ["combined"],
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_master_class(models_directory, model_name):
    model_file = Path(models_directory) / f"{model_name}.py"
    module_name = f"hps_defense_model_{model_name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load defense model from {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    master = getattr(module, "Master", None)
    if master is None or not inspect.isclass(master):
        raise ValueError(f"Defense model '{model_file}' has no 'Master' class")
    return master


def train_model(master_class, model_cfg, pkl_path, run_name):
    temp_config = write_model_config(run_name, model_cfg)
    try:
        master = master_class(temp_config)
        _, model = master._run(pkl_path)
    finally:
        Path(temp_config).unlink(missing_ok=True)
    if not hasattr(model, "config"):
        model.config = getattr(model, "args", AttrDict(top_k=1))
    return model


def score_rounds(model, rounds):
    scores = []
    for round_data, adjacency in rounds:
        reset = getattr(model, "reset", None)
        if callable(reset):
            reset()
        result = model.predict(round_data, adjacency)
        if not isinstance(result, tuple) or len(result) < 2:
            raise ValueError("Defense model predict() must return (flags, anomaly_scores)")
        scores.append(np.asarray(result[1], dtype=np.float64).reshape(-1))
    if not scores:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(scores)


def cleanup_model(model):
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
#  Search-space helpers
# ---------------------------------------------------------------------------

def _is_search_list(value):
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(item, (int, float, str, bool)) or item is None for item in value)
    )


def split_search_space(model_cfg):
    search_space = {}
    fixed = {}
    for key, value in model_cfg.items():
        if key in _MODEL_RESERVED_KEYS:
            continue
        if _is_search_list(value):
            search_space[key] = list(value)
        else:
            fixed[key] = value
    return search_space, fixed


def search_space_size(search_space):
    if not search_space:
        return 1
    size = 1
    for values in search_space.values():
        unique = {json.dumps(value, default=str, sort_keys=True) for value in values}
        size *= max(1, len(unique))
    return size


# ---------------------------------------------------------------------------
#  Results persistence
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_results(path):
    if not Path(path).exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        log_warn(f"Could not read existing results ({path}); starting fresh.")
        return {}


def _write_results(path, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results["updated_at"] = _now()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)


def _trial_to_dict(trial):
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": dict(trial.params),
        "duration_seconds": (
            round(trial.duration.total_seconds(), 3) if trial.duration is not None else None
        ),
    }


def _sync_trials(entry, study):
    entry["trials"] = [_trial_to_dict(trial) for trial in study.trials]
    try:
        best = study.best_trial
        entry["best_trial"] = {
            "number": best.number,
            "value": best.value,
            "params": dict(best.params),
        }
    except ValueError:
        entry["best_trial"] = None


# ---------------------------------------------------------------------------
#  Per-model search
# ---------------------------------------------------------------------------

def _trial_progress(number, space_size, n_trials):
    completed = number + 1
    if space_size < n_trials:
        return (
            f"Trial {completed}/{n_trials} "
            f"(search space has only {space_size} distinct combination(s))"
        )
    return f"Trial {completed}/{n_trials}"


def run_model_search(
    model_name,
    master_class,
    fixed,
    search_space,
    n_trials,
    feature_keys,
    trn_pkl,
    full_pkl,
    val_rounds,
    gen_rounds,
    algorithm,
    optuna_cfg,
    results,
    results_path,
):
    space_size = search_space_size(search_space)
    effective_cap = min(n_trials, space_size)
    entry = {
        "model_name": model_name,
        "status": "running",
        "feature_keys": feature_keys,
        "n_trials": n_trials,
        "search_space_size": space_size,
        "effective_trial_cap": effective_cap,
        "search_space": search_space,
        "fixed_params": fixed,
        "trials": [],
        "best_trial": None,
        "final_model": None,
        "started_at": _now(),
        "duration_seconds": None,
    }
    results.setdefault("models", {})[model_name] = entry
    _write_results(results_path, results)

    if not search_space:
        log_warn(f"[{model_name}] No list-valued parameters found; all trials use the fixed config.")
    log_info(
        f"[{model_name}] feature_keys={feature_keys}; Optuna budget={n_trials} trial(s); "
        f"search space has {space_size} distinct combination(s) (effective cap {effective_cap})"
    )
    if space_size < n_trials:
        log_info(
            f"[{model_name}] budget exceeds the {space_size} distinct combination(s): "
            "TPESampler may repeat configurations and some combinations may remain unsampled."
        )

    def objective(trial):
        progress = _trial_progress(trial.number, space_size, n_trials)
        params = {
            name: trial.suggest_categorical(name, values)
            for name, values in search_space.items()
        }
        try:
            best_so_far = study.best_value
            best_txt = f"best so far NPD={best_so_far:.6f}"
        except ValueError:
            best_so_far = None
            best_txt = "best so far n/a"
        log_info(f"[{model_name}] {progress} | {best_txt} | params={params}")

        cfg = dict(fixed)
        cfg.update(params)
        trial_t0 = time()
        model = train_model(master_class, cfg, trn_pkl, f"hps_{model_name}_{trial.number}")
        try:
            s_val = score_rounds(model, val_rounds)
            s_gen = score_rounds(model, gen_rounds)
            npd = compute_npd(s_val, s_gen, algorithm.epsilon)
        finally:
            cleanup_model(model)

        outcome = "new best" if best_so_far is None or npd > best_so_far else "no improvement"
        log_info(
            f"[{model_name}] {progress} done: NPD={npd:.6f} ({outcome}) "
            f"({fmt_seconds(time() - trial_t0)})"
        )
        return npd

    def persist_callback(study, trial):
        _sync_trials(entry, study)
        _write_results(results_path, results)

    sampler = optuna.samplers.TPESampler(seed=optuna_cfg.sampler_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=model_name)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=optuna_cfg.timeout,
        callbacks=[persist_callback],
        catch=(Exception,),
    )

    _sync_trials(entry, study)
    if entry["best_trial"] is None:
        entry["status"] = "failed"
        log_error(f"[{model_name}] every trial failed; no best configuration available.")
        return entry

    best_params = entry["best_trial"]["params"]
    log_done(
        f"[{model_name}] best NPD={entry['best_trial']['value']:.6f} params={best_params}; "
        "retraining on the full dataset"
    )
    # Algorithm step 5: retrain the winning configuration on the full matrix so
    # the final model uses all available data.  No NPD is reported for this
    # retrained model: X_val was part of the final training set, so any
    # post-retrain NPD would be an in-sample value that is not comparable to
    # the trial NPDs (which are the selection metric).  The retrained model is
    # reported through its params and optional checkpoint instead.
    final_cfg = dict(fixed)
    final_cfg.update(best_params)
    final_model = train_model(master_class, final_cfg, full_pkl, f"hps_{model_name}_final")
    saved_path = None
    try:
        if algorithm.save_final_models_dir:
            save_dir = Path(algorithm.save_final_models_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_fn = getattr(final_model, "save_model", None)
            if callable(save_fn):
                target = save_dir / f"{model_name}.pt"
                save_fn(str(target))
                saved_path = str(target)
            else:
                log_warn(f"[{model_name}] model does not support save_model; skipping checkpoint.")
    finally:
        cleanup_model(final_model)

    entry["final_model"] = {
        "params": best_params,
        "saved_path": saved_path,
    }
    entry["status"] = "completed"
    log_done(
        f"[{model_name}] final model retrained on the full dataset"
        + (f" and saved to {saved_path}" if saved_path else " (no checkpoint requested)")
    )
    return entry


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def run_hps(config_path, clean=False):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    config = load_hps_config(config_path)
    algorithm = config.algorithm
    optuna_cfg = config.optuna

    log_section("Hyperparameter Search (AutoUAD / NPD)")
    log_config("config_file", str(config_path))
    log_config("data_path", config.data_path)
    log_config("models_directory", config.models_directory)
    log_config("output_file", config.output_file)
    log_config("validation_split_ratio", algorithm.validation_split_ratio)
    log_config("epsilon", algorithm.epsilon)
    log_config("split_seed", algorithm.split_seed)
    log_config("n_trials", optuna_cfg.n_trials)

    output_path = Path(config.output_file)
    if clean and output_path.exists():
        output_path.unlink()
        log_info(f"--clean: removed {output_path}")

    overall_t0 = time()
    log_section("Loading default feature data")
    default_keys = [algorithm.feature_key]
    default_data = load_feature_data(
        config.data_path,
        default_keys,
        algorithm.max_samples,
        algorithm.split_seed,
    )
    matrix = default_data.matrix
    log_info(
        f"Loaded {len(matrix)} '{algorithm.feature_key}' sample(s) of dimension "
        f"{matrix.shape[1]} from {len(default_data.dataset_tags) or 1} dataset(s): "
        f"{default_data.dataset_tags or ['combined']}"
    )
    if default_data.skipped:
        log_warn(f"Skipped {default_data.skipped} sample(s) without a usable '{algorithm.feature_key}'.")

    results = _read_results(output_path)
    if not results:
        trn_indexes, val_indexes = split_indexes(
            len(matrix), algorithm.validation_split_ratio, algorithm.split_seed
        )
        results = {
            "script": "HyperParameterSearch.py",
            "algorithm": "AutoUAD (NPD-guided)",
            "config_file": str(config_path),
            "created_at": _now(),
            "updated_at": _now(),
            "data": {
                "data_path": str(config.data_path),
                "dataset_tags": default_data.dataset_tags,
                "feature_key": algorithm.feature_key,
                "n_samples": int(len(matrix)),
                "n_train": int(len(trn_indexes)),
                "n_val": int(len(val_indexes)),
                "feature_dim": int(matrix.shape[1]),
                "max_samples": algorithm.max_samples,
                "n_skipped_samples": int(default_data.skipped),
            },
            "algorithm_params": {
                "validation_split_ratio": algorithm.validation_split_ratio,
                "epsilon": algorithm.epsilon,
                "split_seed": algorithm.split_seed,
                "gaussian_seed": algorithm.gaussian_seed,
                "round_size": algorithm.round_size,
                "score_chunk_size": algorithm.score_chunk_size,
                "synthetic_topology": algorithm.synthetic_topology,
            },
            "optuna_params": {
                "n_trials": optuna_cfg.n_trials,
                "sampler_seed": optuna_cfg.sampler_seed,
                "timeout": optuna_cfg.timeout,
            },
            "models": {},
        }

    temp_dir = Path(tempfile.mkdtemp(prefix="hps-data-"))
    data_cache = {tuple(default_keys): default_data}

    completed_before = {
        name for name, entry in results.get("models", {}).items()
        if entry.get("status") == "completed"
    }

    def _prepare_model_datasets(model_name, model_cfg):
        feature_keys = list(model_cfg["feature_keys"])
        cache_key = tuple(feature_keys)
        data = data_cache.get(cache_key)
        if data is None:
            data = load_feature_data(
                config.data_path,
                feature_keys,
                algorithm.max_samples,
                algorithm.split_seed,
            )
            data_cache[cache_key] = data
            log_info(
                f"Loaded {len(data.matrix)} sample(s) for feature_keys={feature_keys} "
                f"(dimension {data.matrix.shape[1]}); skipped {data.skipped}."
            )
        x_trn, x_val, x_gen, trn_indexes, val_indexes = prepare_search_data(
            data.matrix,
            algorithm.validation_split_ratio,
            algorithm.split_seed,
            algorithm.gaussian_seed,
        )
        auxiliary = data.auxiliary
        auxiliary_kinds = data.auxiliary_kinds
        auxiliary_trn = {
            key: [values[i] for i in trn_indexes] for key, values in auxiliary.items()
        }
        auxiliary_val = {
            key: [values[i] for i in val_indexes] for key, values in auxiliary.items()
        }
        trn_pkl = write_synthetic_pkl(
            temp_dir / f"{model_name}-train.pkl",
            x_trn,
            algorithm.round_size,
            algorithm.synthetic_topology,
            feature_keys,
            auxiliary_trn,
            auxiliary_kinds,
        )
        full_pkl = write_synthetic_pkl(
            temp_dir / f"{model_name}-full.pkl",
            data.matrix,
            algorithm.round_size,
            algorithm.synthetic_topology,
            feature_keys,
            auxiliary,
            auxiliary_kinds,
        )
        val_rounds = build_rounds(
            x_val,
            algorithm.score_chunk_size,
            algorithm.synthetic_topology,
            feature_keys,
            auxiliary_val,
            auxiliary_kinds,
        )
        gen_rounds = build_rounds(
            x_gen,
            algorithm.score_chunk_size,
            algorithm.synthetic_topology,
            feature_keys,
            None,
            auxiliary_kinds,
        )
        log_info(
            f"Synthetic datasets ready for {model_name}: {len(x_trn)} train / "
            f"{len(data.matrix)} full sample(s); validation and Gaussian sets chunked "
            f"into {len(val_rounds)} graph(s) of up to {algorithm.score_chunk_size} node(s)."
        )
        return trn_pkl, full_pkl, val_rounds, gen_rounds

    try:
        for model_name, model_cfg in config.models.items():
            if model_name in completed_before:
                log_section(f"Model: {model_name} -- SKIPPED")
                log_info(f"'{model_name}' already completed in {output_path}; keeping its results.")
                continue

            log_section(f"Model: {model_name}")
            feature_keys = list(model_cfg["feature_keys"])
            global_n_trials = optuna_cfg.n_trials
            model_n_trials = model_cfg.get("n_trials")
            n_trials = model_n_trials if model_n_trials is not None else global_n_trials
            search_space, fixed = split_search_space(dict(model_cfg))
            log_config("feature_keys", feature_keys)
            log_config(
                "n_trials",
                f"{n_trials}" + (
                    f" (per-model override; global {global_n_trials})"
                    if model_n_trials is not None else ""
                ),
            )
            log_info(
                f"Search space: {search_space if search_space else '(none, fixed config)'}"
            )
            log_config("fixed_params", json.dumps(fixed, default=str, sort_keys=True))

            model_t0 = time()
            try:
                trn_pkl, full_pkl, val_rounds, gen_rounds = _prepare_model_datasets(
                    model_name, model_cfg
                )
                master_class = load_master_class(config.models_directory, model_name)
                entry = run_model_search(
                    model_name,
                    master_class,
                    fixed,
                    search_space,
                    n_trials,
                    feature_keys,
                    str(trn_pkl),
                    str(full_pkl),
                    val_rounds,
                    gen_rounds,
                    algorithm,
                    optuna_cfg,
                    results,
                    output_path,
                )
                entry["duration_seconds"] = round(time() - model_t0, 3)
                log_info(f"Model '{model_name}' finished in {fmt_seconds(time() - model_t0)}")
            except KeyboardInterrupt:
                log_warn("KeyboardInterrupt received. Completed results are preserved.")
                raise
            except Exception as exc:
                entry = results.get("models", {}).get(model_name, {})
                entry["status"] = "failed"
                entry["error"] = str(exc)
                entry["duration_seconds"] = round(time() - model_t0, 3)
                results.setdefault("models", {})[model_name] = entry
                log_error(f"Model '{model_name}' failed: {exc}")
                for line in traceback.format_exc().strip().splitlines():
                    log_error(line)
            finally:
                _write_results(output_path, results)
    except KeyboardInterrupt:
        log_warn("KeyboardInterrupt received. All completed results have been saved.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        _write_results(output_path, results)
        log_section("Hyperparameter Search Finished")
        log_info(f"Total elapsed: {fmt_seconds(time() - overall_t0)}")
        log_info(f"Results JSON: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised AutoUAD/NPD hyperparameter search.")
    parser.add_argument("config_file", type=str, help="Path to the HPS configuration file.")
    parser.add_argument("--clean", action="store_true", help="Delete existing results and start fresh.")
    parsed_args = parser.parse_args()
    run_hps(parsed_args.config_file, clean=parsed_args.clean)
