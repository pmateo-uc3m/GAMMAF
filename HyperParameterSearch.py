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
one).  The number of executed trials is capped by the number of distinct
search-space combinations (``effective_cap = min(n_trials, space_size)``) so
a budget larger than the space cannot spend trials on duplicated
configurations.  Early stopping can end a model's search before its budget once
the best NPD stops improving by at least a configured percentage for a
configured number of consecutive trials (``optuna.early_stopping_patience`` and
``optuna.early_stopping_min_improvement_pct``; both global, each model's
improvement history is tracked independently).  After the search the best
configuration is retrained on the full feature matrix; that retrained model
(params + optional checkpoint) is reported in the results JSON.  Trial NPDs are
the quality metrics: no NPD is reported for the retrained model because X_val is
part of its training data, which would make that value in-sample and not
comparable to the trial NPDs.

Config schema (see ``config-examples/hyperparameter-search-config.yaml``)::

    data_path: paper-experiments/exp1/exp1-train-data.pkl
    models_directory: defense-models
    output_file: hps/hps-results.json

    algorithm:
      validation_split_ratio: 0.3
      epsilon: 1.0e-9
      split_seed: 42
      gaussian_seed: 42
      max_debates: null               # optional cap on the number of debates
      feature_key: st_embedding       # default primary key (see feature_keys)
      save_final_models_dir: null

    optuna:
      n_trials: 50
      sampler_seed: 42
      timeout: null
      early_stopping_patience: null          # consecutive non-improving trials -> stop (null = off)
      early_stopping_min_improvement_pct: 0.0  # % NPD gain over the best that counts as progress

    training:                        # global model-training defaults (per-model override below)
      n_epochs_lr_reduce: 5          # epochs without >= threshold % improvement -> LR reduction
      lr_reduce_improvement_pct: 1.0
      lr_reduce_factor: 0.5
      min_lr: 1.0e-6
      n_epochs_early_stop: 12        # epochs without >= threshold % improvement -> stop training
      early_stop_improvement_pct: 1.0

    models:                          # one section per defense model file stem
      BlindGuard:
        seed: 42                     # run seed (sampler + per-trial training seeds)
        hidden_dim: [64, 128, 256]   # list value -> searched
        learning_rate: [0.001]       # list value -> searched
        input_dim: 1152              # scalar -> held constant
        ...
      XG-Guard:                      # multi-key model
        feature_keys: [st_embedding, tk_embedding]
        n_trials: 10                 # optional per-model Optuna budget override
        split_seed: 542              # optional per-model debate-split seed
        gaussian_seed: 1542          # optional per-model Gaussian-proxy seed
        ...
    # Any `training` key may also be set inside a model section to override the
    # global value for that model only (scalar = fixed, list = searched).

Resume
------
Rerunning with the same ``output_file`` resumes the sweep: every configured
model whose entry is recorded as cleanly completed (``status: completed`` with
its winning trial and retrained final model) is skipped, and only the remaining
models run.  Entries left ``running`` by an interrupted search, or ``failed``,
are rerun from scratch; results are appended to the same JSON.

Per-model seeds
---------------
One seed per model (``seed``) controls the whole model optimization run: it
seeds Optuna's sampler, the final retrain, and every trial's training through
``trial_seed = seed + 1 + trial_number``.  Each Optuna step therefore gets its
own randomization, so a parameter combination sampled twice does not produce a
bit-identical model/score; the whole run stays reproducible from the single
model seed and the per-trial seed is logged and recorded in the results JSON.
``split_seed``/``gaussian_seed`` optionally override the global
``algorithm.split_seed``/``algorithm.gaussian_seed`` for one model, so
different models' searches use different train/validation partitions and
different Gaussian proxies (the split stays fixed across every trial of one
model, as the algorithm requires).  Both effective seeds are logged and
recorded in the results JSON; a per-model ``gaussian_seed`` defaults to the
per-model ``split_seed``.

Real graphs
-----------
Training entries are used as the actual graphs they are: every debate in the
pkl is a graph with its own adjacency (``topology``), agent order and already
specific topology (``tree``, ``chain``, ``star``, ``random`` ...).  The
primary feature vectors of all agents/rounds form the tabular matrix ``X``,
which is standardized globally (one scaler over all agents, keeping each
feature zero-mean/unit-variance across the dataset so the Gaussian proxy
stays valid; per-graph standardization would center each graph locally,
destroy the global zero-mean property and suppress the within-graph
deviations the detectors score).  Debates are split 70/30 **at debate level**,
stratified by ``(topology_name, dataset_tag)`` so all topologies and datasets
appear in both splits and no round of a debate is shared between train and
validation.  ``X_val`` scores come from the held-out debates; ``X_gen`` is
sampled from the ``X_trn`` per-feature mean/variance exactly as before and is
partitioned 1:1 onto the validation graph structures (same size and adjacency
per validation round) so structure is not a confound.

Per-model feature keys
----------------------
The first key in ``feature_keys`` is the *primary* key: its fixed-length
vectors form the tabular matrix used by the AutoUAD/NPD algorithm
(standardization, split, Gaussian proxy, NPD). Every key in the list is
attached to the per-agent dicts handed to the model, exactly as the live
evaluation pipeline does. ``feature_key: <key>`` is a shorthand for a
single-entry list; if neither is given, the global
``algorithm.feature_key`` is used. Auxiliary keys are synthesized from the
generated primary vectors for the Gaussian graphs (only the primary key
exists in that synthetic set); they are preserved verbatim everywhere else.

Every record written for model training is one debate with all its rounds and
its real topology, with ``malicious_agent_indexes`` emptied and
``is_malicious`` zeroed so the search stays label-free.
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

from EvaluationConfigCheck import (
    merge_training_defaults,
    validate_training_config,
    write_model_config,
)
from LoggingUtils import (
    fmt_seconds,
    log_config,
    log_done,
    log_error,
    log_info,
    log_section,
    log_subsection,
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
    "training",
    "optuna",
    "models",
}
_ROOT_REQUIRED = {"data_path", "output_file", "models"}

_ALGORITHM_KEYS = {
    "validation_split_ratio",
    "epsilon",
    "split_seed",
    "gaussian_seed",
    "max_debates",
    "feature_key",
    "save_final_models_dir",
}

_OPTUNA_KEYS = {
    "n_trials",
    "sampler_seed",
    "timeout",
    "early_stopping_patience",
    "early_stopping_min_improvement_pct",
}

_MODEL_RESERVED_KEYS = {"feature_key", "feature_keys", "n_trials", "split_seed", "gaussian_seed"}


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
    max_debates = raw.get("max_debates")
    if max_debates is not None:
        max_debates = _require_int(max_debates, "algorithm.max_debates", minimum=2)
    save_dir = raw.get("save_final_models_dir")
    if save_dir is not None:
        _require_str(save_dir, "algorithm.save_final_models_dir")
    return AttrDict(
        validation_split_ratio=ratio,
        epsilon=epsilon,
        split_seed=split_seed,
        gaussian_seed=gaussian_seed,
        max_debates=max_debates,
        feature_key=_require_str(raw.get("feature_key", "st_embedding"), "algorithm.feature_key"),
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
    early_stopping_patience = raw.get("early_stopping_patience")
    if early_stopping_patience is not None:
        early_stopping_patience = _require_int(
            early_stopping_patience, "optuna.early_stopping_patience", minimum=1
        )
    early_stopping_min_improvement_pct = _require_number(
        raw.get("early_stopping_min_improvement_pct", 0.0),
        "optuna.early_stopping_min_improvement_pct",
        minimum=0.0,
    )
    return AttrDict(
        n_trials=n_trials,
        sampler_seed=sampler_seed,
        timeout=timeout,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_improvement_pct=early_stopping_min_improvement_pct,
    )


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


def _validate_models(raw, models_directory, default_feature_key, training=None):
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

        # Global training defaults first; a per-model value (scalar or search
        # list) overrides the global one for this model only.
        entry = merge_training_defaults(model_cfg, training)
        feature_keys = _resolve_feature_keys(entry, f"models.{model_name}", default_feature_key)
        per_model_trials = entry.get("n_trials")
        if per_model_trials is not None:
            per_model_trials = _require_int(
                per_model_trials, f"models.{model_name}.n_trials", minimum=1
            )
        per_model_split_seed = entry.get("split_seed")
        if per_model_split_seed is not None:
            per_model_split_seed = _require_int(
                per_model_split_seed, f"models.{model_name}.split_seed"
            )
        per_model_gaussian_seed = entry.get("gaussian_seed")
        if per_model_gaussian_seed is not None:
            per_model_gaussian_seed = _require_int(
                per_model_gaussian_seed, f"models.{model_name}.gaussian_seed"
            )
        for key in _MODEL_RESERVED_KEYS:
            entry.pop(key, None)
        if "seed" not in entry:
            entry["seed"] = 42

        normalized = AttrDict(entry)
        normalized["feature_keys"] = feature_keys
        normalized["n_trials"] = per_model_trials
        normalized["split_seed"] = per_model_split_seed
        normalized["gaussian_seed"] = per_model_gaussian_seed
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
    training = validate_training_config(raw.get("training"))
    return AttrDict(
        data_path=_require_path(raw["data_path"], "data_path"),
        models_directory=models_directory,
        output_file=_require_str(raw["output_file"], "output_file"),
        algorithm=algorithm,
        training=training,
        optuna=_validate_optuna(_require_mapping(raw, "optuna", "root")),
        models=_validate_models(raw["models"], models_directory, algorithm.feature_key, training),
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


def _resolve_debate_adjacency(debate, record):
    adjacency = debate.get("topology")
    if adjacency is None:
        adjacency = record.get("topology")
    if adjacency is None:
        return None
    try:
        array = np.asarray(adjacency, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[0] != array.shape[1]:
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array


def _compact_auxiliary(value):
    """Return a compact numeric form of an auxiliary payload when possible.

    The generation pkls store token embeddings as Python float lists, which
    cost many times the memory of float32 arrays once unpickled; the defense
    models only ever consume float32 tensors, so compacting here keeps the
    loaded graphs (and the temporary training pkls written from them) small
    without changing the values the models see.  Non-numeric payloads are kept
    verbatim.
    """
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    try:
        return np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return value


def load_graph_data(data_path, feature_keys):
    """Load the real debates of ``data_path`` as graphs.

    Each graph keeps the debate adjacency, its topology name, dataset tag and
    round/agent order.  Every agent payload carries the requested feature keys
    plus the standard ``agent_id``/``answer``/``is_malicious`` fields.
    ``feature_keys[0]`` is the primary key whose vectors form the tabular
    matrix used by the AutoUAD/NPD algorithm; the remaining keys ride along.
    Auxiliary payloads are compacted to float32 arrays where possible so a
    multi-key load does not retain the raw Python float lists.
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

    graphs = []
    auxiliary_kinds = {}
    skipped_debates = 0
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        record_topology_name = record.get("topology_name")
        record_dataset_tag = record.get("dataset_tag")
        for debate_index, debate in enumerate(record.get("results", []) or []):
            if not isinstance(debate, dict):
                continue
            adjacency = _resolve_debate_adjacency(debate, record)
            debate_rounds = debate.get("debate_rounds", []) or []
            if adjacency is None or not debate_rounds:
                skipped_debates += 1
                continue

            graph_rounds = []
            valid = True
            for debate_round in debate_rounds:
                if not isinstance(debate_round, list) or len(debate_round) != adjacency.shape[0]:
                    valid = False
                    break
                round_agents = []
                for agent_position, agent in enumerate(debate_round):
                    if not isinstance(agent, dict):
                        valid = False
                        break
                    vector = _as_feature_vector(agent.get(primary_key))
                    if vector is None:
                        valid = False
                        break
                    payloads = {}
                    for key in auxiliary_keys:
                        value = agent.get(key)
                        if value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
                            valid = False
                            break
                        payloads[key] = value
                    if not valid:
                        break

                    agent_payload = AttrDict(
                        agent_id=agent.get("agent_id", agent_position),
                        answer=agent.get("answer", "A"),
                        is_malicious=0,
                    )
                    agent_payload[primary_key] = vector
                    for key, value in payloads.items():
                        kind = "sequence" if isinstance(value, (list, tuple)) else "value"
                        compacted = _compact_auxiliary(value)
                        agent_payload[key] = compacted
                        if compacted is not value:
                            # Drop the raw Python float list from the source
                            # payload immediately so the multi-key peak stays at
                            # the unpickle size instead of payload + compact copy.
                            agent[key] = compacted
                        if key not in auxiliary_kinds:
                            auxiliary_kinds[key] = kind
                    round_agents.append(agent_payload)
                if not valid:
                    break
                graph_rounds.append(round_agents)

            if not valid:
                skipped_debates += 1
                continue

            topology_name = debate.get("topology_name", record_topology_name)
            if not topology_name:
                topology_name = f"debate_{record_index}_{debate_index}"
            graphs.append(
                AttrDict(
                    debate_id=debate.get(
                        "debate_id", f"{topology_name}_{record_index}_{debate_index}"
                    ),
                    topology_name=str(topology_name),
                    dataset_tag=str(debate.get("dataset_tag", record_dataset_tag) or "combined"),
                    adjacency=adjacency,
                    rounds=graph_rounds,
                )
            )

    if not graphs:
        raise ValueError(f"No usable graphs found in {data_path}")

    return AttrDict(
        graphs=graphs,
        dataset_tags=dataset_tags,
        auxiliary_kinds=auxiliary_kinds,
        skipped_debates=skipped_debates,
    )


def select_graphs(loaded, topologies, max_debates, seed):
    graphs = loaded.graphs
    if topologies:
        allowed = {topologies} if isinstance(topologies, str) else set(topologies)
        graphs = [graph for graph in graphs if graph.topology_name in allowed]
        if not graphs:
            available = sorted({graph.topology_name for graph in loaded.graphs})
            raise ValueError(
                f"No graphs match the configured topologies {sorted(allowed)}; "
                f"available topologies: {available}"
            )
    if max_debates is not None and len(graphs) > max_debates:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(graphs), size=max_debates, replace=False))
        graphs = [graphs[i] for i in chosen]
    return graphs


def graph_statistics(graphs):
    topology_counts = {}
    dataset_counts = {}
    n_rounds = 0
    n_agents = 0
    for graph in graphs:
        topology_counts[graph.topology_name] = topology_counts.get(graph.topology_name, 0) + 1
        dataset_counts[graph.dataset_tag] = dataset_counts.get(graph.dataset_tag, 0) + 1
        n_rounds += len(graph.rounds)
        n_agents += sum(len(round_agents) for round_agents in graph.rounds)
    return AttrDict(
        n_debates=len(graphs),
        n_rounds=n_rounds,
        n_agents=n_agents,
        topology_counts=topology_counts,
        dataset_counts=dataset_counts,
    )


def standardize_graphs(graphs, primary_key):
    """Global feature standardization over all agents of all graphs.

    One scaler is fit on the pooled agent vectors so every feature is
    zero-mean/unit-variance across the dataset; the Gaussian proxy sampled
    later from the training statistics relies on that global centering.
    Per-graph standardization would center each graph locally, destroy the
    global zero-mean property and suppress the within-graph deviations the
    detectors score, so it is intentionally not used.
    """
    if not graphs:
        raise ValueError("Hyperparameter search requires at least one graph")
    matrix = np.stack(
        [agent[primary_key] for graph in graphs for round_agents in graph.rounds for agent in round_agents]
    )
    scaler = StandardScaler().fit(matrix)
    standardized = scaler.transform(matrix).astype(np.float32)

    position = 0
    out = []
    for graph in graphs:
        new_rounds = []
        for round_agents in graph.rounds:
            new_agents = []
            for agent in round_agents:
                new_agent = dict(agent)
                new_agent[primary_key] = standardized[position]
                position += 1
                new_agents.append(new_agent)
            new_rounds.append(new_agents)
        new_graph = AttrDict(dict(graph))
        new_graph["rounds"] = new_rounds
        out.append(new_graph)
    return out


def split_graphs(graphs, validation_split_ratio, split_seed):
    """Debate-level split stratified by (topology_name, dataset_tag)."""
    strata = {}
    for index, graph in enumerate(graphs):
        strata.setdefault((graph.topology_name, graph.dataset_tag), []).append(index)

    rng = np.random.default_rng(split_seed)
    train_indexes = []
    val_indexes = []
    for key in sorted(strata):
        indexes = list(strata[key])
        shuffled = [indexes[i] for i in rng.permutation(len(indexes))]
        if len(shuffled) < 2:
            train_indexes.extend(shuffled)
            continue
        n_val = int(round(validation_split_ratio * len(shuffled)))
        n_val = min(max(n_val, 1), len(shuffled) - 1)
        val_indexes.extend(shuffled[:n_val])
        train_indexes.extend(shuffled[n_val:])
    return [graphs[i] for i in train_indexes], [graphs[i] for i in val_indexes]


def prepare_graph_search_data(graphs, feature_keys, validation_split_ratio, split_seed, gaussian_seed):
    """Standardize globally, split debates, and build the Gaussian proxy.

    ``X_val`` is the pooled held-out debate agents; ``X_gen`` is sampled from
    the ``X_trn`` per-feature mean/variance (unchanged AutoUAD step) and is
    later partitioned 1:1 over the validation graph structures.
    """
    primary_key = feature_keys[0]
    standardized = standardize_graphs(graphs, primary_key)
    trn_graphs, val_graphs = split_graphs(standardized, validation_split_ratio, split_seed)
    if not trn_graphs or not val_graphs:
        raise ValueError("The stratified debate split produced an empty train or validation set")

    trn_vectors = np.stack(
        [agent[primary_key] for graph in trn_graphs for round_agents in graph.rounds for agent in round_agents]
    )
    mu_trn = trn_vectors.mean(axis=0)
    sigma2_trn = trn_vectors.var(axis=0)
    n_val_agents = sum(len(round_agents) for graph in val_graphs for round_agents in graph.rounds)

    gen_rng = np.random.default_rng(gaussian_seed)
    x_gen = gen_rng.normal(
        loc=mu_trn, scale=np.sqrt(sigma2_trn), size=(n_val_agents, trn_vectors.shape[1])
    ).astype(np.float32)
    return standardized, trn_graphs, val_graphs, x_gen


def compute_npd(s_val, s_gen, epsilon=1e-9):
    s_val = np.asarray(s_val, dtype=np.float64)
    s_gen = np.asarray(s_gen, dtype=np.float64)
    numerator = (s_gen.mean() - s_val.mean()) ** 2
    denominator = 2.0 * (s_gen.var() + s_val.var()) + epsilon
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
#  Defense-model bridge (graphs -> model train/score)
# ---------------------------------------------------------------------------

def _synthetic_auxiliary_value(kind, vector):
    if kind == "sequence":
        return [np.asarray(vector, dtype=np.float32)]
    return np.asarray(vector, dtype=np.float32)


def build_real_rounds(graphs):
    """Group real graphs into per-debate ``(round_data, adjacency)`` sequences.

    Keeping the rounds inside their debate lets stateful models (e.g. CASPIAN)
    see the debate's turn sequence, exactly as the live evaluation loop does,
    while the concatenated per-agent scores keep the debate-major order of the
    previous flat representation.
    """
    debates = []
    for graph in graphs:
        rounds = [
            ([dict(agent) for agent in round_agents], graph.adjacency)
            for round_agents in graph.rounds
        ]
        debates.append(AttrDict(debate_id=graph.debate_id, rounds=rounds))
    return debates


def build_gen_rounds(val_graphs, x_gen, feature_keys, auxiliary_kinds=None):
    """Partition the Gaussian rows 1:1 over the validation graph structures."""
    primary_key = feature_keys[0]
    auxiliary_keys = list(feature_keys[1:])
    auxiliary_kinds = auxiliary_kinds or {}
    debates = []
    position = 0
    for graph in val_graphs:
        rounds = []
        for round_agents in graph.rounds:
            n_agents = len(round_agents)
            round_data = []
            for i in range(n_agents):
                vector = np.asarray(x_gen[position + i], dtype=np.float32)
                agent = {
                    "agent_id": i,
                    "answer": "A",
                    "is_malicious": 0,
                    primary_key: vector,
                }
                for key in auxiliary_keys:
                    agent[key] = _synthetic_auxiliary_value(
                        auxiliary_kinds.get(key, "sequence"), vector
                    )
                round_data.append(agent)
            rounds.append((round_data, graph.adjacency))
            position += n_agents
        debates.append(AttrDict(debate_id=f"gaussian::{graph.debate_id}", rounds=rounds))
    if position != len(x_gen):
        raise ValueError(
            f"Gaussian partition mismatch: consumed {position} rows for {len(x_gen)} samples"
        )
    return debates


def write_debate_pkl(path, graphs, feature_keys):
    """Write one record per real debate (all rounds, real topology, no labels)."""
    primary_key = feature_keys[0]
    auxiliary_keys = list(feature_keys[1:])
    records = []
    for graph in graphs:
        adjacency_list = np.asarray(graph.adjacency).tolist()
        rounds_payload = []
        for round_agents in graph.rounds:
            round_payload = []
            for agent in round_agents:
                entry = {
                    "agent_id": agent.get("agent_id"),
                    "answer": agent.get("answer", "A"),
                    "is_malicious": 0,
                    primary_key: agent[primary_key],
                }
                for key in auxiliary_keys:
                    entry[key] = agent[key]
                round_payload.append(entry)
            rounds_payload.append(round_payload)
        records.append(
            {
                "topology_name": graph.topology_name,
                "topology": adjacency_list,
                "dataset_tag": graph.dataset_tag,
                "debate_id": graph.debate_id,
                "results": [
                    {
                        "debate_rounds": rounds_payload,
                        "malicious_agent_indexes": [],
                        "topology": adjacency_list,
                        "dataset_tag": graph.dataset_tag,
                    }
                ],
            }
        )

    payload = {
        "data": records,
        "idx_metadata": {},
        "idx_metadata_flat": [],
        "dataset_tags": sorted({graph.dataset_tag for graph in graphs}),
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


def _predict_supports_trace_id(model):
    try:
        parameters = inspect.signature(model.predict).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "trace_id"
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def score_rounds(model, debates):
    """Score each debate as a unit and return the concatenated agent scores.

    Every round is still scored with its own adjacency (the per-graph structure
    is preserved), but a stateful model sees all rounds of a debate in order
    before its state is released, mirroring the live evaluation loop's
    ``begin_trace``/``end_trace`` handling.  Stateless models are unaffected.
    """
    supports_trace = _predict_supports_trace_id(model)
    begin_trace = getattr(model, "begin_trace", None)
    end_trace = getattr(model, "end_trace", None)
    reset = getattr(model, "reset", None)
    scores = []
    for debate in debates:
        rounds = debate.rounds
        if not rounds:
            continue
        trace_id = debate.debate_id if supports_trace else None
        if trace_id is not None and callable(begin_trace):
            begin_trace(trace_id, rounds[0][1])
        elif callable(reset):
            reset()
        try:
            for round_data, adjacency in rounds:
                if trace_id is not None:
                    result = model.predict(round_data, adjacency, trace_id=trace_id)
                else:
                    result = model.predict(round_data, adjacency)
                if not isinstance(result, tuple) or len(result) < 2:
                    raise ValueError("Defense model predict() must return (flags, anomaly_scores)")
                scores.append(np.asarray(result[1], dtype=np.float64).reshape(-1))
        finally:
            if trace_id is not None and callable(end_trace):
                end_trace(trace_id)
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


def relative_improvement_pct(value, best_value):
    """Percent improvement of ``value`` over ``best_value``.

    Returns ``None`` when there is no baseline yet.  NPD is non-negative, so a
    zero baseline is treated as no improvement unless the new value is strictly
    positive.
    """
    if best_value is None:
        return None
    if best_value <= 0.0:
        return float("inf") if value > best_value else 0.0
    return (value - best_value) / best_value * 100.0


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
    """Atomically persist ``results`` so an interrupted write cannot corrupt it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results["updated_at"] = _now()
    payload = json.dumps(results, indent=2, default=str)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(payload, encoding="utf-8")
    temp_path.replace(path)


def _is_completed_model_entry(entry):
    """True only for a model entry that finished cleanly.

    Interrupted searches leave ``status`` at ``"running"`` and failures at
    ``"failed"``; both are rerun from scratch.  A completed entry always carries
    its winning trial and the retrained final model.
    """
    if not isinstance(entry, dict) or entry.get("status") != "completed":
        return False
    if not isinstance(entry.get("final_model"), dict):
        return False
    return entry.get("best_trial") is not None


def _trial_to_dict(trial, trial_seeds=None):
    trial_seeds = trial_seeds or {}
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": dict(trial.params),
        "seed": trial_seeds.get(trial.number),
        "duration_seconds": (
            round(trial.duration.total_seconds(), 3) if trial.duration is not None else None
        ),
    }


def _sync_trials(entry, study, trial_seeds=None):
    entry["trials"] = [_trial_to_dict(trial, trial_seeds) for trial in study.trials]
    try:
        best = study.best_trial
        entry["best_trial"] = {
            "number": best.number,
            "value": best.value,
            "params": dict(best.params),
            "seed": (trial_seeds or {}).get(best.number),
        }
    except ValueError:
        entry["best_trial"] = None


def trial_seed_for(run_seed, trial_number):
    """Deterministic per-trial training seed derived from one run seed.

    The run seed alone controls the whole model optimization run: Optuna's
    sampler seed, every trial's model-initialisation/shuffling/split RNGs and
    the final retrain.  Deriving ``run_seed + 1 + trial_number`` gives each
    Optuna step its own randomization while keeping the whole run reproducible
    from the single seed (repeated parameter combinations therefore do not
    produce bit-identical models/scores).
    """
    return int(run_seed) + 1 + int(trial_number)


# ---------------------------------------------------------------------------
#  Per-model search
# ---------------------------------------------------------------------------

def _trial_progress(number, executed_trials):
    return f"Trial {number + 1}/{executed_trials}"


def run_model_search(
    model_name,
    master_class,
    fixed,
    search_space,
    n_trials,
    feature_keys,
    trn_pkl,
    full_pkl,
    val_debates,
    gen_debates,
    algorithm,
    optuna_cfg,
    results,
    results_path,
    run_seed,
    split_seed,
    gaussian_seed,
):
    space_size = search_space_size(search_space)
    effective_cap = min(n_trials, space_size)
    patience = optuna_cfg.early_stopping_patience
    min_improvement_pct = optuna_cfg.early_stopping_min_improvement_pct
    trial_seeds = {}
    early_stopping_state = {
        "best_value": None,
        "stalled_trials": 0,
        "stopped": False,
    }
    entry = {
        "model_name": model_name,
        "status": "running",
        "feature_keys": feature_keys,
        "n_trials": n_trials,
        "effective_trial_cap": effective_cap,
        "executed_trials": 0,
        "search_space_size": space_size,
        "run_seed": run_seed,
        "split_seed": split_seed,
        "gaussian_seed": gaussian_seed,
        "search_space": search_space,
        "fixed_params": fixed,
        "early_stopping": {
            "enabled": patience is not None,
            "patience": patience,
            "min_improvement_pct": min_improvement_pct,
            "stopped_early": False,
            "stalled_trials": 0,
            "best_value": None,
        },
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
        f"search space has {space_size} distinct combination(s); running "
        f"{effective_cap} trial(s) (effective cap = min(budget, space size))"
    )
    log_info(
        f"[{model_name}] run_seed={run_seed} controls the Optuna sampler and every trial's "
        f"training seed (run_seed+1+trial_number); split_seed={split_seed} and "
        f"gaussian_seed={gaussian_seed} stay fixed across trials"
    )
    if patience is not None:
        log_info(
            f"[{model_name}] early stopping enabled: stop after {patience} consecutive "
            f"trial(s) without at least {min_improvement_pct}% NPD improvement over the best"
        )

    def objective(trial):
        progress = _trial_progress(trial.number, effective_cap)
        params = {
            name: trial.suggest_categorical(name, values)
            for name, values in search_space.items()
        }
        trial_seed = trial_seed_for(run_seed, trial.number)
        trial_seeds[trial.number] = trial_seed
        try:
            best_so_far = study.best_value
            best_txt = f"best so far NPD={best_so_far:.6f}"
        except ValueError:
            best_so_far = None
            best_txt = "best so far n/a"
        log_info(
            f"[{model_name}] {progress} | {best_txt} | seed={trial_seed} | params={params}"
        )

        cfg = dict(fixed)
        cfg.update(params)
        cfg["seed"] = trial_seed
        trial_t0 = time()
        model = train_model(master_class, cfg, trn_pkl, f"hps_{model_name}_{trial.number}")
        try:
            s_val = score_rounds(model, val_debates)
            s_gen = score_rounds(model, gen_debates)
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
        _sync_trials(entry, study, trial_seeds)
        _write_results(results_path, results)

    def early_stopping_callback(study, trial):
        if patience is None:
            return
        value = trial.value
        best_value = early_stopping_state["best_value"]
        if value is None:
            early_stopping_state["stalled_trials"] += 1
        elif best_value is None:
            early_stopping_state["best_value"] = value
            early_stopping_state["stalled_trials"] = 0
        elif (
            value > best_value
            and relative_improvement_pct(value, best_value) >= min_improvement_pct
        ):
            early_stopping_state["best_value"] = value
            early_stopping_state["stalled_trials"] = 0
        else:
            early_stopping_state["stalled_trials"] += 1
        if early_stopping_state["stalled_trials"] < patience:
            return
        early_stopping_state["stopped"] = True
        study.stop()
        best_txt = (
            f"{early_stopping_state['best_value']:.6f}"
            if early_stopping_state["best_value"] is not None
            else "n/a"
        )
        log_info(
            f"[{model_name}] early stopping triggered after trial {trial.number + 1}: "
            f"{patience} consecutive trial(s) without at least {min_improvement_pct}% "
            f"NPD improvement over the best ({best_txt}); stopping the search early"
        )

    sampler = optuna.samplers.TPESampler(seed=run_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=model_name)
    study.optimize(
        objective,
        n_trials=effective_cap,
        timeout=optuna_cfg.timeout,
        callbacks=[persist_callback, early_stopping_callback],
        catch=(Exception,),
    )

    _sync_trials(entry, study, trial_seeds)
    executed_trials = len(study.trials)
    entry["executed_trials"] = executed_trials
    entry["early_stopping"]["stopped_early"] = early_stopping_state["stopped"]
    entry["early_stopping"]["stalled_trials"] = early_stopping_state["stalled_trials"]
    entry["early_stopping"]["best_value"] = early_stopping_state["best_value"]
    if patience is not None:
        if early_stopping_state["stopped"]:
            log_info(
                f"[{model_name}] search stopped early after {executed_trials}/{effective_cap} "
                f"trial(s) (configured budget {n_trials}); proceeding with the best trial"
            )
        else:
            log_info(
                f"[{model_name}] early stopping not triggered; "
                f"{executed_trials}/{effective_cap} trial(s) executed"
            )
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
    # reported through its params and optional checkpoint instead.  It is
    # trained with the run seed (the single seed controlling the whole run).
    final_cfg = dict(fixed)
    final_cfg.update(best_params)
    final_cfg["seed"] = run_seed
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

def _log_model_summary(config, results):
    """Print the final per-model summary of the search.

    For every configured model: the best combination of the searched (i.e.
    non-fixed) parameters, the number of executed Optuna steps, the total model
    time and the average time per Optuna step.
    """
    log_section("Hyperparameter Search Summary")
    entries = results.get("models", {})
    for model_name in config.models:
        entry = entries.get(model_name)
        log_subsection(model_name)
        if not entry:
            log_config("status", "no results recorded")
            continue

        best = entry.get("best_trial")
        searched = entry.get("search_space") or {}
        if best is None:
            best_params = "(no successful trial)"
        elif not searched:
            best_params = "(all parameters fixed)"
        else:
            best_params = json.dumps(best.get("params", {}), default=str, sort_keys=True)

        trials = entry.get("trials") or []
        steps = entry.get("executed_trials")
        if steps is None:
            steps = len(trials)
        durations = [
            trial["duration_seconds"]
            for trial in trials
            if trial.get("duration_seconds") is not None
        ]
        total = entry.get("duration_seconds")
        if durations:
            per_step = sum(durations) / len(durations)
        elif total is not None and steps:
            per_step = total / steps
        else:
            per_step = None

        log_config("status", entry.get("status", "unknown"))
        log_config("best params (searched)", best_params)
        log_config("optuna steps", steps)
        log_config("total time", fmt_seconds(total) if total is not None else "n/a")
        log_config("time per step", fmt_seconds(per_step) if per_step is not None else "n/a")


def run_hps(config_path, clean=False):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    config = load_hps_config(config_path)
    algorithm = config.algorithm
    training = config.training
    optuna_cfg = config.optuna

    log_section("Hyperparameter Search (AutoUAD / NPD)")
    log_config("config_file", str(config_path))
    log_config("data_path", config.data_path)
    log_config("models_directory", config.models_directory)
    log_config("output_file", config.output_file)
    log_config("validation_split_ratio", algorithm.validation_split_ratio)
    log_config("epsilon", algorithm.epsilon)
    log_config("split_seed", algorithm.split_seed)
    log_config("gaussian_seed", algorithm.gaussian_seed)
    log_config("max_debates", algorithm.max_debates)
    log_config("feature_key", algorithm.feature_key)
    log_config("n_trials", optuna_cfg.n_trials)
    log_config(
        "training (global)",
        (
            f"n_epochs_lr_reduce={training.n_epochs_lr_reduce}, "
            f"n_epochs_early_stop={training.n_epochs_early_stop}, "
            f"lr_reduce_improvement_pct={training.lr_reduce_improvement_pct}, "
            f"early_stop_improvement_pct={training.early_stop_improvement_pct}, "
            f"lr_reduce_factor={training.lr_reduce_factor}, min_lr={training.min_lr}"
        ),
    )
    log_config(
        "early_stopping",
        (
            f"patience={optuna_cfg.early_stopping_patience} trial(s), "
            f"min_improvement={optuna_cfg.early_stopping_min_improvement_pct}% "
            "(global thresholds, per-model histories)"
        )
        if optuna_cfg.early_stopping_patience is not None
        else "disabled",
    )

    output_path = Path(config.output_file)
    if clean and output_path.exists():
        output_path.unlink()
        log_info(f"--clean: removed {output_path}")

    overall_t0 = time()
    results = _read_results(output_path)
    completed_before = {
        name
        for name in config.models
        if _is_completed_model_entry(results.get("models", {}).get(name))
    }
    pending_models = [name for name in config.models if name not in completed_before]
    if completed_before:
        log_info(
            f"Resume: {len(completed_before)} model(s) already completed in {output_path}; "
            f"skipping {sorted(completed_before)}"
        )
    if not pending_models:
        _log_model_summary(config, results)
        log_section("Hyperparameter Search Finished")
        log_info(
            f"All {len(config.models)} configured model(s) are already completed in "
            f"{output_path}; nothing to do."
        )
        log_info(f"Total elapsed: {fmt_seconds(time() - overall_t0)}")
        return

    log_section("Loading default graph data")
    data_cache = {}
    default_keys = [algorithm.feature_key]
    default_data = load_graph_data(config.data_path, default_keys)
    data_cache[tuple(default_keys)] = default_data
    default_stats = graph_statistics(default_data.graphs)
    feature_dim = default_data.graphs[0].rounds[0][0][algorithm.feature_key].shape[0]
    log_info(
        f"Loaded {default_stats.n_debates} debate(s), {default_stats.n_rounds} round graph(s), "
        f"{default_stats.n_agents} '{algorithm.feature_key}' agent sample(s) of dimension "
        f"{feature_dim} from {len(default_data.dataset_tags) or 1} dataset(s): "
        f"{default_data.dataset_tags or ['combined']}"
    )
    log_info(f"Topology distribution: {default_stats.topology_counts}")
    log_info(f"Dataset distribution: {default_stats.dataset_counts}")
    if default_data.skipped_debates:
        log_warn(f"Skipped {default_data.skipped_debates} debate(s) with unusable graphs or keys.")

    if not results:
        results = {
            "script": "HyperParameterSearch.py",
            "algorithm": "AutoUAD (NPD-guided, real training graphs)",
            "config_file": str(config_path),
            "created_at": _now(),
            "updated_at": _now(),
            "data": {
                "data_path": str(config.data_path),
                "dataset_tags": default_data.dataset_tags,
                "feature_key": algorithm.feature_key,
                "n_debates": int(default_stats.n_debates),
                "n_rounds": int(default_stats.n_rounds),
                "n_agents": int(default_stats.n_agents),
                "feature_dim": int(feature_dim),
                "topology_distribution": default_stats.topology_counts,
                "dataset_distribution": default_stats.dataset_counts,
                "max_debates": algorithm.max_debates,
                "n_skipped_debates": int(default_data.skipped_debates),
            },
            "algorithm_params": {
                "validation_split_ratio": algorithm.validation_split_ratio,
                "epsilon": algorithm.epsilon,
                "split_seed": algorithm.split_seed,
                "gaussian_seed": algorithm.gaussian_seed,
                "max_debates": algorithm.max_debates,
                "split_unit": "debate",
                "split_stratification": ["topology_name", "dataset_tag"],
                "normalization": "global",
                "x_gen_structures": "validation_graphs",
            },
            "optuna_params": {
                "n_trials": optuna_cfg.n_trials,
                "sampler_seed": optuna_cfg.sampler_seed,
                "timeout": optuna_cfg.timeout,
                "early_stopping_patience": optuna_cfg.early_stopping_patience,
                "early_stopping_min_improvement_pct": (
                    optuna_cfg.early_stopping_min_improvement_pct
                ),
            },
            "training_params": {
                "n_epochs_lr_reduce": training.n_epochs_lr_reduce,
                "lr_reduce_improvement_pct": training.lr_reduce_improvement_pct,
                "lr_reduce_factor": training.lr_reduce_factor,
                "min_lr": training.min_lr,
                "n_epochs_early_stop": training.n_epochs_early_stop,
                "early_stop_improvement_pct": training.early_stop_improvement_pct,
            },
            "models": {},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="hps-data-", dir=output_path.parent))

    def _prepare_model_datasets(model_name, model_cfg, fixed, split_seed, gaussian_seed):
        feature_keys = list(model_cfg["feature_keys"])
        cache_key = tuple(feature_keys)
        loaded = data_cache.get(cache_key)
        if loaded is None:
            loaded = load_graph_data(config.data_path, feature_keys)
            data_cache[cache_key] = loaded
            log_info(
                f"Loaded {len(loaded.graphs)} debate(s) for feature_keys={feature_keys} "
                f"(skipped {loaded.skipped_debates})."
            )
        graphs = select_graphs(
            loaded, fixed.get("topologies"), algorithm.max_debates, split_seed
        )
        standardized, trn_graphs, val_graphs, x_gen = prepare_graph_search_data(
            graphs,
            feature_keys,
            algorithm.validation_split_ratio,
            split_seed,
            gaussian_seed,
        )
        trn_pkl = write_debate_pkl(temp_dir / f"{model_name}-train.pkl", trn_graphs, feature_keys)
        full_pkl = write_debate_pkl(temp_dir / f"{model_name}-full.pkl", standardized, feature_keys)
        val_debates = build_real_rounds(val_graphs)
        gen_debates = build_gen_rounds(val_graphs, x_gen, feature_keys, loaded.auxiliary_kinds)

        total_stats = graph_statistics(graphs)
        trn_stats = graph_statistics(trn_graphs)
        val_stats = graph_statistics(val_graphs)
        log_info(
            f"Graph data ready for {model_name}: {trn_stats.n_debates} train / "
            f"{val_stats.n_debates} validation debate(s) (stratified by topology+dataset, "
            f"split_seed={split_seed}); {trn_stats.n_agents} train / {val_stats.n_agents} "
            f"validation agent sample(s); {len(val_debates)} real validation debate(s) + "
            f"{len(gen_debates)} Gaussian debate(s) with mirrored structures "
            f"(gaussian_seed={gaussian_seed})."
        )
        dataset_stats = AttrDict(
            n_debates=total_stats.n_debates,
            n_rounds=total_stats.n_rounds,
            n_agents=total_stats.n_agents,
            n_train_debates=trn_stats.n_debates,
            n_val_debates=val_stats.n_debates,
            n_train_agents=trn_stats.n_agents,
            n_val_agents=val_stats.n_agents,
            topology_distribution=total_stats.topology_counts,
            dataset_distribution=total_stats.dataset_counts,
        )
        return trn_pkl, full_pkl, val_debates, gen_debates, dataset_stats

    def _setup_model(model_name, model_cfg):
        """Log and resolve one model's budget, seeds and search space."""
        log_section(f"Model: {model_name}")
        feature_keys = list(model_cfg["feature_keys"])
        global_n_trials = optuna_cfg.n_trials
        model_n_trials = model_cfg.get("n_trials")
        n_trials = model_n_trials if model_n_trials is not None else global_n_trials
        model_split_seed = model_cfg.get("split_seed")
        split_seed = model_split_seed if model_split_seed is not None else algorithm.split_seed
        model_gaussian_seed = model_cfg.get("gaussian_seed")
        if model_gaussian_seed is not None:
            gaussian_seed = model_gaussian_seed
        elif model_split_seed is not None:
            gaussian_seed = model_split_seed
        else:
            gaussian_seed = algorithm.gaussian_seed
        search_space, fixed = split_search_space(dict(model_cfg))
        run_seed = int(model_cfg.get("seed", optuna_cfg.sampler_seed))
        log_config("feature_keys", feature_keys)
        log_config(
            "n_trials",
            f"{n_trials}" + (
                f" (per-model override; global {global_n_trials})"
                if model_n_trials is not None else ""
            ),
        )
        log_config(
            "run_seed",
            f"{run_seed} (model seed; controls sampler, per-trial training seeds "
            "and final retrain)",
        )
        log_config(
            "split_seed",
            f"{split_seed}" + (
                f" (per-model override; global {algorithm.split_seed})"
                if model_split_seed is not None else ""
            ),
        )
        log_config(
            "gaussian_seed",
            f"{gaussian_seed}" + (
                f" (per-model override; global {algorithm.gaussian_seed})"
                if model_gaussian_seed is not None else ""
            ),
        )
        log_info(
            f"Search space: {search_space if search_space else '(none, fixed config)'}"
        )
        log_config("fixed_params", json.dumps(fixed, default=str, sort_keys=True))
        return feature_keys, n_trials, split_seed, gaussian_seed, search_space, fixed, run_seed

    def _register_model_failure(model_name, exc, model_t0):
        """Record a failed model without stopping the rest of the sweep."""
        entry = results.get("models", {}).get(model_name)
        if entry is None:
            entry = {}
        entry.setdefault("model_name", model_name)
        entry.setdefault("trials", [])
        entry.setdefault("best_trial", None)
        entry.setdefault("final_model", None)
        entry.setdefault("started_at", _now())
        entry["status"] = "failed"
        entry["error"] = str(exc)
        entry["duration_seconds"] = round(time() - model_t0, 3)
        results.setdefault("models", {})[model_name] = entry
        log_error(f"Model '{model_name}' failed; continuing with the next model.")
        for line in traceback.format_exc().strip().splitlines():
            log_error(line)
        return entry

    try:
        for model_name, model_cfg in config.models.items():
            if model_name in completed_before:
                log_section(f"Model: {model_name} -- SKIPPED")
                log_info(f"'{model_name}' already completed in {output_path}; keeping its results.")
                continue

            model_t0 = time()
            try:
                (
                    feature_keys,
                    n_trials,
                    split_seed,
                    gaussian_seed,
                    search_space,
                    fixed,
                    run_seed,
                ) = _setup_model(model_name, model_cfg)
                trn_pkl, full_pkl, val_debates, gen_debates, dataset_stats = _prepare_model_datasets(
                    model_name, model_cfg, fixed, split_seed, gaussian_seed
                )
                dataset_stats["split_seed"] = split_seed
                dataset_stats["gaussian_seed"] = gaussian_seed
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
                    val_debates,
                    gen_debates,
                    algorithm,
                    optuna_cfg,
                    results,
                    output_path,
                    run_seed,
                    split_seed,
                    gaussian_seed,
                )
                entry["data"] = dataset_stats
                entry["duration_seconds"] = round(time() - model_t0, 3)
                log_info(f"Model '{model_name}' finished in {fmt_seconds(time() - model_t0)}")
            except KeyboardInterrupt:
                log_warn("KeyboardInterrupt received. Completed results are preserved.")
                raise
            except Exception as exc:
                _register_model_failure(model_name, exc, model_t0)
            finally:
                _write_results(output_path, results)
    except KeyboardInterrupt:
        log_warn("KeyboardInterrupt received. All completed results have been saved.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        _write_results(output_path, results)
        _log_model_summary(config, results)
        log_section("Hyperparameter Search Finished")
        log_info(f"Total elapsed: {fmt_seconds(time() - overall_t0)}")
        log_info(f"Results JSON: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised AutoUAD/NPD hyperparameter search.")
    parser.add_argument("config_file", type=str, help="Path to the HPS configuration file.")
    parser.add_argument("--clean", action="store_true", help="Delete existing results and start fresh.")
    parsed_args = parser.parse_args()
    run_hps(parsed_args.config_file, clean=parsed_args.clean)
