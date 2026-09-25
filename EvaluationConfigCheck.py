"""Single source of truth for Main Evaluation configuration.

This module is the only place where the evaluation YAML is read, validated,
defaulted and normalised.  It also owns the hyperparameter-search expansion
and the per-model (defense model) configuration loading used by the defense
models themselves.

`Evaluation` here means both the standard evaluation and the consolidated
hyperparameter search (``--hps``), which share the same YAML.

Usage::

    from EvaluationConfigCheck import load_evaluation_config
    config = load_evaluation_config("config-examples/evaluation-config.yaml")

or, from the command line, to validate a config without running anything::

    python EvaluationConfigCheck.py config-examples/evaluation-config.yaml
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import itertools
import json
import os
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from Utils import AttrDict


# ---------------------------------------------------------------------------
#  Schema
# ---------------------------------------------------------------------------

_ROOT_KEYS = {
    "models_directory",
    "output_file",
    "train_pkl_path",
    "llm",
    "debate",
    "datasets",
    "evaluation",
    "text_processor_path",
    "text_processor_class_name",
    "text_processor_kwargs",
    "text_processor_device",
    "defense_model_train_configs",
    "hyperparameter_search",
    "training",
}
_ROOT_REQUIRED = {
    "models_directory",
    "output_file",
    "llm",
    "debate",
    "datasets",
    "evaluation",
    "text_processor_path",
    "text_processor_class_name",
    "defense_model_train_configs",
}

_LLM_KEYS = {"timeout", "llm_max_retries", "max_concurrent_inference"}

_DEBATE_KEYS = {
    "num_agents",
    "num_malicious_agents",
    "malicious_seed",
    "max_rounds",
    "consensus_threshold",
    "check_consensus_only_unflagged",
    "no_consensus_check",
    "new_random_each_question",
    "random_topo_seed",
    "density_range_for_random_topo",
    "clean_debates",
}

_DATASET_KEYS = {
    "tag",
    "loader_tag",
    "num_questions",
    "num_questions_on_random_topo",
    "questions_random_seed",
    "ma_dataset_path",
    "prompts_file",
    "hps_indexes",
}

_EVALUATION_KEYS = {
    "questions_path",
    "questions_class_name",
    "python_seed",
    "numpy_seed",
    "answer_seed",
    "top_k_defense",
    "no_defense_baseline",
    "save_traces",
    "debug_mode",
    "static_adjacency_mode",
    "topologies_file",
    "topologies_from_pkl",
}

_HPS_KEYS = {
    "total_samples",
    "run_samples",
    "split_seed",
    "index_pkl",
    "index_pkl_dir",
    "results_csv",
}

_DATASET_TAG_ALIASES = {
    "INJECAGENT": "TA",
    "INJECAGENTTA": "TA",
    "MSMARCO": "MA",
    "MSMARCOCONTAMINATED": "MA",
}


# ---------------------------------------------------------------------------
#  Validation helpers
# ---------------------------------------------------------------------------

def _load_yaml(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"Evaluation configuration file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in evaluation configuration '{path}': {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Evaluation configuration root must be a mapping")
    return data


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown configuration key(s) at {location}: {', '.join(unknown)}"
        )


def _require_mapping(parent: dict[str, Any], key: str, location: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration field '{location}.{key}' must be a mapping")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Configuration field '{name}' must be boolean")
    return value


def _require_positive(value: Any, name: str, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Configuration field '{name}' must be a positive number")
    if value <= 0 or (integer and int(value) != value):
        raise ValueError(
            f"Configuration field '{name}' must be a positive "
            f"{'integer' if integer else 'number'}"
        )


def _require_int(value: Any, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Configuration field '{name}' must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"Configuration field '{name}' must be >= {minimum}")
    return value


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration field '{name}' must be a non-empty string")
    return value


def _require_path(value: Any, name: str) -> str:
    _require_str(value, name)
    if not Path(value).exists():
        raise ValueError(f"Configured path does not exist: {name}={value}")
    return value


def _safe_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
    return cleaned.strip("_") or "unnamed"


# ---------------------------------------------------------------------------
#  Loader resolution (questions loader classes)
# ---------------------------------------------------------------------------

def _normalize_tag(tag: str) -> str:
    return "".join(ch for ch in str(tag).upper() if ch.isalnum())


def _load_module(file_path: str | Path):
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"Questions loader file does not exist: {path}")
    module_name = f"configcheck_loader_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_class_from_path(file_path: str | Path, class_name: str):
    """Load an explicit class by name from a Python file."""
    module = _load_module(file_path)
    loader = getattr(module, class_name, None)
    if loader is None:
        raise ValueError(f"Class '{class_name}' not found in '{file_path}'")
    return loader


def get_available_dataset_tags(file_path: str | Path) -> dict[str, type]:
    """Return ``{loader TAG: loader class}`` declared in *file_path*."""
    module = _load_module(file_path)
    return {
        obj.TAG: obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if isinstance(getattr(obj, "TAG", None), str) and obj.TAG
    }


def resolve_loader_tag_from_path(
    file_path: str | Path, dataset_tag: str, explicit_loader_tag: str | None = None
) -> type:
    """Resolve a config tag (or explicit loader TAG) to a loader class."""
    classes = get_available_dataset_tags(file_path)

    if explicit_loader_tag:
        for tag, cls in classes.items():
            if tag == explicit_loader_tag or _normalize_tag(tag) == _normalize_tag(explicit_loader_tag):
                return cls
        raise ValueError(
            f"Unknown loader_tag '{explicit_loader_tag}' for dataset tag '{dataset_tag}'. "
            f"Available loader TAGs: {sorted(classes)}"
        )

    if dataset_tag in classes:
        return classes[dataset_tag]

    normalized = _normalize_tag(dataset_tag)
    for tag, cls in classes.items():
        if _normalize_tag(tag) == normalized:
            return cls

    alias = _DATASET_TAG_ALIASES.get(normalized)
    if alias and alias in classes:
        return classes[alias]

    raise ValueError(
        f"Could not resolve dataset tag '{dataset_tag}' to a questions loader in "
        f"{file_path}. Available loader TAGs: {sorted(classes)}"
    )


# ---------------------------------------------------------------------------
#  Section validation
# ---------------------------------------------------------------------------

def _validate_llm(raw: dict[str, Any]) -> AttrDict:
    _reject_unknown(raw, _LLM_KEYS, "llm")
    for key in ("timeout", "llm_max_retries", "max_concurrent_inference"):
        if key not in raw:
            raise ValueError(f"Missing required llm field: 'llm.{key}'")
    _require_positive(raw["timeout"], "llm.timeout")
    _require_positive(raw["llm_max_retries"], "llm.llm_max_retries", integer=True)
    _require_positive(
        raw["max_concurrent_inference"], "llm.max_concurrent_inference", integer=True
    )
    return AttrDict(
        timeout=float(raw["timeout"]),
        llm_max_retries=int(raw["llm_max_retries"]),
        max_concurrent_inference=int(raw["max_concurrent_inference"]),
    )


def _validate_debate(raw: dict[str, Any]) -> AttrDict:
    _reject_unknown(raw, _DEBATE_KEYS, "debate")
    required = (
        "num_agents",
        "num_malicious_agents",
        "malicious_seed",
        "max_rounds",
        "consensus_threshold",
        "random_topo_seed",
        "density_range_for_random_topo",
    )
    for key in required:
        if key not in raw:
            raise ValueError(f"Missing required debate field: 'debate.{key}'")

    _require_positive(raw["num_agents"], "debate.num_agents", integer=True)
    _require_int(raw["num_malicious_agents"], "debate.num_malicious_agents", minimum=0)
    if raw["num_malicious_agents"] > raw["num_agents"]:
        raise ValueError(
            "Configuration field 'debate.num_malicious_agents' must be between 0 "
            "and 'debate.num_agents'"
        )
    _require_positive(raw["max_rounds"], "debate.max_rounds", integer=True)
    _require_positive(raw["consensus_threshold"], "debate.consensus_threshold")
    if not 0 < float(raw["consensus_threshold"]) <= 1:
        raise ValueError("Configuration field 'debate.consensus_threshold' must be in (0, 1]")
    _require_int(raw["malicious_seed"], "debate.malicious_seed")
    _require_int(raw["random_topo_seed"], "debate.random_topo_seed")

    density = raw["density_range_for_random_topo"]
    if (
        not isinstance(density, list)
        or len(density) != 2
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in density)
    ):
        raise ValueError(
            "Configuration field 'debate.density_range_for_random_topo' must be a "
            "[min, max] list of numbers"
        )
    if not (0 <= density[0] <= density[1] <= 1):
        raise ValueError(
            "Configuration field 'debate.density_range_for_random_topo' must satisfy "
            "0 <= min <= max <= 1"
        )

    return AttrDict(
        num_agents=int(raw["num_agents"]),
        num_malicious_agents=int(raw["num_malicious_agents"]),
        malicious_seed=int(raw["malicious_seed"]),
        max_rounds=int(raw["max_rounds"]),
        consensus_threshold=float(raw["consensus_threshold"]),
        check_consensus_only_unflagged=_require_bool(
            raw.get("check_consensus_only_unflagged", False),
            "debate.check_consensus_only_unflagged",
        ),
        no_consensus_check=_require_bool(
            raw.get("no_consensus_check", False), "debate.no_consensus_check"
        ),
        new_random_each_question=_require_bool(
            raw.get("new_random_each_question", True), "debate.new_random_each_question"
        ),
        random_topo_seed=int(raw["random_topo_seed"]),
        density_range_for_random_topo=[float(density[0]), float(density[1])],
        clean_debates=_require_bool(
            raw.get("clean_debates", False), "debate.clean_debates"
        ),
    )


def _validate_evaluation_section(raw: dict[str, Any]) -> AttrDict:
    _reject_unknown(raw, _EVALUATION_KEYS, "evaluation")
    required = (
        "questions_path",
        "python_seed",
        "numpy_seed",
        "answer_seed",
        "top_k_defense",
        "no_defense_baseline",
        "save_traces",
        "debug_mode",
        "static_adjacency_mode",
    )
    for key in required:
        if key not in raw:
            raise ValueError(f"Missing required evaluation field: 'evaluation.{key}'")

    _require_path(raw["questions_path"], "evaluation.questions_path")
    _require_int(raw["python_seed"], "evaluation.python_seed")
    _require_int(raw["numpy_seed"], "evaluation.numpy_seed")
    _require_int(raw["answer_seed"], "evaluation.answer_seed")
    _require_positive(raw["top_k_defense"], "evaluation.top_k_defense", integer=True)

    questions_class_name = raw.get("questions_class_name")
    if questions_class_name is not None:
        _require_str(questions_class_name, "evaluation.questions_class_name")

    topologies_file = raw.get("topologies_file")
    if topologies_file is not None:
        _require_path(topologies_file, "evaluation.topologies_file")
    topologies_from_pkl = raw.get("topologies_from_pkl")
    if topologies_from_pkl is not None:
        _require_path(topologies_from_pkl, "evaluation.topologies_from_pkl")

    return AttrDict(
        questions_path=raw["questions_path"],
        questions_class_name=questions_class_name,
        python_seed=int(raw["python_seed"]),
        numpy_seed=int(raw["numpy_seed"]),
        answer_seed=int(raw["answer_seed"]),
        top_k_defense=int(raw["top_k_defense"]),
        no_defense_baseline=_require_bool(
            raw["no_defense_baseline"], "evaluation.no_defense_baseline"
        ),
        save_traces=_require_bool(raw["save_traces"], "evaluation.save_traces"),
        debug_mode=_require_bool(raw["debug_mode"], "evaluation.debug_mode"),
        static_adjacency_mode=_require_bool(
            raw["static_adjacency_mode"], "evaluation.static_adjacency_mode"
        ),
        topologies_file=topologies_file,
        topologies_from_pkl=topologies_from_pkl,
    )


def _validate_datasets(raw: Any, evaluation: AttrDict, hps: AttrDict | None) -> list[AttrDict]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("Configuration field 'datasets' must be a non-empty list")

    entries: list[AttrDict] = []
    seen_config_tags: set[str] = set()
    for index, item in enumerate(raw):
        location = f"datasets[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"Configuration field '{location}' must be a mapping")
        _reject_unknown(item, _DATASET_KEYS, location)
        if not item.get("tag"):
            raise ValueError(f"Configuration field '{location}.tag' is required")
        tag = _require_str(item["tag"], f"{location}.tag")
        if tag in seen_config_tags:
            raise ValueError(
                f"Duplicate dataset tag '{tag}' in 'datasets'. Tags must be unique; "
                "the same loader may appear multiple times with different configs."
            )
        seen_config_tags.add(tag)

        for key in ("num_questions", "num_questions_on_random_topo"):
            if key not in item:
                raise ValueError(f"Missing required field: '{location}.{key}'")
            _require_int(item[key], f"{location}.{key}", minimum=0)
        if "questions_random_seed" not in item:
            raise ValueError(
                f"Missing required field: '{location}.questions_random_seed'"
            )
        _require_int(item["questions_random_seed"], f"{location}.questions_random_seed")

        ma_dataset_path = item.get("ma_dataset_path")
        if ma_dataset_path is not None:
            _require_path(ma_dataset_path, f"{location}.ma_dataset_path")

        prompts_file = item.get("prompts_file")
        if prompts_file is not None:
            _require_path(prompts_file, f"{location}.prompts_file")

        hps_indexes = item.get("hps_indexes")
        if hps_indexes is not None:
            _require_str(hps_indexes, f"{location}.hps_indexes")

        if evaluation.questions_class_name:
            loader_cls = load_class_from_path(
                evaluation.questions_path, evaluation.questions_class_name
            )
            loader_tag = getattr(loader_cls, "TAG", tag)
        else:
            loader_cls = resolve_loader_tag_from_path(
                evaluation.questions_path, tag, explicit_loader_tag=item.get("loader_tag")
            )
            loader_tag = loader_cls.TAG

        if hps_indexes is not None:
            hps_index_path = hps_indexes
        elif hps is not None and hps.index_pkl_dir is not None:
            hps_index_path = str(
                Path(hps.index_pkl_dir) / f"{_safe_filename(tag)}-index.pkl"
            )
        else:
            hps_index_path = None

        entries.append(
            AttrDict(
                tag=tag,
                loader_tag=loader_tag,
                loader_class=loader_cls,
                num_questions=int(item["num_questions"]),
                num_questions_on_random_topo=int(item["num_questions_on_random_topo"]),
                questions_random_seed=int(item["questions_random_seed"]),
                ma_dataset_path=ma_dataset_path,
                prompts_file=prompts_file,
                hps_indexes=hps_indexes,
                hps_index_path=hps_index_path,
            )
        )
    return entries


def _normalize_model_config(entry: dict[str, Any]) -> AttrDict:
    """Normalise one defense-model configuration entry (aliases + defaults)."""
    if not isinstance(entry, dict):
        raise ValueError("Defense model configuration entries must be mappings")
    config = AttrDict(copy.deepcopy(entry))

    if "seed" not in config:
        raise ValueError("Defense model configuration entries require a 'seed'")
    if "lr" not in config and "learning_rate" in config:
        config["lr"] = config["learning_rate"]
    if "epochs" not in config and "num_epochs" in config:
        config["epochs"] = config["num_epochs"]
    if "lr_patience_factor" in config and "lr_reduce_factor" not in config:
        config["lr_reduce_factor"] = config["lr_patience_factor"]
    if "lr_patience_max" in config and "n_epochs_lr_reduce" not in config:
        config["n_epochs_lr_reduce"] = config["lr_patience_max"]
    if "lr_improvement_pct" in config and "lr_reduce_improvement_pct" not in config:
        config["lr_reduce_improvement_pct"] = config["lr_improvement_pct"]
    if "early_stop" in config and "n_epochs_early_stop" not in config:
        config["n_epochs_early_stop"] = config["early_stop"]
    config.setdefault("n_epochs_lr_reduce", 5)
    config.setdefault("lr_reduce_improvement_pct", 1.0)
    config.setdefault("lr_reduce_factor", 0.5)
    config.setdefault("min_lr", 1e-6)
    config.setdefault("n_epochs_early_stop", 12)
    config.setdefault("early_stop_improvement_pct", config["lr_reduce_improvement_pct"])
    config.setdefault("val_split", 0.2)
    config.setdefault("prop_steps", 2)
    config.setdefault("persistence_decay", 0.5)
    config.setdefault("base_weight", 0.2)
    config.setdefault("reciprocity_weight", 0.2)
    config.setdefault("cluster_weight", 1.0)
    config.setdefault("persistence_weight", 0.8)
    config.setdefault("pair_weight", 1.5)
    config.setdefault("deviation_weight", 0.3)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("target_ema_decay", 0.8)
    config.setdefault("influence_ema_decay", 0.8)
    config.setdefault("max_persistence_window", 64)
    config.setdefault("spine_top_k", 3)
    component_weights = config.get("component_weights")
    if component_weights is None:
        config["component_weights"] = {
            "outgoing": 1.0,
            "amplifier": 1.0,
            "bridge": 1.0,
        }
    elif not isinstance(component_weights, dict):
        raise ValueError(
            "Defense model configuration 'component_weights' must be a mapping"
        )
    config.setdefault("topologies", None)
    config.setdefault("device", "cpu")
    config.setdefault("threshold", None)
    config.setdefault("top_k", 1)

    seed = config["seed"]
    if not isinstance(seed, (list, tuple)):
        config.setdefault("split_seed", seed)
        config.setdefault("dataloader_seed", seed)
        config.setdefault("data_seed", seed)
    return config


_TRAINING_KEYS = {
    "n_epochs_lr_reduce",
    "lr_reduce_improvement_pct",
    "lr_reduce_factor",
    "min_lr",
    "n_epochs_early_stop",
    "early_stop_improvement_pct",
}

_TRAINING_DEFAULTS = {
    "n_epochs_lr_reduce": 5,
    "lr_reduce_improvement_pct": 1.0,
    "lr_reduce_factor": 0.5,
    "min_lr": 1e-6,
    "n_epochs_early_stop": 12,
    "early_stop_improvement_pct": 1.0,
}


def _require_number(value: Any, name: str, minimum: float | None = None,
                    maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Configuration field '{name}' must be a number")
    if minimum is not None and value < minimum:
        raise ValueError(f"Configuration field '{name}' must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"Configuration field '{name}' must be <= {maximum}")
    return float(value)


def validate_training_config(raw: Any) -> AttrDict:
    """Validate the optional global ``training`` section.

    These values are the defaults for every defense model's training loop;
    a model's own config section overrides any of them for that model only.
    """
    if raw is None:
        return AttrDict(_TRAINING_DEFAULTS)
    if not isinstance(raw, dict):
        raise ValueError("Configuration field 'training' must be a mapping")
    _reject_unknown(raw, _TRAINING_KEYS, "training")

    training = AttrDict(_TRAINING_DEFAULTS)
    if "n_epochs_lr_reduce" in raw:
        training["n_epochs_lr_reduce"] = _require_int(
            raw["n_epochs_lr_reduce"], "training.n_epochs_lr_reduce", minimum=1
        )
    if "n_epochs_early_stop" in raw:
        training["n_epochs_early_stop"] = _require_int(
            raw["n_epochs_early_stop"], "training.n_epochs_early_stop", minimum=1
        )
    if "lr_reduce_improvement_pct" in raw:
        training["lr_reduce_improvement_pct"] = _require_number(
            raw["lr_reduce_improvement_pct"], "training.lr_reduce_improvement_pct", minimum=0.0
        )
    if "early_stop_improvement_pct" in raw:
        training["early_stop_improvement_pct"] = _require_number(
            raw["early_stop_improvement_pct"], "training.early_stop_improvement_pct", minimum=0.0
        )
    if "lr_reduce_factor" in raw:
        training["lr_reduce_factor"] = _require_number(
            raw["lr_reduce_factor"], "training.lr_reduce_factor", minimum=0.0, maximum=1.0
        )
    if "min_lr" in raw:
        training["min_lr"] = _require_number(raw["min_lr"], "training.min_lr", minimum=0.0)
    return training


def merge_training_defaults(model_cfg: dict[str, Any], training: AttrDict | None) -> dict[str, Any]:
    """Fill a model config's missing training keys from the global section."""
    merged = dict(model_cfg)
    if training:
        for key, value in training.items():
            merged.setdefault(key, value)
    return merged


def _validate_model_configs(raw: Any, train_pkl_path: str | None, training: AttrDict | None = None) -> AttrDict:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "'defense_model_train_configs' must be a non-empty mapping of "
            "model_name -> config (or list of configs)"
        )

    normalized = AttrDict()
    for model_name, model_cfg in raw.items():
        entries = model_cfg if isinstance(model_cfg, list) else [model_cfg]
        if not entries or any(not isinstance(item, dict) for item in entries):
            raise ValueError(
                f"Defense model configuration '{model_name}' must be a mapping or a "
                f"non-empty list of mappings"
            )

        normalized_entries = []
        for index, entry in enumerate(entries):
            config = _normalize_model_config(merge_training_defaults(entry, training))
            if "pkl_train" not in config:
                if not train_pkl_path:
                    raise ValueError(
                        f"Defense model '{model_name}' has no 'pkl_train' and no "
                        f"top-level 'train_pkl_path' fallback is configured"
                    )
                config["pkl_train"] = train_pkl_path
            _require_path(config["pkl_train"], f"defense_model_train_configs.{model_name}.pkl_train")
            if "run_name" not in config:
                config["run_name"] = model_name if len(entries) == 1 else f"{model_name}_{index}"
            normalized_entries.append(config)
        normalized[model_name] = normalized_entries
    return normalized


def _validate_hyperparameter_search(raw: Any) -> AttrDict | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("Configuration field 'hyperparameter_search' must be a mapping")
    _reject_unknown(raw, _HPS_KEYS, "hyperparameter_search")

    if "total_samples" not in raw:
        raise ValueError("Missing required field: 'hyperparameter_search.total_samples'")
    if "results_csv" not in raw:
        raise ValueError("Missing required field: 'hyperparameter_search.results_csv'")
    _require_positive(
        raw["total_samples"], "hyperparameter_search.total_samples", integer=True
    )
    run_samples = raw.get("run_samples")
    if run_samples is not None:
        _require_positive(run_samples, "hyperparameter_search.run_samples", integer=True)
    split_seed = _require_int(raw.get("split_seed", 42), "hyperparameter_search.split_seed")

    index_pkl = raw.get("index_pkl")
    index_pkl_dir = raw.get("index_pkl_dir")
    if index_pkl is None and index_pkl_dir is None:
        raise ValueError(
            "'hyperparameter_search' requires 'index_pkl' and/or 'index_pkl_dir' "
            "to persist the selected per-dataset indexes"
        )
    if index_pkl is not None:
        _require_str(index_pkl, "hyperparameter_search.index_pkl")
    if index_pkl_dir is not None:
        _require_str(index_pkl_dir, "hyperparameter_search.index_pkl_dir")
    if index_pkl_dir is None:
        index_pkl_dir = str(Path(index_pkl).parent)
    _require_str(raw["results_csv"], "hyperparameter_search.results_csv")

    return AttrDict(
        total_samples=int(raw["total_samples"]),
        run_samples=int(run_samples) if run_samples is not None else None,
        split_seed=split_seed,
        index_pkl=index_pkl,
        index_pkl_dir=index_pkl_dir,
        results_csv=raw["results_csv"],
    )


def _validate_text_processor(raw: dict[str, Any]) -> AttrDict:
    _require_path(raw["text_processor_path"], "text_processor_path")
    _require_str(raw["text_processor_class_name"], "text_processor_class_name")
    kwargs = raw.get("text_processor_kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("Configuration field 'text_processor_kwargs' must be a mapping")
    device = raw.get("text_processor_device", "cpu")
    _require_str(device, "text_processor_device")
    return AttrDict(
        text_processor_path=raw["text_processor_path"],
        text_processor_class_name=raw["text_processor_class_name"],
        text_processor_kwargs=dict(kwargs),
        text_processor_device=device,
    )


# ---------------------------------------------------------------------------
#  Public loading API
# ---------------------------------------------------------------------------

def load_evaluation_config(config_path: str | Path) -> AttrDict:
    """Load, validate and normalise a Main Evaluation configuration."""
    raw = _load_yaml(config_path)
    _reject_unknown(raw, _ROOT_KEYS, "root")
    for key in _ROOT_REQUIRED:
        if key not in raw:
            raise ValueError(f"Missing required top-level configuration field: '{key}'")

    models_directory = _require_path(raw["models_directory"], "models_directory")
    if not Path(models_directory).is_dir():
        raise ValueError(f"Configuration field 'models_directory' is not a directory: {models_directory}")
    output_file = _require_str(raw["output_file"], "output_file")

    train_pkl_path = raw.get("train_pkl_path")
    if train_pkl_path is not None:
        _require_path(train_pkl_path, "train_pkl_path")

    hps = _validate_hyperparameter_search(raw.get("hyperparameter_search"))
    evaluation = _validate_evaluation_section(
        _require_mapping(raw, "evaluation", "root")
    )
    training = validate_training_config(raw.get("training"))

    config = AttrDict(
        models_directory=models_directory,
        output_file=output_file,
        train_pkl_path=train_pkl_path,
        llm=_validate_llm(_require_mapping(raw, "llm", "root")),
        debate=_validate_debate(_require_mapping(raw, "debate", "root")),
        evaluation=evaluation,
        datasets=_validate_datasets(raw["datasets"], evaluation, hps),
        defense_model_train_configs=_validate_model_configs(
            raw["defense_model_train_configs"], train_pkl_path, training
        ),
        hyperparameter_search=hps,
        training=training,
    )
    config.update(_validate_text_processor(raw))
    return config


def load_defense_model_config(config_path: str | Path) -> AttrDict:
    """Load and normalise a single defense-model configuration file.

    Used by the defense models themselves (their ``Master`` classes) so that
    no model performs config loading, aliasing or defaulting on its own.
    """
    return _normalize_model_config(_load_yaml(config_path))


# ---------------------------------------------------------------------------
#  Hyperparameter-search expansion
# ---------------------------------------------------------------------------

HPS_INTERNAL_KEYS = {"hyperparameter_search"}

STRUCTURAL_LIST_NAMES = {
    "density_range_for_random_topo",
    "topology",
    "topologies",
}


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool)) or value is None


def _is_scalar_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(_is_scalar(x) for x in value)


def _find_list_params(obj, prefix=()):
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in HPS_INTERNAL_KEYS:
                continue
            path = prefix + (key,)
            if isinstance(value, list):
                if key in STRUCTURAL_LIST_NAMES:
                    _descend_structural(value, path, found)
                elif _is_scalar_list(value):
                    found.append((path, list(value)))
                else:
                    _descend_structural(value, path, found)
            elif isinstance(value, dict):
                found.extend(_find_list_params(value, path))
    return found


def _descend_structural(lst, path, found):
    for index, item in enumerate(lst):
        if isinstance(item, dict):
            found.extend(_find_list_params(item, path + (index,)))
        elif isinstance(item, list):
            _descend_structural(item, path + (index,), found)


def _set_path(obj, path, value):
    current = obj
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value


def _path_to_str(path):
    return ".".join(str(part) for part in path)


def _sanitize_name_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value}".replace(".", "-")
    if value is None:
        return "none"
    return str(value).replace(".", "-").replace(" ", "_").replace("/", "_")


def _make_hp_suffix(varied):
    leaves: dict[str, list[str]] = {}
    for key in varied:
        leaves.setdefault(key.split(".")[-1], []).append(key)
    parts = []
    for key, value in sorted(varied.items()):
        leaf = key.split(".")[-1]
        name = leaf if len(leaves[leaf]) == 1 else key.replace(".", "").replace("_", "")
        parts.append(f"{name}{_sanitize_name_value(value)}")
    return "_".join(parts) if parts else ""


def _expand_model_config(model_cfg):
    list_params = _find_list_params(model_cfg)
    if not list_params:
        return [(copy.deepcopy(model_cfg), {})]

    list_params.sort(key=lambda item: _path_to_str(item[0]))
    paths = [path for path, _ in list_params]
    value_lists = [values for _, values in list_params]

    expanded = []
    for combo in itertools.product(*value_lists):
        effective = copy.deepcopy(model_cfg)
        varied = {}
        for path, value in zip(paths, combo):
            _set_path(effective, path, value)
            varied[_path_to_str(path)] = value
        expanded.append((effective, varied))
    return expanded


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _config_signature(model_name: str, effective_dict: dict) -> str:
    payload = {"model": model_name, "config": effective_dict}
    return sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _strip_hps_internal(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    for key in HPS_INTERNAL_KEYS:
        out.pop(key, None)
    return out


def build_run_plans(config_path: str | Path) -> list[dict]:
    """Expand the model-config section into one plan per (model, HP combo)."""
    raw = _load_yaml(config_path)
    section = raw.get("defense_model_train_configs")
    if not isinstance(section, dict) or not section:
        raise ValueError(
            "Missing embedded defense train config section "
            "'defense_model_train_configs' in the evaluation config"
        )

    global_cfg = _strip_hps_internal(raw)
    plans = []
    for model_name, model_cfg in section.items():
        if isinstance(model_cfg, dict):
            entries = [model_cfg]
        elif isinstance(model_cfg, list):
            entries = model_cfg
        else:
            raise ValueError(
                f"Defense model configuration '{model_name}' must be a mapping or a "
                f"list of mappings"
            )

        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Defense model configuration '{model_name}' entries must be mappings"
                )
            base_name = entry.get("run_name", model_name)

            for combo_cfg, model_varied in _expand_model_config(entry):
                varied = {
                    f"defense_model_train_configs.{model_name}.{key}": value
                    for key, value in model_varied.items()
                }
                hp_suffix = _make_hp_suffix(varied)
                run_name = f"{base_name}_{hp_suffix}" if hp_suffix else base_name
                combo_cfg["run_name"] = run_name

                effective = copy.deepcopy(global_cfg)
                effective["defense_model_train_configs"] = {model_name: combo_cfg}

                identity_cfg = copy.deepcopy(combo_cfg)
                identity_cfg.pop("run_name", None)

                plans.append(
                    {
                        "model_name": model_name,
                        "run_name": run_name,
                        "eff": effective,
                        "varied": varied,
                        "signature": _config_signature(model_name, identity_cfg),
                    }
                )
    return plans


def write_effective_config(effective_dict: dict) -> str:
    """Write an expanded effective config to a temporary YAML file."""
    fd, temp_path = tempfile.mkstemp(prefix="hps-main-", suffix=".yaml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(effective_dict, handle, sort_keys=False)
    except Exception:
        Path(temp_path).unlink(missing_ok=True)
        raise
    return temp_path


def write_model_config(run_name: str, model_config: dict) -> str:
    """Write one model configuration to a temporary YAML file."""
    fd, temp_path = tempfile.mkstemp(prefix=f"{_safe_filename(run_name)}-", suffix=".yaml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(model_config), handle, sort_keys=False)
    except Exception:
        Path(temp_path).unlink(missing_ok=True)
        raise
    return temp_path


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate a Main Evaluation config YAML.")
    parser.add_argument("config_file", type=str, help="Path to the evaluation config YAML.")
    args = parser.parse_args()
    config = load_evaluation_config(args.config_file)
    datasets = ", ".join(f"{e.tag}->{e.loader_tag}" for e in config.datasets)
    mode = "hyperparameter search" if config.hyperparameter_search is not None else "standard"
    print(f"OK: {args.config_file}")
    print(f"    mode: {mode}")
    print(f"    datasets: {datasets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
