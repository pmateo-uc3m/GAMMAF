"""Fail-fast validation for the evaluation configuration."""

from pathlib import Path
from typing import Any

import yaml


OPTIONAL_BOOLEAN_DEFAULTS = {
    "no_consensus_check": False,
    "check_consensus_only_unflagged": False,
    "new_random_each_question": False,
    "no_defense_baseline": False,
    "clean_debates_with_empty_responses": False,
    "debug_mode": False,
    "static_adjacency_mode": False,
    "save_traces": False,
}


def _require_mapping(parent: dict[str, Any], key: str, location: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration field '{location}.{key}' must be a mapping")
    return value


def _require_positive(value: Any, name: str, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Configuration field '{name}' must be a positive number")
    if value <= 0 or (integer and int(value) != value):
        raise ValueError(f"Configuration field '{name}' must be a positive {'integer' if integer else 'number'}")


def _check_path(value: Any, name: str, must_exist: bool = True) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration field '{name}' must be a non-empty path")
    if must_exist and not Path(value).exists():
        raise ValueError(f"Configured path does not exist: {name}={value}")


def validate_evaluation_config(config_path: str | Path) -> dict[str, Any]:
    """Validate an evaluation YAML before model training or inference starts.

    Returns the parsed mapping with optional live-evaluation booleans filled in
    as false. The returned mapping is informational; the normal project config
    loader remains responsible for constructing its AttrDict configuration.
    """
    path = Path(config_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"Evaluation configuration file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in evaluation configuration '{path}': {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Evaluation configuration root must be a mapping")

    for key in ("models_directory", "output_file", "train_pkl_path"):
        if key not in config:
            raise ValueError(f"Missing required top-level configuration field: '{key}'")
    _check_path(config["models_directory"], "models_directory")
    _check_path(config["train_pkl_path"], "train_pkl_path")

    model_configs = config.get("defense_model_train_configs")
    if not isinstance(model_configs, dict) or not model_configs:
        raise ValueError("'defense_model_train_configs' must be a non-empty mapping")
    for model_name, model_config in model_configs.items():
        configs = model_config if isinstance(model_config, list) else [model_config]
        if not configs or any(not isinstance(item, dict) for item in configs):
            raise ValueError(f"Model configuration '{model_name}' must be a mapping or list of mappings")

    live = _require_mapping(config, "live_evaluation_config", "root")
    required_live = (
        "timeout", "questions_path", "questions_random_seed", "num_agents",
        "num_malicious_agents", "max_rounds", "consensus_threshold",
        "max_concurrent_inference", "num_questions", "n_questions_on_random_topo",
        "text_processor_path", "text_processor_class_name",
    )
    for key in required_live:
        if key not in live:
            raise ValueError(f"Missing required live_evaluation_config field: '{key}'")
    _check_path(live["questions_path"], "live_evaluation_config.questions_path")
    _check_path(live["text_processor_path"], "live_evaluation_config.text_processor_path")
    _require_positive(live["num_agents"], "live_evaluation_config.num_agents", integer=True)
    _require_positive(live["max_rounds"], "live_evaluation_config.max_rounds", integer=True)
    _require_positive(live["max_concurrent_inference"], "live_evaluation_config.max_concurrent_inference", integer=True)
    if live["num_malicious_agents"] < 0 or live["num_malicious_agents"] > live["num_agents"]:
        raise ValueError("live_evaluation_config.num_malicious_agents must be between 0 and num_agents")
    if not 0 < float(live["consensus_threshold"]) <= 1:
        raise ValueError("live_evaluation_config.consensus_threshold must be in (0, 1]")

    for key, default in OPTIONAL_BOOLEAN_DEFAULTS.items():
        value = live.setdefault(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"Configuration field 'live_evaluation_config.{key}' must be boolean")

    return config
