"""Single source of truth for Data Generation configuration.

This module is the only place where the generation YAML is read, validated,
defaulted and normalised into the runtime configuration consumed by
``TrainDataGeneration.py`` and ``DebateDataGenerationLoop.py``.  No other
module may load the generation config or fill in missing values.

Usage::

    from GenerationConfigCheck import load_generation_config, build_debate_config
    config = load_generation_config("config-examples/generation-config.yaml")

or, from the command line, to validate a config without running anything::

    python GenerationConfigCheck.py config-examples/generation-config.yaml
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import yaml

from Utils import AttrDict


# ---------------------------------------------------------------------------
#  Schema
# ---------------------------------------------------------------------------

_ROOT_KEYS = {
    "llm",
    "debate",
    "datasets",
    "output_dir",
    "output_file",
    "process_text",
    "clean_debates",
    "text_process_workers",
    "text_processor_path",
    "text_processor_class_name",
    "text_processor_kwargs",
    "text_processor_device",
    "verbose",
}
_ROOT_REQUIRED = {
    "llm",
    "debate",
    "datasets",
    "output_dir",
    "output_file",
    "process_text",
    "clean_debates",
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
}

_DATASET_KEYS = {
    "tag",
    "loader_tag",
    "num_questions",
    "num_questions_on_random_topo",
    "questions_random_seed",
    "ma_dataset_path",
    "prompts_file",
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
        raise ValueError(f"Generation configuration file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in generation configuration '{path}': {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Generation configuration root must be a mapping")
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


# ---------------------------------------------------------------------------
#  Loader resolution
# ---------------------------------------------------------------------------

def _normalize_tag(tag: str) -> str:
    return "".join(ch for ch in str(tag).upper() if ch.isalnum())


def _dataset_loader_classes() -> dict[str, type]:
    import DatasetManager

    return {
        cls.TAG: cls
        for _, cls in inspect.getmembers(DatasetManager, inspect.isclass)
        if hasattr(cls, "TAG")
    }


def resolve_loader_tag(config_tag: str, explicit_loader_tag: str | None = None) -> str:
    """Resolve a config dataset tag to a canonical loader TAG."""
    classes = _dataset_loader_classes()

    if explicit_loader_tag:
        for tag in classes:
            if tag == explicit_loader_tag or _normalize_tag(tag) == _normalize_tag(explicit_loader_tag):
                return tag
        raise ValueError(
            f"Unknown loader_tag '{explicit_loader_tag}' for dataset tag '{config_tag}'. "
            f"Available loader TAGs: {sorted(classes)}"
        )

    if config_tag in classes:
        return config_tag

    normalized = _normalize_tag(config_tag)
    for tag in classes:
        if _normalize_tag(tag) == normalized:
            return tag

    alias = _DATASET_TAG_ALIASES.get(normalized)
    if alias and alias in classes:
        return alias

    raise ValueError(
        f"Could not resolve dataset tag '{config_tag}' to a DatasetManager loader. "
        f"Available loader TAGs: {sorted(classes)}"
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
    )


def _validate_datasets(raw: Any) -> list[AttrDict]:
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

        loader_tag = resolve_loader_tag(tag, item.get("loader_tag"))

        entries.append(
            AttrDict(
                tag=tag,
                loader_tag=loader_tag,
                loader_class=_dataset_loader_classes()[loader_tag],
                num_questions=int(item["num_questions"]),
                num_questions_on_random_topo=int(item["num_questions_on_random_topo"]),
                questions_random_seed=int(item["questions_random_seed"]),
                ma_dataset_path=ma_dataset_path,
                prompts_file=prompts_file,
            )
        )
    return entries


def _validate_text_processor(raw: dict[str, Any], process_text: bool) -> AttrDict:
    path = raw.get("text_processor_path")
    class_name = raw.get("text_processor_class_name")
    if process_text:
        _require_path(path, "text_processor_path")
        _require_str(class_name, "text_processor_class_name")
    elif path is not None:
        _require_path(path, "text_processor_path")
        _require_str(class_name, "text_processor_class_name")

    kwargs = raw.get("text_processor_kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("Configuration field 'text_processor_kwargs' must be a mapping")
    device = raw.get("text_processor_device")
    if device is not None:
        _require_str(device, "text_processor_device")

    return AttrDict(
        text_processor_path=path,
        text_processor_class_name=class_name,
        text_processor_kwargs=dict(kwargs),
        text_processor_device=device,
    )


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def load_generation_config(config_path: str | Path) -> AttrDict:
    """Load, validate and normalise a generation configuration."""
    raw = _load_yaml(config_path)
    _reject_unknown(raw, _ROOT_KEYS, "root")
    for key in _ROOT_REQUIRED:
        if key not in raw:
            raise ValueError(f"Missing required top-level configuration field: '{key}'")

    process_text = _require_bool(raw["process_text"], "process_text")
    clean_debates = _require_bool(raw["clean_debates"], "clean_debates")
    output_dir = _require_str(raw["output_dir"], "output_dir")
    output_file = _require_str(raw["output_file"], "output_file")
    if not output_file.lower().endswith(".pkl"):
        raise ValueError("Configuration field 'output_file' must end with '.pkl'")
    workers = _require_int(raw.get("text_process_workers", 0), "text_process_workers", minimum=0)
    verbose = _require_bool(raw.get("verbose", False), "verbose")

    config = AttrDict(
        llm=_validate_llm(_require_mapping(raw, "llm", "root")),
        debate=_validate_debate(_require_mapping(raw, "debate", "root")),
        datasets=_validate_datasets(raw["datasets"]),
        output_dir=output_dir,
        output_file=output_file,
        process_text=process_text,
        clean_debates=clean_debates,
        text_process_workers=workers,
        verbose=verbose,
    )
    config.update(_validate_text_processor(raw, process_text))
    return config


def build_debate_config(
    config: AttrDict,
    entry: AttrDict,
    topology_name: str,
    adjacency: list[list[int]],
    num_questions: int,
    questions_random_seed: int,
) -> AttrDict:
    """Build the per-topology runtime config consumed by DebateOrchestration."""
    return AttrDict(
        timeout=config.llm.timeout,
        llm_max_retries=config.llm.llm_max_retries,
        max_concurrent_inference=config.llm.max_concurrent_inference,
        num_agents=config.debate.num_agents,
        num_malicious_agents=config.debate.num_malicious_agents,
        malicious_seed=config.debate.malicious_seed,
        max_rounds=config.debate.max_rounds,
        consensus_threshold=config.debate.consensus_threshold,
        check_consensus_only_unflagged=config.debate.check_consensus_only_unflagged,
        no_consensus_check=config.debate.no_consensus_check,
        random_topo_seed=config.debate.random_topo_seed,
        density_range_for_random_topo=list(config.debate.density_range_for_random_topo),
        topology=adjacency,
        is_random_topology=(topology_name == "random"),
        topology_name=topology_name,
        num_questions=num_questions,
        questions_random_seed=questions_random_seed,
        loader_class=entry.loader_class,
        dataset_tag=entry.loader_tag,
        ma_dataset_path=entry.ma_dataset_path,
        prompts_file=entry.prompts_file,
        verbose=config.verbose,
    )


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate a Data Generation config YAML.")
    parser.add_argument("config_file", type=str, help="Path to the generation config YAML.")
    args = parser.parse_args()
    config = load_generation_config(args.config_file)
    datasets = ", ".join(f"{e.tag}->{e.loader_tag}" for e in config.datasets)
    print(f"OK: {args.config_file}")
    print(f"    datasets: {datasets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
