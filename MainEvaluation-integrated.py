"""Integrated evaluation entry point for legacy and temporal models."""

import importlib.util
import pickle
import runpy
import sys
from pathlib import Path

import Utils as _utils
from ConfigCheck import validate_evaluation_config

_load_config_from_path = _utils.load_config_from_path

root = Path(__file__).resolve().parent

if len(sys.argv) < 2:
    raise ValueError("Usage: python MainEvaluation-integrated.py <config_file> [--clean]")
validate_evaluation_config(sys.argv[1])


class _LegacyPickleList(list):
    """List-format training pickles with the metadata API expected by MainEvaluation."""

    def get(self, key, default=None):
        if key == "idx_metadata":
            return default
        return default


_pickle_load = pickle.load


def _load_pickle_compatibility(file_obj, *args, **kwargs):
    value = _pickle_load(file_obj, *args, **kwargs)
    if isinstance(value, list) and not isinstance(value, _LegacyPickleList):
        return _LegacyPickleList(value)
    return value


# MainEvaluation historically calls .get("idx_metadata") on the training
# pickle. Keep list pickles valid for the integrated entry point only.
pickle.load = _load_pickle_compatibility


def _load_config_with_optional_bool_defaults(config_path):
    """Normalize optional evaluator switches without hiding required settings."""
    config = _load_config_from_path(config_path)
    optional_defaults = {
        "no_consensus_check": False,
        "check_consensus_only_unflagged": False,
        "new_random_each_question": False,
        "no_defense_baseline": False,
        "clean_debates_with_empty_responses": False,
        "debug_mode": False,
        "static_adjacency_mode": False,
        "save_traces": False,
    }
    live_config = config.get("live_evaluation_config")
    if live_config is not None:
        for key, default in optional_defaults.items():
            live_config.setdefault(key, default)
    return config


# MainEvaluation imports this function after the compatibility layer is
# installed. Required configuration fields still fail normally; only known
# optional boolean switches receive false defaults.
_utils.load_config_from_path = _load_config_with_optional_bool_defaults

# The Guardian adapter loads the original loop under an alias. Register that
# alias before loading it so the integrated adapter can reuse its implementation.
guardian_spec = importlib.util.spec_from_file_location(
    "EvaluationDebateLoop_guardian", root / "EvaluationDebateLoop-guardian.py"
)
guardian_module = importlib.util.module_from_spec(guardian_spec)
sys.modules[guardian_spec.name] = guardian_module
guardian_spec.loader.exec_module(guardian_module)

integrated_spec = importlib.util.spec_from_file_location(
    "EvaluationDebateLoop", root / "EvaluationDebateLoop-integrated.py"
)
integrated_module = importlib.util.module_from_spec(integrated_spec)
sys.modules["EvaluationDebateLoop"] = integrated_module
sys.modules[integrated_spec.name] = integrated_module
integrated_spec.loader.exec_module(integrated_module)

runpy.run_path(str(root / "MainEvaluation.py"), run_name="__main__")
