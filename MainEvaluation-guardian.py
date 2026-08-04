"""Guardian entry point using the isolated temporal evaluation adapter."""

import importlib.util
import pickle
import runpy
import sys
from pathlib import Path

import Utils as _utils
from ConfigCheck import validate_evaluation_config

root = Path(__file__).resolve().parent

if len(sys.argv) < 2:
    raise ValueError("Usage: python MainEvaluation-guardian.py <config_file> [--clean]")
validate_evaluation_config(sys.argv[1])
_load_config_from_path = _utils.load_config_from_path


def _load_config_with_optional_bool_defaults(config_path):
    config = _load_config_from_path(config_path)
    defaults = {
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
        for key, default in defaults.items():
            live_config.setdefault(key, default)
    return config


_utils.load_config_from_path = _load_config_with_optional_bool_defaults


class _LegacyPickleList(list):
    def get(self, key, default=None):
        return default


_pickle_load = pickle.load


def _load_pickle_compatibility(file_obj, *args, **kwargs):
    value = _pickle_load(file_obj, *args, **kwargs)
    return _LegacyPickleList(value) if isinstance(value, list) else value


pickle.load = _load_pickle_compatibility
spec = importlib.util.spec_from_file_location("EvaluationDebateLoop", root / "EvaluationDebateLoop-guardian.py")
module = importlib.util.module_from_spec(spec)
sys.modules["EvaluationDebateLoop"] = module
spec.loader.exec_module(module)
runpy.run_path(str(root / "MainEvaluation.py"), run_name="__main__")
