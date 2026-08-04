"""Evaluation loop adapter supporting legacy and temporal defense models.

Temporal models may expose ``predict(round, adjacency, temporal_rounds=...)``;
legacy models continue to receive the original two arguments.
"""

import importlib.util
import inspect
import sys
from pathlib import Path

if "EvaluationDebateLoop_guardian" not in sys.modules:
    _path = Path(__file__).resolve().with_name("EvaluationDebateLoop-guardian.py")
    _spec = importlib.util.spec_from_file_location("EvaluationDebateLoop_guardian", _path)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _module
    _spec.loader.exec_module(_module)
else:
    _module = sys.modules["EvaluationDebateLoop_guardian"]

_GuardianOrchestration = _module.LiveDebateOrchestration


class _PredictCompatibilityAdapter:
    """Keep the existing model object while adapting only predict dispatch."""

    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __setattr__(self, name, value):
        if name == "_model":
            object.__setattr__(self, name, value)
        else:
            setattr(self._model, name, value)

    def predict(self, round_data, adjacency, temporal_rounds=None):
        predict = self._model.predict
        try:
            parameters = inspect.signature(predict).parameters.values()
            supports_temporal = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters)
            supports_temporal = supports_temporal or "temporal_rounds" in inspect.signature(predict).parameters
        except (TypeError, ValueError):
            supports_temporal = False
        if supports_temporal:
            return predict(round_data, adjacency, temporal_rounds=temporal_rounds)
        return predict(round_data, adjacency)


class LiveDebateOrchestration(_GuardianOrchestration):
    """Use Guardian temporal execution while retaining legacy model support."""

    def debate_question(self, defense_model, *args, **kwargs):
        return super().debate_question(_PredictCompatibilityAdapter(defense_model), *args, **kwargs)


# MainEvaluation imports these symbols from the module under the standard name.
__all__ = ["LiveDebateOrchestration"]
