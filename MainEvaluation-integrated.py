"""Integrated evaluation entry point for legacy and temporal models."""

import importlib.util
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parent

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
