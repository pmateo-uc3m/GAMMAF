"""Guardian entry point using the isolated temporal evaluation adapter."""

import importlib.util
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("EvaluationDebateLoop", root / "EvaluationDebateLoop-guardian.py")
module = importlib.util.module_from_spec(spec)
sys.modules["EvaluationDebateLoop"] = module
spec.loader.exec_module(module)
runpy.run_path(str(root / "MainEvaluation.py"), run_name="__main__")
