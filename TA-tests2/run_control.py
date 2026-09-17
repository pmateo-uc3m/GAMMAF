"""Run one no-injection control for a TA2 arm/experiment.

Uses the same prompt and seeds as the matching real experiment, with the
control dataset (benign response in place of the anomalous one).

Usage:
    python TA-tests2/run_control.py A 01
"""

import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from run_all import PYTHON, ROOT, set_prompts_file  # noqa: E402


def main():
    arm = sys.argv[1]
    exp = int(sys.argv[2])
    prompt_rel = f"prompts/TA2-test-prompts/TA2-test-{exp:02d}.json"
    config_rel = f"config-examples/generation-config-TA2-{arm}-control-{exp:02d}.yaml"
    out_dir = os.path.join(ROOT, "TA-tests2", f"arm-{arm}", f"exp-{exp:02d}-control")
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, "TA-att.pkl")
    if os.path.exists(pkl_path):
        print(f"[skip] control arm {arm} exp {exp:02d} already exists")
        return
    if not os.path.exists(os.path.join(ROOT, config_rel)):
        raise FileNotFoundError(config_rel)

    set_prompts_file(prompt_rel)
    print(f"[run ] control arm {arm} exp {exp:02d} with {prompt_rel}")
    with open(os.path.join(out_dir, "generation.log"), "w", encoding="utf-8") as log:
        subprocess.run([PYTHON, "TrainDataGeneration.py", config_rel],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if not os.path.exists(pkl_path):
        print(f"[fail] control arm {arm} exp {exp:02d}: no pkl")
        return
    with open(os.path.join(out_dir, "asr.log"), "w", encoding="utf-8") as log:
        subprocess.run([PYTHON, "auxiliary/compute_ta_asr.py", pkl_path,
                        "--json-out", os.path.join(out_dir, "metrics.json")],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    print(f"[done] control arm {arm} exp {exp:02d}")


if __name__ == "__main__":
    main()
