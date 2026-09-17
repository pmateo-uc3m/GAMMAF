"""Sequential driver for the TA2 experiments.

For each (arm, experiment) it:
1. points ``InjecAgentLoader.PROMPTS_FILE`` at the arm's prompt variant,
2. runs ``TrainDataGeneration.py`` with the matching YAML config,
3. evaluates the generated pkl with ``auxiliary/compute_ta_asr.py``.

Existing pkl files are skipped, so the driver can be re-run to append work.

Usage:
    python TA-tests2/run_all.py A            # arm A, experiments 01..14
    python TA-tests2/run_all.py B 01 08      # arm B, experiments 01..08
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = "/project_antwerp/gammaf-env/bin/python"
DATASET_MANAGER = os.path.join(ROOT, "DatasetManager.py")

PROMPTS_PATTERN = re.compile(
    r'(class InjecAgentLoader\(MMLULoader\):[\s\S]*?PROMPTS_FILE = ")([^"]*)(")'
)


def set_prompts_file(relative_path):
    with open(DATASET_MANAGER, "r", encoding="utf-8") as handle:
        source = handle.read()
    replaced, count = PROMPTS_PATTERN.subn(
        lambda match: match.group(1) + relative_path + match.group(3), source, count=1
    )
    if count != 1:
        raise RuntimeError("Could not locate InjecAgentLoader.PROMPTS_FILE")
    with open(DATASET_MANAGER, "w", encoding="utf-8") as handle:
        handle.write(replaced)


def write_metadata(exp_dir, arm, exp, prompt_path, dataset_path):
    meta = {
        "phase": "TA2",
        "arm": arm,
        "experiment": exp,
        "prompt_file": prompt_path,
        "dataset_path": dataset_path,
        "yaml_config": f"config-examples/generation-config-TA2-{arm}-test-{exp:02d}.yaml",
        "pkl_path": f"TA-tests2/arm-{arm}/exp-{exp:02d}/TA-att.pkl",
        "seeds": {
            "questions_random_seed": 43522,
            "malicious_randomization_seed": (3600 if arm == "A" else 3800) + exp,
            "random_topo_seed": (5600 if arm == "A" else 5800) + exp,
        },
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(exp_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def run_experiment(arm, exp):
    exp_dir = os.path.join(ROOT, "TA-tests2", f"arm-{arm}", f"exp-{exp:02d}")
    os.makedirs(exp_dir, exist_ok=True)
    pkl_path = os.path.join(exp_dir, "TA-att.pkl")
    if os.path.exists(pkl_path):
        print(f"[skip] arm {arm} exp {exp:02d} already has a pkl")
        return

    prompt_rel = f"prompts/TA2-test-prompts/TA2-test-{exp:02d}.json"
    config_rel = f"config-examples/generation-config-TA2-{arm}-test-{exp:02d}.yaml"
    dataset_rel = (
        "TA/TA-generation/output/TA_dataset_operational.json"
        if arm == "A"
        else "TA/TA-generation/output/TA_dataset_redirect.json"
    )
    if not os.path.exists(os.path.join(ROOT, dataset_rel)):
        raise FileNotFoundError(f"Dataset for arm {arm} not found: {dataset_rel}")

    set_prompts_file(prompt_rel)
    write_metadata(exp_dir, arm, exp, prompt_rel, dataset_rel)
    print(f"[run ] arm {arm} exp {exp:02d}: {prompt_rel}")

    with open(os.path.join(exp_dir, "generation.log"), "w", encoding="utf-8") as log:
        subprocess.run(
            [PYTHON, "TrainDataGeneration.py", config_rel],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    if not os.path.exists(pkl_path):
        print(f"[fail] arm {arm} exp {exp:02d}: no pkl produced")
        return

    with open(os.path.join(exp_dir, "asr.log"), "w", encoding="utf-8") as log:
        subprocess.run(
            [PYTHON, "auxiliary/compute_ta_asr.py", pkl_path,
             "--json-out", os.path.join(exp_dir, "metrics.json")],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    print(f"[done] arm {arm} exp {exp:02d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["A", "B"])
    parser.add_argument("first", type=int, nargs="?", default=1)
    parser.add_argument("last", type=int, nargs="?", default=14)
    args = parser.parse_args()

    for exp in range(args.first, args.last + 1):
        run_experiment(args.arm, exp)


if __name__ == "__main__":
    main()
