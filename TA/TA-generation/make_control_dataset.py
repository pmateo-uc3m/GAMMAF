"""Build a no-injection control dataset from a realistic TA dataset.

The control keeps the same tools, user instructions, benign responses and
attack metadata, but replaces every anomalous tool response with the benign
one. Running the same prompts on the control measures the spurious tool-call
base rate (agents calling the attack tool without any injected content), which
is subtracted from the real run's metrics.

Usage:
    python TA/TA-generation/make_control_dataset.py \
        --input TA/TA-generation/output/TA_dataset_operational.json \
        --out TA/TA-generation/output/TA_dataset_operational_control.json
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Build a no-injection control dataset")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        entries = json.load(handle)

    control = []
    for entry in entries:
        item = dict(entry)
        item["Anomalous tool response"] = entry.get("Bening tool response", "")
        item["control"] = True
        item["control_of"] = os.path.basename(args.input)
        control.append(item)

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.getcwd(), out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(control, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {len(control)} control entries to {out_path}")


if __name__ == "__main__":
    main()
