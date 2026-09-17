"""Realistic anomalous-response generator for the TA (InjecAgent) benchmark.

Reads an already-enriched TA dataset (``TA_dataset.json``) plus the raw
InjecAgent entries it was built from, and rewrites every ``Anomalous tool
response`` so that the attacker's request is embedded as ordinary tool-output
content. Two arms are supported (``operational`` and ``redirect``); see
``prompts/enrichment_realistic.py``.

New files only: the original dataset, configs and prompts are never modified.

Usage:
    python TA/TA-generation/realistic_enrichment.py --config TA/TA-generation/config_operational.yaml
    python TA/TA-generation/realistic_enrichment.py --config ... --limit 20 --out /tmp/pilot.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from llm import build_llm, load_llm_settings  # noqa: E402
from prompts.enrichment_realistic import (  # noqa: E402
    SYSTEM_PROMPT,
    build_user_prompt,
)


FORBIDDEN_PHRASES = [
    "ignore previous",
    "ignore all previous",
    "ignore the previous",
    "system prompt",
    "as an ai",
    "language model",
    "you must",
    "disregard",
    "new instruction",
    "prompt injection",
]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


class RealisticGenerationError(Exception):
    pass


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(os.getcwd(), path)


def extract_json_object(text):
    candidate = str(text or "").strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```$", "", candidate)
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", candidate):
        try:
            obj, _ = decoder.raw_decode(candidate[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def validate_payload(payload, benign_response):
    if not isinstance(payload, dict):
        return None, "payload is not a JSON object"
    anomalous = payload.get("anomalous_tool_response")
    params = payload.get("attack_params", {})
    if not isinstance(anomalous, str) or not anomalous.strip():
        return None, "'anomalous_tool_response' missing or empty"
    if not isinstance(params, dict):
        return None, "'attack_params' is not an object"
    lowered = anomalous.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lowered:
            return None, f"forbidden meta-language: {phrase!r}"
    if anomalous.strip() == str(benign_response).strip():
        return None, "anomalous response equals the benign response"
    return {"anomalous_tool_response": anomalous.strip(), "attack_params": params}, None


def _normalize(value):
    return re.sub(r"\s+", "", str(value or "").lower())


def _flatten_params(params, prefix=""):
    items = []
    if isinstance(params, dict):
        for key, value in params.items():
            items.extend(_flatten_params(value, f"{prefix}{key}_"))
    elif isinstance(params, list):
        for index, value in enumerate(params):
            items.extend(_flatten_params(value, f"{prefix}{index}_"))
    elif params is not None and str(params).strip():
        items.append((prefix.rstrip("_") or "value", str(params).strip()))
    return items


def clean_attack_params(params, attacker_instruction, anomalous, benign):
    """Keep only parameter values that are distinctive to the attack.

    A value is kept when it appears in the attacker instruction (or in the
    attacker's original anomalous response) and does NOT also appear in the
    benign response. This prevents false positives from user-side values such
    as the user's own email address.
    """
    attack_text = _normalize(attacker_instruction) + _normalize(anomalous)
    benign_text = _normalize(benign)
    kept = {}
    dropped = {}
    for key, value in _flatten_params(params):
        value_norm = _normalize(value)
        if len(value) < 4 or value_norm not in attack_text or value_norm in benign_text:
            dropped[key] = value
            continue
        if key in kept:
            key = f"{key}_{len(kept) + 1}"
        kept[key] = value
    kept_values = {_normalize(value) for value in kept.values()}
    for email in EMAIL_RE.findall(str(attacker_instruction)):
        email_norm = _normalize(email)
        if email_norm and email_norm not in benign_text and email_norm not in kept_values:
            key = "recipient" if "recipient" not in kept else f"recipient_{len(kept) + 1}"
            kept[key] = email
            kept_values.add(email_norm)
    return kept, dropped


class RealisticGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.arm = cfg.get("arm", "operational")
        self.max_retries = int(cfg.get("generation", {}).get("max_retries", 3))
        self.concurrency = int(cfg.get("generation", {}).get("concurrency", 100))
        method = cfg.get("llm", {}).get("structured_output_method", "json_mode")
        settings_path = resolve(cfg["llm"]["settings_file"])
        settings = load_llm_settings(settings_path)
        self.llm = build_llm(settings, method)
        self.failures = []

    def generate_one(self, enriched_entry, raw_entry):
        benign = enriched_entry.get("Bening tool response", "")
        original = enriched_entry.get("Anomalous tool response", "")
        context_entry = dict(enriched_entry)
        context_entry["Attacker Tools"] = raw_entry.get("Attacker Tools", [])
        context_entry["Attacker Instruction"] = raw_entry.get("Attacker Instruction", "")
        context_entry["Attack Type"] = raw_entry.get("Attack Type", "")

        last_error = None
        for attempt in range(self.max_retries + 1):
            prompt = build_user_prompt(
                context_entry, benign, original, self.arm,
                retry_note=last_error if attempt else None,
            )
            try:
                response = self.llm.invoke([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ])
                payload = extract_json_object(response.content)
                validated, error = validate_payload(payload, benign)
            except Exception as exc:  # noqa: BLE001
                validated, error = None, str(exc)
            if validated:
                cleaned, dropped = clean_attack_params(
                    validated["attack_params"],
                    context_entry["Attacker Instruction"],
                    validated["anomalous_tool_response"],
                    benign,
                )
                validated["attack_params"] = cleaned
                validated["attack_params_dropped"] = dropped
                return validated
            last_error = error
            if attempt < self.max_retries:
                time.sleep(1.0 * (2 ** attempt))
        raise RealisticGenerationError(
            f"failed to produce a valid realistic response after {self.max_retries + 1} "
            f"attempts; last error: {last_error}"
        )

    def process(self, enriched, raw_by_index, out_path, limit=None):
        entries = enriched[:limit] if limit else enriched
        results = {}
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_index = {}
            for position, entry in enumerate(entries):
                source_index = entry.get("source_entry_index")
                raw_entry = raw_by_index.get(source_index)
                if raw_entry is None:
                    self.failures.append({
                        "source_index": source_index,
                        "error": "raw entry not found",
                    })
                    continue
                future = executor.submit(self.generate_one, entry, raw_entry)
                future_to_index[future] = position

            for future in tqdm(as_completed(future_to_index), total=len(future_to_index),
                               desc=f"Realistic enrichment [{self.arm}]", unit="entry"):
                position = future_to_index[future]
                entry = entries[position]
                source_index = entry.get("source_entry_index")
                raw_entry = raw_by_index.get(source_index, {})
                try:
                    validated = future.result()
                except Exception as exc:  # noqa: BLE001
                    self.failures.append({
                        "source_index": source_index,
                        "error": str(exc),
                    })
                    continue
                output = dict(entry)
                output["Anomalous tool response"] = validated["anomalous_tool_response"]
                output["attack_params"] = validated["attack_params"]
                output["attack_params_dropped"] = validated.get("attack_params_dropped", {})
                output["Attacker Tools"] = raw_entry.get("Attacker Tools", [])
                output["Attacker Instruction"] = raw_entry.get("Attacker Instruction", "")
                output["Attack Type"] = raw_entry.get("Attack Type", "")
                output["realism_arm"] = self.arm
                results[position] = output

        ordered = [results[i] for i in sorted(results)]
        out_path = resolve(out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(ordered, handle, ensure_ascii=False, indent=2)
        failures_path = os.path.splitext(out_path)[0] + "_failures.json"
        if self.failures:
            with open(failures_path, "w", encoding="utf-8") as handle:
                json.dump(self.failures, handle, ensure_ascii=False, indent=2)
        print()
        print("=" * 72)
        print(f"  Realistic enrichment summary (arm={self.arm})")
        print("=" * 72)
        print(f"  entries processed : {len(entries)}")
        print(f"  entries written   : {len(ordered)}")
        print(f"  failures          : {len(self.failures)}")
        print(f"  output            : {out_path}")
        if self.failures:
            print(f"  failures file     : {failures_path}")
        print("=" * 72)
        return ordered


def main():
    parser = argparse.ArgumentParser(description="Realistic anomalous-response generator")
    parser.add_argument("--config", required=True, help="Path to the arm config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N entries")
    parser.add_argument("--out", default=None, help="Override output path")
    parser.add_argument("--input", default=None, help="Override enriched input dataset path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_path = resolve(args.input or cfg["input"]["enriched_dataset"])
    raw_path = resolve(cfg["input"]["raw_dataset"])
    out_path = args.out or cfg["output"]["path"]

    with open(input_path, "r", encoding="utf-8") as handle:
        enriched = json.load(handle)
    with open(raw_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    raw_by_index = {index: entry for index, entry in enumerate(raw)}

    generator = RealisticGenerator(cfg)
    generator.process(enriched, raw_by_index, out_path, limit=args.limit)


if __name__ == "__main__":
    main()
