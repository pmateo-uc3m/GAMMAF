import json
import logging
import os
import random

import yaml

from data_io import load_source_dataset
from llm import build_llm, load_llm_settings
from prompts.enrichment import SYSTEM_PROMPT, build_user_prompt
from schema import (
    F_ANOMALOUS_RESPONSE,
    F_AVAILABLE_TOOLS,
    F_BENIGN_RESPONSE,
    F_SOURCE_FILE,
    F_SOURCE_INDEX,
    F_TOOLS_DESCRIPTION,
    F_USER_INSTRUCTION,
    build_output_entry,
)
from validation import parse_json_object, validate_llm_output, validate_output_entry


class GenerationError(Exception):
    pass


def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_entries(dataset, num_entries, random_seed):
    total = len(dataset)
    if num_entries > total:
        raise ValueError(
            f"num_entries ({num_entries}) exceeds the number of available source "
            f"entries ({total}). Reduce num_entries or provide more data."
        )
    rng = random.Random(random_seed)
    indices = rng.sample(range(total), num_entries)
    return sorted(indices)


def extract_available_tools(entry):
    user_tool = entry.get("User Tool")
    attacker_tools = entry.get("Attacker Tools") or []
    tools = []
    if user_tool:
        tools.append(user_tool)
    for tool in attacker_tools:
        if tool not in tools:
            tools.append(tool)
    return tools


class EnrichmentGenerator:
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger("TA-generation")
        llm_settings_file = cfg["llm"]["settings_file"]
        llm_settings = load_llm_settings(llm_settings_file)
        self.structured_method = cfg["llm"].get("structured_output_method", "json_mode")
        self.llm = build_llm(llm_settings, self.structured_method)
        self.max_retries = cfg["generation"].get("max_retries", 3)
        self.failures = []

    def _call_llm(self, user_prompt):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm.invoke(messages)
        return response.content

    def generate_one(self, source_entry):
        available_tools = extract_available_tools(source_entry)
        prompt = build_user_prompt(source_entry, available_tools)
        last_error = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                self.logger.info("retry attempt %d after: %s", attempt, last_error)
                prompt = build_user_prompt(
                    source_entry, available_tools, retry_note=last_error
                )
            text = self._call_llm(prompt)
            obj = parse_json_object(text)
            if obj is None:
                last_error = "the model did not return valid JSON"
                continue
            err = validate_llm_output(obj)
            if err is None:
                return obj, available_tools, attempt
            last_error = err
        raise GenerationError(
            f"failed to produce valid enrichment after {self.max_retries + 1} "
            f"attempts; last error: {last_error}"
        )

    def process(self, dataset, indices, source_file, output_path):
        entries = []
        total = len(indices)
        for position, idx in enumerate(indices, start=1):
            source_entry = dataset[idx]
            self.logger.info(
                "processing entry %d/%d (source index %d)", position, total, idx
            )
            try:
                result, available_tools, attempts = self.generate_one(source_entry)
                benign = result["benign_tool_response"]
                tools_desc = result["tools_description"]
                out_entry = build_output_entry(
                    source_index=idx,
                    source_file=source_file,
                    available_tools=available_tools,
                    user_instruction=source_entry.get("User Instruction", ""),
                    anomalous_response=source_entry.get("Tool Response", ""),
                    benign_response=benign,
                    tools_description=tools_desc,
                )
                entry_errors = validate_output_entry(out_entry)
                if entry_errors:
                    raise GenerationError("; ".join(entry_errors))
                entries.append(out_entry)
                self.logger.info(
                    "succeeded entry %d/%d (source index %d) after %d attempt(s)",
                    position, total, idx, attempts + 1,
                )
            except Exception as e:
                self.failures.append({"source_index": idx, "error": str(e)})
                self.logger.error(
                    "failed entry %d/%d (source index %d): %s",
                    position, total, idx, e,
                )
                if self.cfg["validation"].get("require_all_entries_success"):
                    raise
        self._write_output(output_path, entries)
        return entries

    def _write_output(self, output_path, entries):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        self.logger.info("wrote %d complete entries to %s", len(entries), output_path)


def write_failures_report(failures, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)
