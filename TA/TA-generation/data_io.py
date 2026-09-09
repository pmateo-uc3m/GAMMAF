import glob
import json
import os

from validation import validate_dataset


def _load_json_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _load_jsonl_file(filepath):
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _candidate_files(path, explicit_files):
    if explicit_files:
        return list(explicit_files)
    if os.path.isfile(path):
        return [path]
    patterns = [
        os.path.join(path, "*test_cases*.json"),
        os.path.join(path, "*test_cases*.jsonl"),
        os.path.join(path, "*.json"),
        os.path.join(path, "*.jsonl"),
    ]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))
    return sorted(set(found))


def load_source_dataset(input_cfg):
    path = input_cfg["path"]
    explicit_files = input_cfg.get("explicit_files") or []
    files = _candidate_files(path, explicit_files)
    if not files:
        raise FileNotFoundError(
            f"No candidate input files found. Check 'input.path' ({path!r}) or "
            "set 'input.explicit_files'."
        )

    last_error = None
    for filepath in files:
        try:
            if filepath.endswith(".jsonl"):
                entries = _load_jsonl_file(filepath)
            else:
                entries = _load_json_file(filepath)
        except Exception as e:
            last_error = f"{filepath}: could not parse ({e})"
            continue
        if not isinstance(entries, list):
            last_error = f"{filepath}: not a JSON array"
            continue
        errors = validate_dataset(entries)
        if not errors:
            return entries, filepath
        last_error = f"{filepath}: schema mismatch ({errors[0]})"

    raise ValueError(
        "No usable InjecAgent test-case dataset found among candidate files. "
        f"Last error: {last_error}"
    )
