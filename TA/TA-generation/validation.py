import json
import re

from schema import REQUIRED_SOURCE_FIELDS, OUTPUT_FIELDS


def validate_source_entry(entry) -> list:
    errors = []
    if not isinstance(entry, dict):
        return ["entry is not a JSON object"]
    for field in REQUIRED_SOURCE_FIELDS:
        if field not in entry:
            errors.append(f"missing required source field: {field}")
    if "User Tool" in entry and not isinstance(entry["User Tool"], str):
        errors.append("'User Tool' must be a string")
    if "Attacker Tools" in entry and not isinstance(entry["Attacker Tools"], list):
        errors.append("'Attacker Tools' must be a list")
    if "Tool Response" in entry and not isinstance(entry["Tool Response"], str):
        errors.append("'Tool Response' must be a string")
    if "User Instruction" in entry and not isinstance(entry["User Instruction"], str):
        errors.append("'User Instruction' must be a string")
    return errors


def validate_dataset(entries) -> list:
    errors = []
    if not isinstance(entries, list):
        return ["dataset is not a JSON array"]
    if len(entries) == 0:
        return ["dataset is empty"]
    for i, entry in enumerate(entries):
        for err in validate_source_entry(entry):
            errors.append(f"entry[{i}]: {err}")
    return errors


def validate_openai_tool_schema(tool) -> str:
    if not isinstance(tool, dict):
        return "tool definition is not an object"
    if tool.get("type") != "function":
        return "tool definition 'type' must be 'function'"
    function = tool.get("function")
    if not isinstance(function, dict):
        return "tool definition missing 'function' object"
    if not isinstance(function.get("name"), str) or not function["name"]:
        return "tool function 'name' must be a non-empty string"
    if not isinstance(function.get("description"), str) or not function["description"]:
        return "tool function 'description' must be a non-empty string"
    parameters = function.get("parameters")
    if parameters is not None and not isinstance(parameters, dict):
        return f"tool '{function['name']}' parameters must be an object"
    return ""


def validate_llm_output(obj) -> str:
    if not isinstance(obj, dict):
        return "output is not a JSON object"
    if "benign_tool_response" not in obj:
        return "missing 'benign_tool_response' key"
    if not isinstance(obj["benign_tool_response"], str) or not obj["benign_tool_response"].strip():
        return "'benign_tool_response' must be a non-empty string"
    if "tools_description" not in obj:
        return "missing 'tools_description' key"
    tools = obj["tools_description"]
    if not isinstance(tools, list) or len(tools) == 0:
        return "'tools_description' must be a non-empty list"
    for i, tool in enumerate(tools):
        err = validate_openai_tool_schema(tool)
        if err:
            return f"tools_description[{i}]: {err}"
    return ""


def parse_json_object(text) -> dict:
    if text is None:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def validate_output_entry(entry) -> list:
    errors = []
    for field in OUTPUT_FIELDS:
        if field not in entry:
            errors.append(f"missing output field: {field}")
    if "Tools description" in entry:
        for i, tool in enumerate(entry["Tools description"]):
            err = validate_openai_tool_schema(tool)
            if err:
                errors.append(f"Tools description[{i}]: {err}")
    return errors
