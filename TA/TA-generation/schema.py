from typing import Any, Dict

OUTPUT_FIELDS = [
    "AvailableTools",
    "User Instruction",
    "Anomalous tool response",
    "Bening tool response",
    "Tools description",
]

REQUIRED_SOURCE_FIELDS = [
    "User Tool",
    "User Instruction",
    "Attacker Tools",
    "Tool Response",
]

F_AVAILABLE_TOOLS = "AvailableTools"
F_USER_INSTRUCTION = "User Instruction"
F_ANOMALOUS_RESPONSE = "Anomalous tool response"
F_BENIGN_RESPONSE = "Bening tool response"
F_TOOLS_DESCRIPTION = "Tools description"
F_SOURCE_INDEX = "source_entry_index"
F_SOURCE_FILE = "source_file"


def build_output_entry(
    source_index: int,
    source_file: str,
    available_tools: list,
    user_instruction: str,
    anomalous_response: str,
    benign_response: str,
    tools_description: list,
) -> Dict[str, Any]:
    return {
        F_AVAILABLE_TOOLS: list(available_tools),
        F_USER_INSTRUCTION: user_instruction,
        F_ANOMALOUS_RESPONSE: anomalous_response,
        F_BENIGN_RESPONSE: benign_response,
        F_TOOLS_DESCRIPTION: tools_description,
        F_SOURCE_INDEX: int(source_index),
        F_SOURCE_FILE: source_file,
    }
