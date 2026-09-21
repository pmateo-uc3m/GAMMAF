import time
from typing import Any, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
import random
from DatasetManager import TAResponseFormat
from LoggingUtils import log_error


class TAAgent:
    """Tool-call aware debate agent used by InjecAgent-style datasets.

    The first-round flow mirrors ``DebateAgent`` (system prompt + first-round
    prompt as the predefined setup messages), then seeds the conversation with
    the dataset's fake tool interaction as structured messages: an
    ``AIMessage`` carrying the previous tool call followed by a ``ToolMessage``
    carrying the fake tool response. After that, a tool-call round-trip runs:
    whenever the model returns a tool call with empty content, the tool-call
    message is retained, a synthetic successful ``ToolMessage`` is appended,
    and the agent is re-invoked. The final content becomes ``message`` (what is
    passed to neighbours and the defense model), the first called tool becomes
    ``answer``/``called_tool``, every tool emitted at any point of the round is
    recorded in ``called_tools``/``tool_calls``, and ``<tool_call>: ...,
    <message>: ...`` is the trace entry recorded in the debate trace.
    """

    def __init__(self,
                 agent_id: int,
                 model,
                 system_prompt: str,
                 first_round_prompt: str,
                 debate_prompt: str,
                 max_retries: int = 3,
                 is_malicious: bool = False,
                 max_tool_call_follow_ups: int = 3,
                 ):
        self.agent_id = agent_id
        self.model = model
        self.system_prompt = system_prompt
        self.first_round_prompt = first_round_prompt
        self.debate_prompt = debate_prompt
        self.messages: List[BaseMessage] = []
        self.current_response: Optional[Any] = None
        self.max_retries = max_retries
        self.is_malicious = is_malicious
        self.max_tool_call_follow_ups = max_tool_call_follow_ups
        self._tools_bound = False

    def _prepare_format_data(self, format_data: dict) -> dict:
        format_data = dict(format_data)
        if self.is_malicious:
            format_data["tool_response"] = format_data.get("anomalous_response", "")
        else:
            format_data["tool_response"] = format_data.get("benign_response", "")
        return format_data

    def _ensure_tools_bound(self, format_data: dict) -> None:
        tools = format_data.get("tools_description")
        if tools and not self._tools_bound:
            self.model = self.model.bind_tools(tools)
            self._tools_bound = True

    @staticmethod
    def _clean_tool_name(name) -> str:
        """Strip server/parser artifacts (e.g. ``NAME<|CHANNEL|>COMMENTARY``)."""
        text = str(name or "")
        text = text.split("<|", 1)[0]
        return text.strip()

    @staticmethod
    def _serialize_arguments(arguments) -> str:
        if arguments is None:
            return ""
        if isinstance(arguments, str):
            return arguments
        try:
            import json
            return json.dumps(arguments, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(arguments)

    def _extract_tool_calls(self, response) -> list:
        """Normalise every tool call emitted by *response* to plain dicts."""
        calls = []
        for tc in getattr(response, "tool_calls", None) or []:
            if isinstance(tc, dict):
                tool_name = self._clean_tool_name(tc.get("name", ""))
                tool_call_id = tc.get("id", "")
                args = tc.get("args", tc.get("arguments", ""))
            else:
                tool_name = self._clean_tool_name(getattr(tc, "name", ""))
                tool_call_id = getattr(tc, "id", "")
                args = getattr(tc, "args", "")
            calls.append({
                "name": tool_name,
                "arguments": self._serialize_arguments(args),
                "tool_call_id": tool_call_id,
            })
        return calls

    def _invoke_with_tool_round_trip(self):
        response = self.model.invoke(self.messages)
        all_calls = []
        called_tool = ""
        final_appended = False
        while True:
            emitted = self._extract_tool_calls(response)
            content = str(response.content or "").strip()
            if emitted:
                self.messages.append(response)
                final_appended = True
                for call_index, call in enumerate(emitted, start=1):
                    all_calls.append({
                        "name": call["name"],
                        "arguments": call["arguments"],
                    })
                    if not called_tool:
                        called_tool = call["name"]
                    self.messages.append(ToolMessage(
                        content="Tool call executed successfully.",
                        tool_call_id=call["tool_call_id"] or f"call_{len(all_calls)}_{call_index}",
                    ))
            # Tool calls emitted alongside a non-empty message are recorded too,
            # but the text response ends the round (no follow-up invocation).
            if not emitted or content:
                break
            if len(all_calls) >= self.max_tool_call_follow_ups:
                break
            response = self.model.invoke(self.messages)
            final_appended = False
        if not final_appended:
            self.messages.append(response)
        return content, called_tool, all_calls

    def _generate(self, format_data: dict):
        self._ensure_tools_bound(format_data)
        last_exception = None
        for i in range(self.max_retries):
            snapshot = len(self.messages)
            try:
                content, called_tool, all_calls = self._invoke_with_tool_round_trip()
                called_tools = [call["name"] for call in all_calls]
                if called_tools:
                    calls_str = ", ".join(f"<tool_call>: {name}" for name in called_tools)
                    trace = f"{calls_str}, <message>: {content}"
                else:
                    trace = f"<message>: {content}"
                self.current_response = TAResponseFormat(
                    message=content,
                    answer=called_tool,
                    called_tool=called_tool,
                    called_tools=called_tools,
                    trace=trace,
                    tool_calls=all_calls,
                )
                return self.current_response
            except Exception as e:
                last_exception = e
                del self.messages[snapshot:]
                if i == self.max_retries - 1:
                    log_error(f"Agent {self.agent_id} all {self.max_retries} attempts failed: {e}")
                else:
                    time.sleep(1)
        raise last_exception

    def _build_fake_tool_call(self, format_data: dict) -> AIMessage:
        """Build the AIMessage representing the pre-executed tool call.

        The user tool is the first entry of ``available_tools`` (the
        InjecAgent pipeline appends the user tool before attacker tools);
        fall back to the first tool definition in ``tools_description``.
        """
        tool_name = ""
        available_tools = format_data.get("available_tools") or []
        if isinstance(available_tools, list):
            for tool in available_tools:
                if isinstance(tool, str) and tool.strip():
                    tool_name = tool
                    break
        if not tool_name:
            tools_description = format_data.get("tools_description") or []
            if isinstance(tools_description, list):
                for tool in tools_description:
                    if isinstance(tool, dict):
                        function = tool.get("function") or {}
                        name = function.get("name", "")
                        if name:
                            tool_name = name
                            break
        return AIMessage(
            content="",
            tool_calls=[{
                "name": tool_name,
                "args": {},
                "id": f"{random.randint(1000, 9999)}_{self.agent_id}",
            }],
        )

    def first_round_generate(self, format_data: dict):
        format_data = self._prepare_format_data(format_data)
        fake_tool_response = format_data.get("tool_response", "")
        # The fake tool result is carried structurally by the ToolMessage
        # appended below; render the prompt without a duplicated plain-text
        # copy of it.
        prompt_format_data = dict(format_data)
        prompt_format_data["tool_response"] = ""
        sys_prompt = self.system_prompt.format(**format_data)
        prompt = self.first_round_prompt.format(**prompt_format_data)
        self.messages.append(SystemMessage(content=sys_prompt))
        self.messages.append(HumanMessage(content=prompt))
        fake_tool_call = self._build_fake_tool_call(format_data)
        self.messages.append(fake_tool_call)
        self.messages.append(ToolMessage(
            content=fake_tool_response,
            tool_call_id=fake_tool_call.tool_calls[0]["id"],
        ))
        return self._generate(format_data)

    def debate_round_generate(self, format_data: dict):
        format_data = self._prepare_format_data(format_data)
        prompt = self.debate_prompt.format(**format_data)
        self.messages.append(HumanMessage(content=prompt))
        return self._generate(format_data)

    def get_current_response(self) -> Optional[Any]:
        return self.current_response

    def get_formatted_response(self) -> Optional[str]:
        if self.current_response:
            return f"Agent {self.agent_id} Response: \n{self.current_response.to_message_content()}"
        return None


# Backwards-compatible alias: this file previously defined ``DebateAgent``.
DebateAgent = TAAgent