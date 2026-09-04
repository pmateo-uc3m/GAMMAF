"""OpenAI-compatible LLM client for contamination generation.

Connects to any OpenAI-compatible endpoint (local or remote) using settings
from ``llm_settings.json``.  Parameters such as ``host_url``, ``model_name``,
``temperature``, ``top_p``, ``max_tokens``, and ``json_mode`` are read from
that file and are never hard-coded in Python.

The client:

* sends the system + user prompts,
* requests structured/JSON output where supported,
* parses and validates the returned JSON,
* retries (bounded) on connection/API/parse/validation errors.

The validated structure must contain a single ``incorrect_answer`` and exactly
``n`` ``adv_passages`` in the same order as the source passages.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple


class LLMGenerationError(Exception):
    """Raised when contamination generation fails after all retries."""


class JSONValidationError(Exception):
    """Raised when an LLM response fails structural validation."""


def _strip_fences(text: str) -> str:
    """Remove markdown code fences and leading/trailing whitespace."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from a string (handles stray text)."""
    candidate = _strip_fences(text)
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", candidate):
        start = match.start()
        try:
            obj, _ = decoder.raw_decode(candidate[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    raise JSONValidationError("could not locate a JSON object in the response")


def validate_contamination(
    payload: Any,
    n: int,
) -> Dict[str, Any]:
    """Validate the LLM payload against the required benchmark structure.

    Returns the validated payload (with a str target answer and list of str
    passages) or raises :class:`JSONValidationError`.
    """
    if not isinstance(payload, dict):
        raise JSONValidationError("payload is not a JSON object")

    incorrect_answer = payload.get("incorrect_answer")
    adv = payload.get("adv_passages")

    if incorrect_answer is None or not isinstance(incorrect_answer, str):
        raise JSONValidationError("'incorrect_answer' missing or not a string")
    if not incorrect_answer.strip():
        raise JSONValidationError("'incorrect_answer' is empty")

    if adv is None or not isinstance(adv, list):
        raise JSONValidationError("'adv_passages' missing or not a list")
    if len(adv) != n:
        raise JSONValidationError(
            f"'adv_passages' has {len(adv)} entries, expected {n}"
        )

    cleaned: List[str] = []
    for i, p in enumerate(adv):
        if not isinstance(p, str):
            raise JSONValidationError(
                f"'adv_passages[{i}]' is not a string"
            )
        if not p.strip():
            raise JSONValidationError(
                f"'adv_passages[{i}]' is empty"
            )
        cleaned.append(p)

    return {
        "incorrect_answer": incorrect_answer.strip(),
        "adv_passages": cleaned,
    }


def validate_answer(payload: Any) -> Dict[str, Any]:
    """Validate an LLM answer-generation payload.

    Returns ``{"answer": str}`` or raises :class:`JSONValidationError`.
    """
    if not isinstance(payload, dict):
        raise JSONValidationError("answer payload is not a JSON object")

    answer = payload.get("answer")
    if answer is None or not isinstance(answer, str):
        raise JSONValidationError("'answer' missing or not a string")
    if not answer.strip():
        raise JSONValidationError("'answer' is empty")

    return {"answer": answer.strip()}


class ContaminationLLM:
    """Thin wrapper around the OpenAI-compatible client with JSON handling."""

    def __init__(self, settings: Dict[str, Any]):
        self.host_url = settings.get("host_url")
        self.api_key = settings.get("api_key", "")
        self.model_name = settings.get("model_name")
        self.timeout = settings.get("timeout", 120)
        self.temperature = settings.get("temperature", 0.7)
        self.top_p = settings.get("top_p", 1.0)
        self.max_tokens = settings.get("max_tokens", 4096)
        self.json_mode = bool(settings.get("json_mode", True))
        self.max_retries = int(settings.get("max_retries", 3))

        if not self.host_url:
            raise ValueError("llm_settings.json must define 'host_url'")
        if not self.model_name:
            raise ValueError("llm_settings.json must define 'model_name'")

        from openai import OpenAI

        self.client = OpenAI(
            base_url=self.host_url,
            api_key=self.api_key or "not-needed",
            timeout=self.timeout,
        )

    def _extract_content(self, completion) -> str:
        choice = completion.choices[0]
        message = choice.message
        content = message.content
        if content is None:
            # Some servers return structured output only in tool/json fields.
            raise JSONValidationError("model returned empty content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
                else:
                    parts.append(str(block))
            return "".join(parts)
        return str(content)

    def _request_json(
        self,
        system_prompt: str,
        user_prompt: str,
        retry_settings: Optional[Dict[str, Any]],
        what: str,
    ) -> str:
        """Send one prompt pair, retrying on failure, return raw content.

        Raises :class:`LLMGenerationError` after all attempts fail.
        """
        rs = retry_settings or {}
        max_attempts = int(
            rs.get("max_llm_attempts", self.max_retries)
        )
        delay = float(rs.get("retry_delay_seconds", 1.0))
        backoff = float(rs.get("backoff_factor", 2.0))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Optional[Exception] = None
        current_delay = delay

        for attempt in range(1, max_attempts + 1):
            try:
                completion = self.client.chat.completions.create(**kwargs)
                return self._extract_content(completion)
            except JSONValidationError as exc:
                last_error = exc
            except Exception as exc:  # connection/API errors
                last_error = exc

            if attempt < max_attempts:
                time.sleep(current_delay)
                current_delay *= backoff

        raise LLMGenerationError(
            f"{what} failed after {max_attempts} attempts: {last_error}"
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        n: int,
        retry_settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate and validate contaminated passages with bounded retries.

        ``retry_settings`` may contain ``max_llm_attempts``,
        ``retry_delay_seconds``, and ``backoff_factor``.
        """
        content = self._request_json(
            system_prompt, user_prompt, retry_settings,
            "contamination generation",
        )
        payload = _extract_json_object(content)
        return validate_contamination(payload, n)

    def generate_answer(
        self,
        system_prompt: str,
        user_prompt: str,
        retry_settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate and validate an answer from safe passages.

        Returns ``{"answer": str}``.  Retries (bounded) on failure, raising
        :class:`LLMGenerationError` if all attempts fail.
        """
        content = self._request_json(
            system_prompt, user_prompt, retry_settings,
            "answer generation",
        )
        payload = _extract_json_object(content)
        return validate_answer(payload)
