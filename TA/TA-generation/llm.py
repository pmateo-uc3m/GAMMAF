import yaml

from langchain_openai import ChatOpenAI


def load_llm_settings(settings_file):
    with open(settings_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_llm(settings, structured_output_method="json_mode"):
    model_kwargs = dict(settings.get("model_kwargs") or {})
    if structured_output_method == "json_mode":
        model_kwargs.setdefault("response_format", {"type": "json_object"})
    return ChatOpenAI(
        base_url=settings["base_url"],
        api_key=settings.get("api_key", "EMPTY"),
        model=settings["model"],
        temperature=settings.get("temperature", 0.0),
        max_tokens=settings.get("max_tokens", 2048),
        timeout=settings.get("timeout", 120),
        model_kwargs=model_kwargs,
    )
