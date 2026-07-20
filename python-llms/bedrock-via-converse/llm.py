import json
import botocore.exceptions
from dataiku.llm.python import BaseLLM
from custombedrock import (
    get_boto3_session,
    convert_messages,
    convert_tools,
    extract_tool_calls,
)

# Pricing per 1k tokens (us-east-1 rates; 0.0 = region-variable or unknown)
_PRICING = {
    # Amazon
    "amazon.nova-pro-v1:0":                        {"prompt": 0.0008,    "completion": 0.0032},
    "amazon.nova-lite-v1:0":                       {"prompt": 0.00006,   "completion": 0.00024},
    "amazon.nova-micro-v1:0":                      {"prompt": 0.000035,  "completion": 0.00014},
    "amazon.titan-text-premier-v1:0":              {"prompt": 0.0005,    "completion": 0.0015},
    "amazon.titan-text-lite-v1":                   {"prompt": 0.00015,   "completion": 0.0002},
    "amazon.titan-text-express-v1":                {"prompt": 0.0002,    "completion": 0.0006},
    "amazon.titan-tg1-large":                      {"prompt": 0.0013,    "completion": 0.0017},
    # Anthropic
    "anthropic.claude-sonnet-4-5-20251015-v1:0":   {"prompt": 0.003,     "completion": 0.015},
    "anthropic.claude-haiku-4-5-20251001-v1:0":    {"prompt": 0.001,     "completion": 0.005},
    "anthropic.claude-opus-4-5-20251015-v1:0":     {"prompt": 0.005,     "completion": 0.025},
    "anthropic.claude-opus-4-1-20250815-v1:0":     {"prompt": 0.015,     "completion": 0.075},
    "anthropic.claude-sonnet-4-0-20250514-v1:0":   {"prompt": 0.003,     "completion": 0.015},
    "anthropic.claude-opus-4-0-20250514-v1:0":     {"prompt": 0.015,     "completion": 0.075},
    "anthropic.claude-3-7-sonnet-20250219-v1:0":   {"prompt": 0.003,     "completion": 0.015},
    "anthropic.claude-3-5-sonnet-20241022-v2:0":   {"prompt": 0.003,     "completion": 0.015},
    "anthropic.claude-3-5-sonnet-20240620-v1:0":   {"prompt": 0.003,     "completion": 0.015},
    "anthropic.claude-3-5-haiku-20241022-v1:0":    {"prompt": 0.0008,    "completion": 0.004},
    "anthropic.claude-3-sonnet-20240229-v1:0":     {"prompt": 0.003,     "completion": 0.015},
    "anthropic.claude-3-haiku-20240307-v1:0":      {"prompt": 0.00025,   "completion": 0.00125},
    "anthropic.claude-3-opus-20240229-v1:0":       {"prompt": 0.015,     "completion": 0.075},
    # AI21 Labs (same rate for prompt and completion)
    "ai21.j2-ultra-v1":                            {"prompt": 0.0188,    "completion": 0.0188},
    "ai21.j2-mid-v1":                              {"prompt": 0.0125,    "completion": 0.0125},
    # Cohere
    "cohere.command-r-plus-v1:0":                  {"prompt": 0.003,     "completion": 0.015},
    "cohere.command-r-v1:0":                       {"prompt": 0.0005,    "completion": 0.0015},
    # Meta
    "meta.llama3-3-70b-instruct-v1:0":             {"prompt": 0.00072,   "completion": 0.00072},
    "meta.llama3-1-8b-instruct-v1:0":              {"prompt": 0.00022,   "completion": 0.00022},
    "meta.llama3-1-70b-instruct-v1:0":             {"prompt": 0.00072,   "completion": 0.00072},
    "meta.llama3-1-405b-instruct-v1:0":            {"prompt": 0.00532,   "completion": 0.016},
    "meta.llama3-8b-instruct-v1:0":                {"prompt": 0.0003,    "completion": 0.0006},
    "meta.llama3-70b-instruct-v1:0":               {"prompt": 0.00265,   "completion": 0.0035},
    # Mistral (0.0 = pricing varies by region)
    "mistral.mistral-7b-instruct-v0:2":            {"prompt": 0.0,       "completion": 0.0},
    "mistral.mixtral-8x7b-instruct-v0:1":          {"prompt": 0.0,       "completion": 0.0},
    "mistral.mistral-small-2402-v1:0":             {"prompt": 0.0,       "completion": 0.0},
    "mistral.mistral-large-2402-v1:0":             {"prompt": 0.0,       "completion": 0.0},
    "mistral.mistral-large-2407-v1:0":             {"prompt": 0.003,     "completion": 0.009},
    # DeepSeek
    "deepseek.r1-v1:0":                            {"prompt": 0.00135,   "completion": 0.0054},
}

# Sentinel values used as visual separators in the SELECT — never valid model IDs
_SEPARATOR_VALUES = {"__amazon__", "__anthropic__", "__ai21__", "__cohere__", "__meta__", "__mistral__", "__deepseek__"}

# Maps inference profile name -> model ID prefix
_PROFILE_PREFIX = {
    "us":   "us",
    "eu":   "eu",
    "apac": "ap",
}


class MyLLM(BaseLLM):
    def __init__(self):
        self.client = None
        self.model_id = None
        self.guardrail_id = None
        self.guardrail_version = "DRAFT"
        self.max_parallelism_val = 8
        self._pricing = {"prompt": 0.0, "completion": 0.0}

    # -------------------------------------------------------------------------

    def set_config(self, config: dict, plugin_config: dict) -> None:
        region = (config.get("region") or "us-east-1").strip()
        s3_connection = (config.get("s3Connection") or "").strip()
        inference_profile = (config.get("inferenceProfile") or "").strip()
        use_guardrail = bool(config.get("useGuardrail", False))
        self.guardrail_id = (config.get("guardrailIdentifier") or "").strip() if use_guardrail else ""
        self.guardrail_version = (config.get("guardrailVersion") or "DRAFT").strip()
        self.max_parallelism_val = int(config.get("maxParallelism") or 8)

        base_model_id = config["modelId"]
        if base_model_id in _SEPARATOR_VALUES:
            raise ValueError(f"'{base_model_id}' is a section separator, not a model. Please select an actual model.")
        self._pricing = _PRICING.get(base_model_id, {"prompt": 0.0, "completion": 0.0})

        if inference_profile:
            prefix = _PROFILE_PREFIX.get(inference_profile.lower(), inference_profile.lower())
            self.model_id = f"{prefix}.{base_model_id}"
        else:
            self.model_id = base_model_id

        session = get_boto3_session(s3_connection, region)
        self.client = session.client("bedrock-runtime", region_name=region)

    def get_max_parallelism(self) -> int:
        return self.max_parallelism_val

    # -------------------------------------------------------------------------

    def _build_request(self, query: dict, settings: dict) -> dict:
        bedrock_messages, bedrock_system = convert_messages(query["messages"])

        req = {
            "modelId": self.model_id,
            "messages": bedrock_messages,
        }

        if bedrock_system:
            req["system"] = bedrock_system

        inf_cfg = _build_inference_config(settings)
        if inf_cfg:
            req["inferenceConfig"] = inf_cfg

        if self.guardrail_id:
            req["guardrailConfig"] = {
                "guardrailIdentifier": self.guardrail_id,
                "guardrailVersion": self.guardrail_version,
            }

        tools = query.get("tools") or []
        if tools:
            req["toolConfig"] = {"tools": convert_tools(tools)}

        return req

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            (prompt_tokens / 1000.0) * self._pricing["prompt"]
            + (completion_tokens / 1000.0) * self._pricing["completion"]
        )

    # -------------------------------------------------------------------------

    def process(self, query, settings, trace):
        req = self._build_request(query, settings)

        try:
            resp = self.client.converse(**req)
        except botocore.exceptions.ClientError as e:
            raise RuntimeError(f"Bedrock Converse error: {e}") from e

        output_content = resp["output"]["message"].get("content", [])
        usage = resp.get("usage", {})
        prompt_tokens = usage.get("inputTokens", 0)
        completion_tokens = usage.get("outputTokens", 0)

        text = "".join(b["text"] for b in output_content if "text" in b)
        tool_calls = extract_tool_calls(output_content)

        return {
            "text": text,
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "estimatedCost": self._compute_cost(prompt_tokens, completion_tokens),
            "toolCalls": tool_calls,
        }

    # -------------------------------------------------------------------------

    def process_stream(self, query, settings, trace):
        req = self._build_request(query, settings)

        try:
            resp = self.client.converse_stream(**req)
        except botocore.exceptions.ClientError as e:
            raise RuntimeError(f"Bedrock ConverseStream error: {e}") from e

        prompt_tokens = 0
        completion_tokens = 0

        # Tool call accumulation: contentBlockIndex -> {id, name, input_parts[]}
        open_tools = {}

        try:
            for event in resp["stream"]:
                if "contentBlockStart" in event:
                    cbs = event["contentBlockStart"]
                    idx = cbs["contentBlockIndex"]
                    start = cbs.get("start", {})
                    if "toolUse" in start:
                        open_tools[idx] = {
                            "id": start["toolUse"]["toolUseId"],
                            "name": start["toolUse"]["name"],
                            "input_parts": [],
                        }

                elif "contentBlockDelta" in event:
                    cbd = event["contentBlockDelta"]
                    idx = cbd["contentBlockIndex"]
                    delta = cbd.get("delta", {})

                    if "text" in delta:
                        yield {"chunk": {"text": delta["text"]}}
                    elif "toolUse" in delta:
                        # Partial JSON fragment — accumulate, parse later
                        if idx in open_tools:
                            open_tools[idx]["input_parts"].append(
                                delta["toolUse"].get("input", "")
                            )

                elif "metadata" in event:
                    usage = event["metadata"].get("usage", {})
                    prompt_tokens = usage.get("inputTokens", 0)
                    completion_tokens = usage.get("outputTokens", 0)

        except botocore.exceptions.EventStreamError as e:
            raise RuntimeError(f"Bedrock stream interrupted: {e}") from e

        # Reconstruct tool calls from accumulated JSON fragments
        tool_calls = []
        for tc in open_tools.values():
            raw_args = "".join(tc["input_parts"])
            tool_calls.append({
                "type": "function",
                "id": tc["id"],
                "function": {
                    "name": tc["name"],
                    "arguments": raw_args,
                },
            })

        yield {
            "footer": {
                "promptTokens": prompt_tokens,
                "completionTokens": completion_tokens,
                "estimatedCost": self._compute_cost(prompt_tokens, completion_tokens),
                "toolCalls": tool_calls,
            }
        }


# ---------------------------------------------------------------------------

def _build_inference_config(settings: dict) -> dict:
    cfg = {}
    if settings.get("temperature") is not None:
        cfg["temperature"] = float(settings["temperature"])
    if settings.get("max_tokens") is not None:
        cfg["maxTokens"] = int(settings["max_tokens"])
    if settings.get("top_p") is not None:
        cfg["topP"] = float(settings["top_p"])
    stop = settings.get("stop") or settings.get("stopSequences")
    if stop:
        cfg["stopSequences"] = stop if isinstance(stop, list) else [stop]
    return cfg
