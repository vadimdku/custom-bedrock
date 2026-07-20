#!/usr/bin/env python3
"""
End-to-end test for the Bedrock LLM — Dataiku environment version.
Runs inside a DSS code notebook or recipe; custombedrock is already on sys.path
and the real dataiku module is available (no mocking required).

Usage (DSS notebook cell):
    exec(open("test_llm_dku.py").read())

Requirements:
  - AWS credentials reachable (env vars, ~/.aws/credentials, instance profile,
    or a valid s3Connection configured below).
  - The target model must be enabled in your Bedrock account.
"""

import json
import logging
import botocore.exceptions

from custombedrock import (
    get_boto3_session,
    convert_messages,
    convert_tools,
    extract_tool_calls,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inline MyLLM — mirrors python-llms/bedrock-via-converse/llm.py exactly,
# but relies on custombedrock already being on sys.path (DSS environment).
# ---------------------------------------------------------------------------

_PRICING = {
    "amazon.nova-pro-v1:0":                        {"prompt": 0.0008,    "completion": 0.0032},
    "amazon.nova-lite-v1:0":                       {"prompt": 0.00006,   "completion": 0.00024},
    "amazon.nova-micro-v1:0":                      {"prompt": 0.000035,  "completion": 0.00014},
    "amazon.titan-text-premier-v1:0":              {"prompt": 0.0005,    "completion": 0.0015},
    "amazon.titan-text-lite-v1":                   {"prompt": 0.00015,   "completion": 0.0002},
    "amazon.titan-text-express-v1":                {"prompt": 0.0002,    "completion": 0.0006},
    "anthropic.claude-sonnet-4-5-20251015-v1:0":   {"prompt": 0.003,     "completion": 0.015},
    "anthropic.claude-haiku-4-5-20251001-v1:0":    {"prompt": 0.001,     "completion": 0.005},
    "anthropic.claude-3-7-sonnet-20250219-v1:0":   {"prompt": 0.003,     "completion": 0.015},
    "anthropic.claude-3-5-sonnet-20241022-v2:0":   {"prompt": 0.003,     "completion": 0.015},
    "anthropic.claude-3-5-haiku-20241022-v1:0":    {"prompt": 0.0008,    "completion": 0.004},
    "anthropic.claude-3-haiku-20240307-v1:0":      {"prompt": 0.00025,   "completion": 0.00125},
    "meta.llama3-3-70b-instruct-v1:0":             {"prompt": 0.00072,   "completion": 0.00072},
    "meta.llama3-1-8b-instruct-v1:0":              {"prompt": 0.00022,   "completion": 0.00022},
    "mistral.mistral-large-2407-v1:0":             {"prompt": 0.003,     "completion": 0.009},
    "deepseek.r1-v1:0":                            {"prompt": 0.00135,   "completion": 0.0054},
}

_SEPARATOR_VALUES = {"__amazon__", "__anthropic__", "__ai21__", "__cohere__", "__meta__", "__mistral__", "__deepseek__"}

_PROFILE_PREFIX = {"us": "us", "eu": "eu", "apac": "ap"}


class MyLLM:
    def __init__(self):
        self.client = None
        self.model_id = None
        self.guardrail_id = None
        self.guardrail_version = "DRAFT"
        self.max_parallelism_val = 8
        self._pricing = {"prompt": 0.0, "completion": 0.0}

    def set_config(self, config: dict, plugin_config: dict) -> None:
        region            = (config.get("region") or "us-east-1").strip()
        s3_connection     = (config.get("s3Connection") or "").strip()
        inference_profile = (config.get("inferenceProfile") or "").strip()
        use_guardrail     = bool(config.get("useGuardrail", False))
        self.guardrail_id      = (config.get("guardrailIdentifier") or "").strip() if use_guardrail else ""
        self.guardrail_version = (config.get("guardrailVersion") or "DRAFT").strip()
        self.max_parallelism_val = int(config.get("maxParallelism") or 8)

        base_model_id = config["modelId"]
        if base_model_id in _SEPARATOR_VALUES:
            raise ValueError(f"'{base_model_id}' is a section separator, not a model.")
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

    def _build_request(self, query: dict, settings: dict) -> dict:
        bedrock_messages, bedrock_system = convert_messages(query["messages"])
        req = {"modelId": self.model_id, "messages": bedrock_messages}
        if bedrock_system:
            req["system"] = bedrock_system
        inf_cfg = _build_inference_config(settings)
        if inf_cfg:
            req["inferenceConfig"] = inf_cfg
        if self.guardrail_id:
            req["guardrailConfig"] = {
                "guardrailIdentifier": self.guardrail_id,
                "guardrailVersion":    self.guardrail_version,
            }
        tools = query.get("tools") or []
        if tools:
            req["toolConfig"] = {"tools": convert_tools(tools)}
        return req

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            (prompt_tokens    / 1000.0) * self._pricing["prompt"]
            + (completion_tokens / 1000.0) * self._pricing["completion"]
        )

    def process(self, query, settings, trace):
        req = self._build_request(query, settings)
        try:
            resp = self.client.converse(**req)
        except botocore.exceptions.ClientError as e:
            raise RuntimeError(f"Bedrock Converse error: {e}") from e

        output_content   = resp["output"]["message"].get("content", [])
        usage            = resp.get("usage", {})
        prompt_tokens    = usage.get("inputTokens", 0)
        completion_tokens = usage.get("outputTokens", 0)
        text             = "".join(b["text"] for b in output_content if "text" in b)
        tool_calls       = extract_tool_calls(output_content)

        return {
            "text":             text,
            "promptTokens":     prompt_tokens,
            "completionTokens": completion_tokens,
            "estimatedCost":    self._compute_cost(prompt_tokens, completion_tokens),
            "toolCalls":        tool_calls,
        }

    def process_stream(self, query, settings, trace):
        req = self._build_request(query, settings)
        try:
            resp = self.client.converse_stream(**req)
        except botocore.exceptions.ClientError as e:
            raise RuntimeError(f"Bedrock ConverseStream error: {e}") from e

        prompt_tokens = 0
        completion_tokens = 0
        open_tools = {}

        try:
            for event in resp["stream"]:
                if "contentBlockStart" in event:
                    cbs   = event["contentBlockStart"]
                    idx   = cbs["contentBlockIndex"]
                    start = cbs.get("start", {})
                    if "toolUse" in start:
                        open_tools[idx] = {
                            "id":          start["toolUse"]["toolUseId"],
                            "name":        start["toolUse"]["name"],
                            "input_parts": [],
                        }
                elif "contentBlockDelta" in event:
                    cbd   = event["contentBlockDelta"]
                    idx   = cbd["contentBlockIndex"]
                    delta = cbd.get("delta", {})
                    if "text" in delta:
                        yield {"chunk": {"text": delta["text"]}}
                    elif "toolUse" in delta and idx in open_tools:
                        open_tools[idx]["input_parts"].append(delta["toolUse"].get("input", ""))
                elif "metadata" in event:
                    usage = event["metadata"].get("usage", {})
                    prompt_tokens    = usage.get("inputTokens", 0)
                    completion_tokens = usage.get("outputTokens", 0)
        except botocore.exceptions.EventStreamError as e:
            raise RuntimeError(f"Bedrock stream interrupted: {e}") from e

        tool_calls = [
            {
                "type": "function",
                "id":   tc["id"],
                "function": {
                    "name":      tc["name"],
                    "arguments": "".join(tc["input_parts"]),
                },
            }
            for tc in open_tools.values()
        ]

        yield {
            "footer": {
                "promptTokens":     prompt_tokens,
                "completionTokens": completion_tokens,
                "estimatedCost":    self._compute_cost(prompt_tokens, completion_tokens),
                "toolCalls":        tool_calls,
            }
        }


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


# ---------------------------------------------------------------------------
# Configuration — edit these to match your environment
# ---------------------------------------------------------------------------

PLUGIN_CONFIG = {}

LLM_CONFIG = {
    "s3Connection":       "",           # leave empty to use default boto3 credential chain
    "region":             "us-east-1",
    "inferenceProfile":   "",           # e.g. "us", "eu", "apac"
    "maxParallelism":     8,
    "useGuardrail":       False,
    "guardrailIdentifier": "",
    "guardrailVersion":   "DRAFT",
    "modelId":            "amazon.nova-pro-v1:0",
}

SETTINGS = {
    "temperature": 0.7,
    "max_tokens":  256,
}

# ---------------------------------------------------------------------------
# Example queries (same as test_llm.py)
# ---------------------------------------------------------------------------

SIMPLE_QUERY = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user",   "content": "What is the capital of France? Answer in one sentence."},
    ],
    "tools": [],
}

MULTI_TURN_QUERY = {
    "messages": [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! How can I help you today?"},
        {"role": "user",      "content": "What is my name?"},
    ],
    "tools": [],
}

TOOL_QUERY = {
    "messages": [
        {"role": "user", "content": "What is the weather in Paris right now?"},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name":        "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            },
        }
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NoopTrace:
    def log(self, *args, **kwargs):
        pass


def _header(title):
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def _print_result(result):
    print(f"  text:              {result['text']!r}")
    print(f"  promptTokens:      {result['promptTokens']}")
    print(f"  completionTokens:  {result['completionTokens']}")
    print(f"  estimatedCost:     ${result['estimatedCost']:.8f}")
    if result.get("toolCalls"):
        print(f"  toolCalls:         {json.dumps(result['toolCalls'], indent=4)}")


def _print_footer(footer):
    print(f"\n  promptTokens:      {footer['promptTokens']}")
    print(f"  completionTokens:  {footer['completionTokens']}")
    print(f"  estimatedCost:     ${footer['estimatedCost']:.8f}")
    if footer.get("toolCalls"):
        print(f"  toolCalls:         {json.dumps(footer['toolCalls'], indent=4)}")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_sync(llm):
    _header("1 · Synchronous completion  (process)")
    result = llm.process(SIMPLE_QUERY, SETTINGS, _NoopTrace())
    _print_result(result)


def test_stream(llm):
    _header("2 · Streaming completion  (process_stream)")
    print("  text (streaming): ", end="", flush=True)
    footer = None
    for event in llm.process_stream(SIMPLE_QUERY, SETTINGS, _NoopTrace()):
        if "chunk" in event:
            print(event["chunk"]["text"], end="", flush=True)
        elif "footer" in event:
            footer = event["footer"]
    print()
    if footer:
        _print_footer(footer)


def test_multi_turn(llm):
    _header("3 · Multi-turn conversation  (process)")
    result = llm.process(MULTI_TURN_QUERY, SETTINGS, _NoopTrace())
    _print_result(result)


def test_tool_call(llm):
    _header("4 · Tool call  (process)")
    result = llm.process(TOOL_QUERY, SETTINGS, _NoopTrace())
    _print_result(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print(f"model:             {LLM_CONFIG['modelId']}")
print(f"region:            {LLM_CONFIG['region']}")
print(f"inferenceProfile:  {LLM_CONFIG['inferenceProfile'] or '(none)'}")
print(f"guardrailId:       {LLM_CONFIG['guardrailIdentifier'] or '(none)'}")

llm = MyLLM()
llm.set_config(LLM_CONFIG, PLUGIN_CONFIG)
print("set_config:        OK")

test_sync(llm)
test_stream(llm)
test_multi_turn(llm)
test_tool_call(llm)

_header("All tests completed")
