#!/usr/bin/env python3
"""
Standalone test script for the custom-bedrock LLM plugin.
Run from anywhere: python python-lib/test_llm.py

Requirements:
  pip install boto3
  AWS credentials must be configured (env vars, ~/.aws/credentials, or instance profile).
  The target model must be enabled in your AWS Bedrock account.
"""

import sys
import os
import json
import types

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PLUGIN_ROOT, "python-lib"))
sys.path.insert(0, os.path.join(PLUGIN_ROOT, "python-llms", "bedrock-via-converse"))

# ---------------------------------------------------------------------------
# Mock dataiku.llm.python.BaseLLM — not available outside DSS
# ---------------------------------------------------------------------------
_dataiku = types.ModuleType("dataiku")
_dataiku_llm = types.ModuleType("dataiku.llm")
_dataiku_llm_python = types.ModuleType("dataiku.llm.python")

class BaseLLM:
    pass

_dataiku_llm_python.BaseLLM = BaseLLM
_dataiku_llm.python = _dataiku_llm_python
_dataiku.llm = _dataiku_llm
sys.modules.setdefault("dataiku", _dataiku)
sys.modules.setdefault("dataiku.llm", _dataiku_llm)
sys.modules.setdefault("dataiku.llm.python", _dataiku_llm_python)

# Import after mocking
from llm import MyLLM  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration — edit these to match your environment
# ---------------------------------------------------------------------------

# All params now live in llm.json, so plugin_config is empty
PLUGIN_CONFIG = {}

# Mirrors config (per-LLM params from llm.json)
LLM_CONFIG = {
    "s3Connection": "",       # leave empty to use default boto3 credential chain
    "region": "us-east-1",
    "inferenceProfile": "",   # e.g. "us", "eu", "apac"
    "maxParallelism": 8,
    "useGuardrail": False,
    "guardrailIdentifier": "",
    "guardrailVersion": "DRAFT",
    "modelId": "amazon.nova-pro-v1:0",
}

# Standard LLM settings passed alongside each query
SETTINGS = {
    "temperature": 0.7,
    "max_tokens": 256,
}


# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------

SIMPLE_QUERY = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is the capital of France? Answer in one sentence."},
    ],
    "tools": [],
}

MULTI_TURN_QUERY = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! How can I help you today?"},
        {"role": "user", "content": "What is my name?"},
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
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city name",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
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

if __name__ == "__main__":
    print(f"model:             {LLM_CONFIG['modelId']}")
    print(f"region:            {LLM_CONFIG['region']}")
    print(f"inferenceProfile:  {LLM_CONFIG['inferenceProfile'] or '(none)'}")
    print(f"guardrailId:       {LLM_CONFIG['guardrailId'] or '(none)'}")

    llm = MyLLM()
    llm.set_config(LLM_CONFIG, PLUGIN_CONFIG)
    print("set_config:        OK")

    test_sync(llm)
    test_stream(llm)
    test_multi_turn(llm)
    test_tool_call(llm)

    _header("All tests completed")
