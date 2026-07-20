#!/usr/bin/env python3
"""
Standalone test script for the Bedrock Mantle LLM plugin.
Run from anywhere: python python-lib/test_llm_mantle.py

Requirements:
  BEDROCK_API_KEY or OPENAI_API_KEY must be exported in the shell.
"""

import json
import os
import sys
import types

PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PLUGIN_ROOT, "python-lib"))
sys.path.insert(0, os.path.join(PLUGIN_ROOT, "python-llms", "bedrock-via-mantle"))

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

from llm import MyLLM  # noqa: E402


PLUGIN_CONFIG = {}

LLM_CONFIG = {
    "region": "us-west-2",
    "apiKey": os.environ.get("BEDROCK_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
    "maxParallelism": 8,
    "modelId": "openai.gpt-5.4",
}

SETTINGS = {
    "temperature": 0.2,
    "max_tokens": 256,
}

SIMPLE_QUERY = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is the capital of France? Answer in one sentence."},
    ],
    "tools": [],
}

MULTI_TURN_QUERY = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What model are you?"},
        {"role": "assistant", "content": "I am ChatGPT running through this host application."},
        {"role": "user", "content": "What version are you?"},
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
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            },
        }
    ],
}


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


if __name__ == "__main__":
    if not (os.environ.get("BEDROCK_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        raise SystemExit("BEDROCK_API_KEY or OPENAI_API_KEY must be set.")

    print(f"model:             {LLM_CONFIG['modelId']}")
    print(f"region:            {LLM_CONFIG['region']}")

    llm = MyLLM()
    llm.set_config(LLM_CONFIG, PLUGIN_CONFIG)
    print("set_config:        OK")

    _header("1 · Synchronous completion  (process)")
    _print_result(llm.process(SIMPLE_QUERY, SETTINGS, _NoopTrace()))

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
        print(f"  promptTokens:      {footer['promptTokens']}")
        print(f"  completionTokens:  {footer['completionTokens']}")
        print(f"  estimatedCost:     ${footer['estimatedCost']:.8f}")
        if footer.get("toolCalls"):
            print(f"  toolCalls:         {json.dumps(footer['toolCalls'], indent=4)}")

    _header("3 · Tool call  (process)")
    _print_result(llm.process(TOOL_QUERY, SETTINGS, _NoopTrace()))

    _header("4 · Multi-turn completion  (process)")
    _print_result(llm.process(MULTI_TURN_QUERY, SETTINGS, _NoopTrace()))
