#!/usr/bin/env python3
"""
Unit tests for custombedrock python-lib.
Run inside a Dataiku code notebook or recipe — custombedrock must already be on sys.path.

Usage (DSS notebook cell):
    exec(open("test_custombedrock.py").read())
"""

import json

from custombedrock import (
    convert_messages,
    convert_tools,
    extract_tool_calls,
    get_boto3_session,
)


# ---------------------------------------------------------------------------
# Minimal test harness
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0


def _assert(condition, label):
    global _passed, _failed
    if condition:
        print(f"  PASS  {label}")
        _passed += 1
    else:
        print(f"  FAIL  {label}")
        _failed += 1


def _assert_eq(actual, expected, label):
    ok = actual == expected
    if not ok:
        print(f"  FAIL  {label}")
        print(f"        expected: {expected!r}")
        print(f"        actual:   {actual!r}")
        global _failed
        _failed += 1
    else:
        print(f"  PASS  {label}")
        global _passed
        _passed += 1


def _section(title):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


# ---------------------------------------------------------------------------
# convert_messages
# ---------------------------------------------------------------------------

_section("convert_messages — system messages")

msgs, sys_blocks = convert_messages([
    {"role": "system", "content": "You are helpful."},
    {"role": "user",   "content": "Hello"},
])
_assert_eq(sys_blocks, [{"text": "You are helpful."}], "system text extracted")
_assert_eq(len(msgs), 1, "one user message remains")
_assert_eq(msgs[0]["role"], "user", "user role preserved")
_assert_eq(msgs[0]["content"], [{"text": "Hello"}], "user content wrapped in block")

_section("convert_messages — no system message")

msgs, sys_blocks = convert_messages([
    {"role": "user",      "content": "Question"},
    {"role": "assistant", "content": "Answer"},
])
_assert_eq(sys_blocks, [], "no system blocks")
_assert_eq(len(msgs), 2, "two messages")

_section("convert_messages — list content (multi-block)")

msgs, _ = convert_messages([
    {"role": "user", "content": [
        {"type": "text", "text": "What is this?"},
    ]},
])
_assert_eq(msgs[0]["content"], [{"text": "What is this?"}], "text block from list")

_section("convert_messages — tool_use block")

msgs, _ = convert_messages([
    {"role": "assistant", "content": [
        {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Paris"}},
    ]},
])
block = msgs[0]["content"][0]
_assert("toolUse" in block, "toolUse key present")
_assert_eq(block["toolUse"]["toolUseId"], "call_1", "toolUseId")
_assert_eq(block["toolUse"]["name"], "get_weather", "tool name")
_assert_eq(block["toolUse"]["input"], {"city": "Paris"}, "tool input")

_section("convert_messages — tool_result block")

msgs, _ = convert_messages([
    {"role": "tool", "content": [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "Sunny, 22°C"},
    ]},
])
block = msgs[0]["content"][0]
_assert("toolResult" in block, "toolResult key present")
_assert_eq(block["toolResult"]["toolUseId"], "call_1", "toolUseId in result")
_assert_eq(block["toolResult"]["content"], [{"text": "Sunny, 22°C"}], "result text wrapped")

_section("convert_messages — empty content skipped")

msgs, _ = convert_messages([
    {"role": "user", "content": ""},
])
_assert_eq(msgs, [], "empty-content message dropped")

_section("convert_messages — multi-turn conversation")

turns = [
    {"role": "system",    "content": "Be concise."},
    {"role": "user",      "content": "Name?"},
    {"role": "assistant", "content": "Alice"},
    {"role": "user",      "content": "Age?"},
]
msgs, sys_blocks = convert_messages(turns)
_assert_eq(len(sys_blocks), 1,        "one system block")
_assert_eq(len(msgs), 3,              "three non-system messages")
_assert_eq(msgs[2]["content"][0]["text"], "Age?", "last user message correct")


# ---------------------------------------------------------------------------
# convert_tools
# ---------------------------------------------------------------------------

_section("convert_tools — OpenAI function format")

dku_tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
bedrock_tools = convert_tools(dku_tools)
_assert_eq(len(bedrock_tools), 1, "one tool")
spec = bedrock_tools[0]["toolSpec"]
_assert_eq(spec["name"],        "get_weather", "tool name")
_assert_eq(spec["description"], "Get weather", "tool description")
_assert_eq(spec["inputSchema"]["json"]["type"], "object", "schema type")

_section("convert_tools — empty list")

_assert_eq(convert_tools([]), [], "empty tools list")

_section("convert_tools — multiple tools")

tools = convert_tools([
    {"type": "function", "function": {"name": "a", "description": "A", "parameters": {}}},
    {"type": "function", "function": {"name": "b", "description": "B", "parameters": {}}},
])
_assert_eq(len(tools), 2, "two tools converted")
_assert_eq(tools[1]["toolSpec"]["name"], "b", "second tool name")


# ---------------------------------------------------------------------------
# extract_tool_calls
# ---------------------------------------------------------------------------

_section("extract_tool_calls — single tool call")

content_blocks = [
    {"text": "I'll check the weather."},
    {"toolUse": {"toolUseId": "call_abc", "name": "get_weather", "input": {"city": "Paris"}}},
]
tool_calls = extract_tool_calls(content_blocks)
_assert_eq(len(tool_calls), 1, "one tool call extracted")
tc = tool_calls[0]
_assert_eq(tc["type"],              "function",    "type is function")
_assert_eq(tc["id"],                "call_abc",    "tool call id")
_assert_eq(tc["function"]["name"],  "get_weather", "function name")
args = json.loads(tc["function"]["arguments"])
_assert_eq(args["city"], "Paris", "arguments parsed correctly")

_section("extract_tool_calls — no tool calls")

_assert_eq(extract_tool_calls([{"text": "Hello"}]), [], "no tool calls")

_section("extract_tool_calls — multiple tool calls")

blocks = [
    {"toolUse": {"toolUseId": "c1", "name": "tool_a", "input": {"x": 1}}},
    {"toolUse": {"toolUseId": "c2", "name": "tool_b", "input": {"y": 2}}},
]
tcs = extract_tool_calls(blocks)
_assert_eq(len(tcs), 2, "two tool calls")
_assert_eq(tcs[0]["id"], "c1", "first call id")
_assert_eq(tcs[1]["function"]["name"], "tool_b", "second call name")


# ---------------------------------------------------------------------------
# get_boto3_session — ambient credentials (no DSS connection)
# ---------------------------------------------------------------------------

_section("get_boto3_session — ambient credentials")

session = get_boto3_session("", "us-east-1")
_assert(session is not None, "session returned")
_assert_eq(session.region_name, "us-east-1", "region set correctly")

session2 = get_boto3_session("", "eu-west-1")
_assert_eq(session2.region_name, "eu-west-1", "different region")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'═' * 55}")
print(f"  Results: {_passed} passed, {_failed} failed")
print(f"{'═' * 55}")
if _failed:
    raise AssertionError(f"{_failed} test(s) failed — see output above.")
