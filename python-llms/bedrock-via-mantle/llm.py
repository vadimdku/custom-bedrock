import json
from urllib import error, request

from dataiku.llm.python import BaseLLM
from custombedrock import (
    convert_messages_openai,
    convert_tools_openai,
    extract_openai_reasoning,
    extract_openai_text,
    extract_openai_tool_calls,
)


_PRICING = {
    # Prices are stored per 1K tokens. Console prices are displayed per 1M tokens.
    "openai.gpt-5.4": {"prompt": 0.00275, "completion": 0.01650},
    "nvidia.nemotron-super-3-120b": {"prompt": 0.00018, "completion": 0.00078},
    "nvidia.nemotron-nano-3-30b": {"prompt": 0.00007, "completion": 0.00029},
    "openai.gpt-oss-20b": {"prompt": 0.00008, "completion": 0.00036},
    "openai.gpt-oss-120b": {"prompt": 0.00018, "completion": 0.00072},
    "nvidia.nemotron-nano-12b-v2": {"prompt": 0.00024, "completion": 0.00072},
    "nvidia.nemotron-nano-9b-v2": {"prompt": 0.00007, "completion": 0.00028},
}

_OPENAI_PATH_MODELS = {"openai.gpt-5.4"}


class MyLLM(BaseLLM):
    def __init__(self):
        self.api_key = None
        self.base_url = None
        self.model_id = None
        self.use_responses_api = False
        self.max_parallelism_val = 8
        self._pricing = {"prompt": 0.0, "completion": 0.0}

    def set_config(self, config: dict, plugin_config: dict) -> None:
        region = (config.get("region") or "us-west-2").strip()
        configured_api_key = (config.get("apiKey") or "").strip()
        self.max_parallelism_val = int(config.get("maxParallelism") or 8)

        self.model_id = config["modelId"].strip()
        self._pricing = _PRICING.get(self.model_id, {"prompt": 0.0, "completion": 0.0})
        self.use_responses_api = self.model_id in _OPENAI_PATH_MODELS
        base_path = "openai/v1" if self.use_responses_api else "v1"
        self.base_url = f"https://bedrock-mantle.{region}.api.aws/{base_path}"

        self.api_key = configured_api_key
        if not self.api_key:
            raise ValueError("No API key found. Provide one in the Bedrock or OpenAI API key field.")

    def get_max_parallelism(self) -> int:
        return self.max_parallelism_val

    def _build_responses_request(self, query: dict, settings: dict) -> dict:
        input_items, instructions = convert_messages_openai(query["messages"])

        req = {
            "model": self.model_id,
            "input": input_items,
            "store": False,
        }

        if instructions:
            req["instructions"] = instructions

        if settings.get("temperature") is not None:
            req["temperature"] = float(settings["temperature"])
        if settings.get("max_tokens") is not None:
            req["max_output_tokens"] = int(settings["max_tokens"])
        if settings.get("top_p") is not None:
            req["top_p"] = float(settings["top_p"])
        if settings.get("reasoning") is not None:
            req["reasoning"] = settings["reasoning"]

        tools = _get_tools(query, settings)
        if tools:
            req["tools"] = convert_tools_openai(tools)
            tool_choice = _convert_tool_choice(settings.get("toolChoice"))
            if tool_choice is not None:
                req["tool_choice"] = tool_choice

        return req

    def _build_chat_request(self, query: dict, settings: dict) -> dict:
        req = {
            "model": self.model_id,
            "messages": [_convert_chat_message(msg) for msg in query["messages"]],
        }

        if settings.get("temperature") is not None:
            req["temperature"] = float(settings["temperature"])
        if settings.get("max_tokens") is not None:
            max_tokens_param = (
                "max_completion_tokens"
                if self.model_id.startswith("openai.gpt-oss-")
                else "max_tokens"
            )
            req[max_tokens_param] = int(settings["max_tokens"])
        if settings.get("top_p") is not None:
            req["top_p"] = float(settings["top_p"])
        if settings.get("reasoning_effort") is not None:
            req["reasoning_effort"] = settings["reasoning_effort"]

        tools = _get_tools(query, settings)
        if tools:
            req["tools"] = _convert_chat_tools(tools)
            tool_choice = _convert_tool_choice(settings.get("toolChoice"))
            if tool_choice is not None:
                req["tool_choice"] = tool_choice

        return req

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            (prompt_tokens / 1000.0) * self._pricing["prompt"]
            + (completion_tokens / 1000.0) * self._pricing["completion"]
        )

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post_json(self, path: str, payload: dict, stream: bool = False):
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            return request.urlopen(req)
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Bedrock Mantle HTTP {e.code}: {detail}") from e
        except error.URLError as e:
            raise RuntimeError(f"Bedrock Mantle connection error: {e}") from e

    def process(self, query, settings, trace):
        if not self.use_responses_api:
            return self._process_chat(query, settings)

        req = self._build_responses_request(query, settings)
        resp = self._post_json("/responses", req)
        response = json.loads(resp.read().decode("utf-8"))

        usage = response.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        reasoning = extract_openai_reasoning(response)

        result = {
            "text": extract_openai_text(response),
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "estimatedCost": self._compute_cost(prompt_tokens, completion_tokens),
            "toolCalls": extract_openai_tool_calls(response),
            "finishReason": _responses_finish_reason(response),
        }
        _add_reasoning_artifact(result, reasoning)
        return result

    def process_stream(self, query, settings, trace):
        if not self.use_responses_api:
            yield from self._process_chat_stream(query, settings)
            return

        req = self._build_responses_request(query, settings)
        req["stream"] = True

        resp = self._post_json("/responses", req, stream=True)

        data_lines = []
        final_response = None
        response_tool_call_parts = {}
        reasoning_parts = []
        reasoning_streamed = False

        for raw_line in resp:
            line = raw_line.decode("utf-8")

            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
                continue

            if line.strip():
                continue

            if not data_lines:
                continue

            payload = "\n".join(data_lines)
            data_lines = []

            if payload == "[DONE]":
                break

            event = json.loads(payload)
            event_type = event.get("type", "")

            if event_type == "error":
                raise RuntimeError(f"Bedrock Mantle stream error: {event}")

            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    yield {"chunk": {"text": delta}}

            elif event_type == "response.refusal.delta":
                delta = event.get("delta", "")
                if delta:
                    yield {"chunk": {"text": delta}}

            elif event_type in {"response.reasoning_text.delta", "response.reasoning_summary_text.delta"}:
                delta = event.get("delta", "")
                if delta:
                    reasoning_parts.append(delta)
                    yield {"chunk": {"artifacts": [_reasoning_artifact_delta(delta)]}}
                    reasoning_streamed = True

            elif event_type in {"response.reasoning_text.done", "response.reasoning_summary_text.done"}:
                text = event.get("text", "")
                if text:
                    reasoning_parts = [text]

            elif event_type == "response.completed":
                final_response = event.get("response", {})

            elif event_type in {"response.output_item.added", "response.output_item.done"}:
                item = event.get("item") or {}
                if item.get("type") == "function_call":
                    _record_responses_tool_item(response_tool_call_parts, item, event)

            elif event_type == "response.function_call_arguments.delta":
                _record_responses_tool_arguments_delta(response_tool_call_parts, event)

            elif event_type == "response.function_call_arguments.done":
                _record_responses_tool_arguments_done(response_tool_call_parts, event)

        if data_lines and data_lines != ["[DONE]"]:
            event = json.loads("\n".join(data_lines))
            if event.get("type") == "response.completed":
                final_response = event.get("response", {})

        final_response = final_response or {}
        usage = final_response.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        tool_calls = extract_openai_tool_calls(final_response) or _build_responses_tool_calls(response_tool_call_parts)
        reasoning = extract_openai_reasoning(final_response) or "".join(reasoning_parts)

        if tool_calls:
            yield {"chunk": {"toolCalls": _with_tool_call_indexes(tool_calls)}}
        if reasoning and not reasoning_streamed:
            yield {"chunk": {"artifacts": [_reasoning_artifact(reasoning)]}}

        yield {
            "footer": {
                "finishReason": _responses_finish_reason(final_response),
                "promptTokens": prompt_tokens,
                "completionTokens": completion_tokens,
                "estimatedCost": self._compute_cost(prompt_tokens, completion_tokens),
                "toolCalls": tool_calls,
            }
        }

    def _process_chat(self, query, settings):
        req = self._build_chat_request(query, settings)
        resp = self._post_json("/chat/completions", req)
        response = json.loads(resp.read().decode("utf-8"))

        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        reasoning = _extract_chat_reasoning(message)

        result = {
            "text": message.get("content") or "",
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "estimatedCost": self._compute_cost(prompt_tokens, completion_tokens),
            "toolCalls": _extract_chat_tool_calls(message),
            "finishReason": _map_finish_reason(choice.get("finish_reason")),
        }
        _add_reasoning_artifact(result, reasoning)
        return result

    def _process_chat_stream(self, query, settings):
        req = self._build_chat_request(query, settings)
        req["stream"] = True
        req["stream_options"] = {"include_usage": True}

        resp = self._post_json("/chat/completions", req, stream=True)

        prompt_tokens = 0
        completion_tokens = 0
        finish_reason = None
        reasoning_parts = []
        reasoning_streamed = False
        tool_call_parts = {}

        for event in _iter_sse_events(resp):
            if event == "[DONE]":
                break

            data = json.loads(event)
            usage = data.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            completion_tokens = usage.get("completion_tokens", completion_tokens)

            for choice in data.get("choices") or []:
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                reasoning = _extract_chat_reasoning(delta)
                if reasoning:
                    reasoning_parts.append(reasoning)
                    yield {"chunk": {"artifacts": [_reasoning_artifact_delta(reasoning)]}}
                    reasoning_streamed = True
                content = delta.get("content")
                if content:
                    yield {"chunk": {"text": content}}

                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    item = tool_call_parts.setdefault(idx, {
                        "id": "",
                        "name": "",
                        "arguments": [],
                    })
                    if tc.get("id"):
                        item["id"] = tc["id"]
                    function = tc.get("function") or {}
                    if function.get("name"):
                        item["name"] = function["name"]
                    if function.get("arguments"):
                        item["arguments"].append(function["arguments"])

        tool_calls = [
            {
                "type": "function",
                "id": tc["id"],
                "index": idx,
                "function": {
                    "name": tc["name"],
                    "arguments": "".join(tc["arguments"]),
                },
            }
            for _, tc in sorted(tool_call_parts.items())
        ]

        if tool_calls:
            yield {"chunk": {"toolCalls": tool_calls}}
        reasoning = "".join(reasoning_parts)
        if reasoning and not reasoning_streamed:
            yield {"chunk": {"artifacts": [_reasoning_artifact(reasoning)]}}

        yield {
            "footer": {
                "finishReason": _map_finish_reason(finish_reason),
                "promptTokens": prompt_tokens,
                "completionTokens": completion_tokens,
                "estimatedCost": self._compute_cost(prompt_tokens, completion_tokens),
                "toolCalls": tool_calls,
            }
        }


def _iter_sse_events(resp):
    data_lines = []

    for raw_line in resp:
        line = raw_line.decode("utf-8")

        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
            continue

        if line.strip():
            continue

        if data_lines:
            yield "\n".join(data_lines)
            data_lines = []

    if data_lines:
        yield "\n".join(data_lines)


def _get_tools(query: dict, settings: dict) -> list:
    return query.get("tools") or settings.get("tools") or []


def _convert_tool_choice(tool_choice):
    if not tool_choice:
        return None

    if isinstance(tool_choice, str):
        return tool_choice

    if not isinstance(tool_choice, dict):
        return None

    choice_type = tool_choice.get("type")
    if choice_type in {"auto", "required", "none"}:
        return choice_type
    if choice_type == "tool_name" and tool_choice.get("name"):
        return {"type": "function", "name": tool_choice["name"]}
    return None


def _responses_finish_reason(response: dict) -> str:
    status = response.get("status")
    if status == "completed":
        return "STOP"

    incomplete_reason = (response.get("incomplete_details") or {}).get("reason")
    if incomplete_reason:
        return _map_finish_reason(incomplete_reason)

    if response.get("error"):
        return "ERROR"

    return _map_finish_reason(status)


def _map_finish_reason(finish_reason) -> str:
    mapping = {
        "stop": "STOP",
        "completed": "STOP",
        "end_turn": "STOP",
        "length": "LENGTH",
        "max_tokens": "LENGTH",
        "max_output_tokens": "LENGTH",
        "tool_calls": "TOOL_CALLS",
        "function_call": "TOOL_CALLS",
        "content_filter": "CONTENT_FILTER",
        "error": "ERROR",
        None: "UNKNOWN",
    }
    return mapping.get(finish_reason, str(finish_reason or "UNKNOWN").upper())


def _extract_chat_reasoning(message: dict) -> str:
    for key in ("reasoning_content", "reasoning"):
        reasoning = message.get(key)
        if reasoning:
            return _stringify_reasoning(reasoning)
    return ""


def _stringify_reasoning(reasoning) -> str:
    if isinstance(reasoning, str):
        return reasoning
    if isinstance(reasoning, list):
        parts = []
        for item in reasoning:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
        return "".join(parts)
    if isinstance(reasoning, dict):
        if reasoning.get("text"):
            return reasoning["text"]
        return json.dumps(reasoning)
    return str(reasoning)


def _reasoning_artifact(reasoning: str) -> dict:
    return {
        "id": "reasoning",
        "type": "REASONING",
        "parts": [
            {
                "index": 0,
                "type": "TEXT",
                "text": reasoning,
            }
        ],
    }


def _reasoning_artifact_delta(reasoning_delta: str) -> dict:
    return _reasoning_artifact(reasoning_delta)


def _add_reasoning_artifact(result: dict, reasoning: str) -> None:
    if reasoning:
        result["artifacts"] = [_reasoning_artifact(reasoning)]


def _responses_tool_key(item_or_event: dict) -> str:
    return str(
        item_or_event.get("item_id")
        or item_or_event.get("id")
        or item_or_event.get("call_id")
        or item_or_event.get("output_index")
        or len(item_or_event)
    )


def _record_responses_tool_item(tool_call_parts: dict, item: dict, event: dict) -> None:
    key = _responses_tool_key(item)
    tool_call = tool_call_parts.setdefault(key, {
        "id": "",
        "name": "",
        "arguments_parts": [],
        "final_arguments": "",
    })
    tool_call["id"] = item.get("call_id") or item.get("id") or tool_call["id"]
    tool_call["name"] = item.get("name") or tool_call["name"]
    if item.get("arguments"):
        tool_call["final_arguments"] = item["arguments"]


def _record_responses_tool_arguments_delta(tool_call_parts: dict, event: dict) -> None:
    key = _responses_tool_key(event)
    tool_call = tool_call_parts.setdefault(key, {
        "id": event.get("item_id", ""),
        "name": "",
        "arguments_parts": [],
        "final_arguments": "",
    })
    tool_call["arguments_parts"].append(event.get("delta", ""))


def _record_responses_tool_arguments_done(tool_call_parts: dict, event: dict) -> None:
    key = _responses_tool_key(event)
    tool_call = tool_call_parts.setdefault(key, {
        "id": event.get("item_id", ""),
        "name": "",
        "arguments_parts": [],
        "final_arguments": "",
    })
    tool_call["name"] = event.get("name") or tool_call["name"]
    tool_call["final_arguments"] = event.get("arguments", "") or tool_call["final_arguments"]


def _build_responses_tool_calls(tool_call_parts: dict) -> list:
    tool_calls = []
    for tool_call in tool_call_parts.values():
        arguments = tool_call["final_arguments"] or "".join(tool_call["arguments_parts"])
        if not tool_call["name"] and not arguments:
            continue
        tool_calls.append({
            "type": "function",
            "id": tool_call["id"],
            "index": len(tool_calls),
            "function": {
                "name": tool_call["name"],
                "arguments": arguments,
            },
        })
    return tool_calls


def _with_tool_call_indexes(tool_calls: list) -> list:
    indexed = []
    for idx, tool_call in enumerate(tool_calls):
        item = dict(tool_call)
        item.setdefault("index", idx)
        indexed.append(item)
    return indexed


def _convert_chat_message(msg: dict) -> dict:
    role = msg.get("role", "user")
    content = msg.get("content", "")

    if role == "assistant" and msg.get("toolCalls"):
        return {
            "role": "assistant",
            "content": content if isinstance(content, str) else "",
            "tool_calls": [_convert_top_level_chat_tool_call(tc) for tc in msg["toolCalls"]],
        }

    if role == "tool" and msg.get("toolOutputs"):
        return {
            "role": "tool",
            "tool_call_id": msg["toolOutputs"][0].get("callId", ""),
            "content": _stringify_tool_output(msg["toolOutputs"][0].get("output", "")),
        }

    if isinstance(content, str):
        return {"role": role, "content": content}

    if not isinstance(content, list):
        return {"role": role, "content": str(content)}

    parts = []
    tool_calls = []
    for block in content:
        if isinstance(block, str):
            parts.append({"type": "text", "text": block})
            continue

        if not isinstance(block, dict):
            continue

        btype = block.get("type", "")
        if btype == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image_url":
            image_url = block.get("image_url", {})
            if isinstance(image_url, str):
                image_url = {"url": image_url}
            parts.append({"type": "image_url", "image_url": image_url})
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })
        elif btype == "tool_result":
            return {
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": _stringify_tool_output(block.get("content", "")),
            }

    converted = {"role": role, "content": parts if parts else ""}
    if tool_calls:
        converted["tool_calls"] = tool_calls
    return converted


def _convert_top_level_chat_tool_call(tool_call: dict) -> dict:
    function = tool_call.get("function") or {}
    return {
        "id": tool_call.get("id", ""),
        "type": "function",
        "function": {
            "name": function.get("name", ""),
            "arguments": function.get("arguments", ""),
        },
    }


def _stringify_tool_output(content) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content)


def _convert_chat_tools(tools: list) -> list:
    chat_tools = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            chat_tools.append(tool)
        elif "function" in tool:
            chat_tools.append({"type": "function", "function": tool["function"]})
        else:
            chat_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", tool.get("input_schema", {})),
                },
            })
    return chat_tools


def _extract_chat_tool_calls(message: dict) -> list:
    return [
        {
            "type": "function",
            "id": tc.get("id", ""),
            "function": {
                "name": (tc.get("function") or {}).get("name", ""),
                "arguments": (tc.get("function") or {}).get("arguments", ""),
            },
        }
        for tc in message.get("tool_calls") or []
    ]
