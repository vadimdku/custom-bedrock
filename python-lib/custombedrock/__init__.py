import json
import boto3


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------

def get_boto3_session(connection_name: str, region: str) -> boto3.Session:
    """
    Returns a boto3 Session using AWS credentials resolved from the named DSS
    connection via DSSConnectionInfo.get_aws_credential(). Falls back to the
    ambient boto3 credential chain (env vars, ~/.aws/credentials, instance
    profile) on any error or when connection_name is empty.
    """
    if connection_name:
        try:
            import dataiku
            aws_cred = (
                dataiku.api_client()
                .get_connection(connection_name)
                .get_info()
                .get_aws_credential()
            )
            return boto3.Session(
                aws_access_key_id=aws_cred.get("accessKey"),
                aws_secret_access_key=aws_cred.get("secretKey"),
                aws_session_token=aws_cred.get("sessionToken"),
                region_name=region,
            )
        except Exception:
            pass

    return boto3.Session(region_name=region)


# ---------------------------------------------------------------------------
# Message format conversion: Dataiku (OpenAI-like) -> Bedrock Converse API
# ---------------------------------------------------------------------------

def convert_messages(dku_messages: list) -> tuple:
    """
    Splits Dataiku messages into (bedrock_messages, bedrock_system).

    System-role messages are extracted into bedrock_system (a list of
    {text: ...} blocks) — Bedrock requires them at the top level, not inline.
    """
    bedrock_system = []
    bedrock_messages = []

    for msg in dku_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                bedrock_system.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        bedrock_system.append({"text": block["text"]})
            continue

        top_level_content = _convert_top_level_tool_content(msg)
        if top_level_content:
            bedrock_messages.append({"role": _bedrock_role(role), "content": top_level_content})
            continue

        bedrock_content = _convert_content(content)
        if bedrock_content:
            bedrock_messages.append({"role": _bedrock_role(role), "content": bedrock_content})

    return bedrock_messages, bedrock_system


def _bedrock_role(role: str) -> str:
    return "user" if role == "tool" else role


def _convert_top_level_tool_content(msg: dict) -> list:
    if msg.get("role") == "assistant" and msg.get("toolCalls"):
        return [_convert_top_level_tool_call_bedrock(tool_call) for tool_call in msg["toolCalls"]]

    if msg.get("role") == "tool" and msg.get("toolOutputs"):
        return [_convert_top_level_tool_output_bedrock(tool_output) for tool_output in msg["toolOutputs"]]

    return []


def _convert_top_level_tool_call_bedrock(tool_call: dict) -> dict:
    function = tool_call.get("function") or {}
    return {
        "toolUse": {
            "toolUseId": tool_call.get("id", ""),
            "name": function.get("name", ""),
            "input": _parse_json_or_empty(function.get("arguments", "")),
        }
    }


def _convert_top_level_tool_output_bedrock(tool_output: dict) -> dict:
    output = tool_output.get("output", "")
    content = [{"text": output if isinstance(output, str) else json.dumps(output)}]
    return {
        "toolResult": {
            "toolUseId": tool_output.get("callId", ""),
            "content": content,
        }
    }


def _convert_content(content) -> list:
    if isinstance(content, str):
        return [{"text": content}] if content else []

    if not isinstance(content, list):
        return [{"text": str(content)}]

    result = []
    for block in content:
        if isinstance(block, str):
            result.append({"text": block})
            continue

        if not isinstance(block, dict):
            continue

        btype = block.get("type", "")

        if btype == "text":
            result.append({"text": block.get("text", "")})

        elif btype == "tool_use":
            result.append({
                "toolUse": {
                    "toolUseId": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                }
            })

        elif btype == "tool_result":
            raw = block.get("content", "")
            if isinstance(raw, str):
                tc = [{"text": raw}]
            elif isinstance(raw, list):
                tc = [{"text": str(item)} for item in raw]
            else:
                tc = [{"text": str(raw)}]

            tr = {"toolUseId": block.get("tool_use_id", ""), "content": tc}
            if block.get("is_error"):
                tr["status"] = "error"
            result.append({"toolResult": tr})

        elif btype == "image_url":
            # Base64-encoded data URI: data:<mime>;base64,<data>
            url = block.get("image_url", {})
            if isinstance(url, dict):
                url = url.get("url", "")
            if isinstance(url, str) and url.startswith("data:"):
                import base64
                header, data = url.split(",", 1)
                mime = header.split(":")[1].split(";")[0]
                fmt = mime.split("/")[-1].lower()
                fmt = "jpeg" if fmt == "jpg" else fmt
                result.append({
                    "image": {
                        "format": fmt,
                        "source": {"bytes": base64.b64decode(data)},
                    }
                })

    return result


# ---------------------------------------------------------------------------
# Tool definition conversion: Dataiku -> Bedrock toolConfig
# ---------------------------------------------------------------------------

def convert_tools(dku_tools: list) -> list:
    """
    Converts Dataiku tool definitions (OpenAI function-calling format) to
    Bedrock's toolConfig tools list.
    """
    bedrock_tools = []
    for tool in dku_tools:
        if "function" in tool:
            fn = tool["function"]
            name = fn.get("name", "")
            description = fn.get("description", "")
            schema = fn.get("parameters", {})
        else:
            name = tool.get("name", "")
            description = tool.get("description", "")
            schema = tool.get("parameters", tool.get("input_schema", {}))

        bedrock_tools.append({
            "toolSpec": {
                "name": name,
                "description": description,
                "inputSchema": {"json": schema},
            }
        })
    return bedrock_tools


# ---------------------------------------------------------------------------
# Extract tool calls from a Bedrock response content block list
# ---------------------------------------------------------------------------

def extract_tool_calls(content_blocks: list) -> list:
    tool_calls = []
    for block in content_blocks:
        if "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append({
                "type": "function",
                "id": tu.get("toolUseId", ""),
                "function": {
                    "name": tu.get("name", ""),
                    "arguments": json.dumps(tu.get("input", {})),
                },
            })
    return tool_calls


# ---------------------------------------------------------------------------
# Message/tool conversion: Dataiku (OpenAI-like) -> OpenAI Responses API
# ---------------------------------------------------------------------------

def convert_messages_openai(dku_messages: list) -> tuple:
    """
    Converts Dataiku-style messages into (input_items, instructions) for the
    OpenAI Responses API used by bedrock-mantle.
    """
    instructions_parts = []
    input_items = []

    for msg in dku_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            instructions_parts.extend(_extract_system_text(content))
            continue

        extra_items = _convert_openai_top_level_tool_items(msg)
        if extra_items:
            input_items.extend(extra_items)
            continue

        message_content, extra_items = _convert_openai_message_content(role, content)
        if message_content:
            input_items.append({
                "role": role,
                "content": message_content,
            })
        input_items.extend(extra_items)

    instructions = "\n\n".join(part for part in instructions_parts if part).strip()
    return input_items, (instructions or None)


def _convert_openai_top_level_tool_items(msg: dict) -> list:
    role = msg.get("role")

    if role == "assistant" and msg.get("toolCalls"):
        return [_convert_top_level_tool_call_openai(tool_call) for tool_call in msg["toolCalls"]]

    if role == "tool" and msg.get("toolOutputs"):
        return [_convert_top_level_tool_output_openai(tool_output) for tool_output in msg["toolOutputs"]]

    return []


def _convert_top_level_tool_call_openai(tool_call: dict) -> dict:
    function = tool_call.get("function") or {}
    return {
        "type": "function_call",
        "call_id": tool_call.get("id", ""),
        "name": function.get("name", ""),
        "arguments": function.get("arguments", ""),
    }


def _convert_top_level_tool_output_openai(tool_output: dict) -> dict:
    output = tool_output.get("output", "")
    return {
        "type": "function_call_output",
        "call_id": tool_output.get("callId", ""),
        "output": output if isinstance(output, str) else json.dumps(output),
    }


def _extract_system_text(content) -> list:
    if isinstance(content, str):
        return [content]

    if not isinstance(content, list):
        return [str(content)]

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return parts


def _convert_openai_message_content(role: str, content) -> tuple:
    text_type = "output_text" if role == "assistant" else "input_text"

    if isinstance(content, str):
        return ([{"type": text_type, "text": content}] if content else [], [])

    if not isinstance(content, list):
        return ([{"type": text_type, "text": str(content)}], [])

    message_content = []
    extra_items = []
    for block in content:
        if isinstance(block, str):
            message_content.append({"type": text_type, "text": block})
            continue

        if not isinstance(block, dict):
            continue

        btype = block.get("type", "")
        if btype == "text":
            message_content.append({"type": text_type, "text": block.get("text", "")})

        elif btype == "image_url":
            image_url = block.get("image_url", {})
            if isinstance(image_url, dict):
                image_url = image_url.get("url", "")
            if image_url:
                message_content.append({"type": "input_image", "image_url": image_url})

        elif btype == "tool_use":
            extra_items.append({
                "type": "function_call",
                "call_id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": json.dumps(block.get("input", {})),
            })

        elif btype == "tool_result":
            raw = block.get("content", "")
            if isinstance(raw, str):
                output = raw
            elif isinstance(raw, list):
                output = json.dumps(raw)
            else:
                output = json.dumps(raw)
            extra_items.append({
                "type": "function_call_output",
                "call_id": block.get("tool_use_id", ""),
                "output": output,
            })

    return message_content, extra_items


def _parse_json_or_empty(raw: str) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def convert_tools_openai(dku_tools: list) -> list:
    """
    Converts Dataiku/OpenAI-style function tools to Responses API tool specs.
    """
    response_tools = []
    for tool in dku_tools:
        if "function" in tool:
            fn = tool["function"]
            name = fn.get("name", "")
            description = fn.get("description", "")
            schema = fn.get("parameters", {})
        else:
            name = tool.get("name", "")
            description = tool.get("description", "")
            schema = tool.get("parameters", tool.get("input_schema", {}))

        response_tools.append({
            "type": "function",
            "name": name,
            "description": description,
            "parameters": schema,
        })
    return response_tools


def extract_openai_tool_calls(response: dict) -> list:
    tool_calls = []
    for item in response.get("output", []):
        if item.get("type") != "function_call":
            continue
        tool_calls.append({
            "type": "function",
            "id": item.get("call_id") or item.get("id", ""),
            "function": {
                "name": item.get("name", ""),
                "arguments": item.get("arguments", ""),
            },
        })
    return tool_calls


def extract_openai_reasoning(response: dict) -> str:
    parts = []
    for item in response.get("output", []):
        if item.get("type") != "reasoning":
            continue

        for content in item.get("content", []):
            if content.get("type") == "reasoning_text":
                parts.append(content.get("text", ""))

        for summary in item.get("summary", []):
            if summary.get("type") == "summary_text":
                parts.append(summary.get("text", ""))

    return "".join(parts)


def extract_openai_text(response: dict) -> str:
    parts = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            ctype = content.get("type")
            if ctype == "output_text":
                parts.append(content.get("text", ""))
            elif ctype == "refusal":
                parts.append(content.get("refusal", ""))
    return "".join(parts)
