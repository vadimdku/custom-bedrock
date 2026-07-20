# Amazon Bedrock Dataiku LLM Plugin

Custom Dataiku LLM plugin for connecting DSS LLM Mesh to Amazon Bedrock models.

The plugin provides two chat-completion implementations:

- **Amazon Bedrock via Converse API** for standard Bedrock Runtime models.
- **Amazon Bedrock via Mantle API** for Bedrock Mantle models exposed through OpenAI-compatible endpoints.

## Plugin Layout

```text
custom-bedrock/
├── plugin.json
├── code-env/python/
├── python-lib/custombedrock/
├── python-llms/bedrock-via-converse/
└── python-llms/bedrock-via-mantle/
```

## Requirements

- Dataiku DSS with custom Python LLM support.
- Python 3.9 through 3.12 in the plugin code environment.
- `boto3>=1.34.0`.
- Network access from the DSS backend/code-env runtime to the relevant Bedrock endpoints.
- A runtime trust store that can validate AWS TLS certificates.

The plugin does not override TLS verification or ship a custom CA bundle. Certificate trust should be fixed at the DSS host, container, or code-environment level.

## LLM Types

### Amazon Bedrock via Converse API

Use this type for standard Bedrock Runtime models that support the Bedrock Converse API.

Configuration:

- `S3 connection`: Optional Dataiku connection used to resolve AWS credentials.
- `Region`: AWS region for Bedrock Runtime. Defaults to `us-east-1`.
- `Inference profile`: Optional cross-region inference profile prefix, such as `us`, `eu`, or `apac`.
- `Max parallelism`: Maximum concurrent calls for this model definition.
- `Bedrock Guardrail`: Optional Bedrock guardrail settings.
- `Model`: Bedrock model ID.

Credential behavior:

- If `S3 connection` is set, the plugin uses the AWS credentials from that Dataiku connection.
- If `S3 connection` is empty, the plugin falls back to the ambient Boto3 credential chain available to the DSS runtime.

Supported behavior:

- Chat completion.
- Streaming completion.
- System prompts.
- Tool/function definitions and tool-call extraction.
- Basic image data URI conversion for models that support image content through Converse.
- Estimated cost calculation for models listed in the plugin pricing map.

### Amazon Bedrock via Mantle API

Use this type for Bedrock Mantle models exposed through OpenAI-compatible APIs.

Configuration:

- `Region`: AWS region for the Bedrock Mantle endpoint. Defaults to `us-west-2`.
- `Bedrock API key`: Short-term Bedrock API key.
- `Max parallelism`: Maximum concurrent calls for this model definition.
- `Model`: Mantle model ID.

Credential behavior:

- Mantle uses the configured `Bedrock API key` directly.
- The API key field is a Dataiku password parameter.

Supported Mantle models:

| Model label | Model ID | Prompt price | Completion price |
| --- | --- | ---: | ---: |
| OpenAI GPT-5.4 | `openai.gpt-5.4` | $2.75 / 1M tokens | $16.50 / 1M tokens |
| NVIDIA Nemotron 3 Super 120B | `nvidia.nemotron-super-3-120b` | $0.18 / 1M tokens | $0.78 / 1M tokens |
| NVIDIA Nemotron Nano 3 30B | `nvidia.nemotron-nano-3-30b` | $0.07 / 1M tokens | $0.29 / 1M tokens |
| OpenAI GPT OSS 20B | `openai.gpt-oss-20b` | $0.08 / 1M tokens | $0.36 / 1M tokens |
| OpenAI GPT OSS 120B | `openai.gpt-oss-120b` | $0.18 / 1M tokens | $0.72 / 1M tokens |
| NVIDIA Nemotron Nano 12B v2 VL BF16 | `nvidia.nemotron-nano-12b-v2` | $0.24 / 1M tokens | $0.72 / 1M tokens |
| NVIDIA Nemotron Nano 9B v2 | `nvidia.nemotron-nano-9b-v2` | $0.07 / 1M tokens | $0.28 / 1M tokens |

Endpoint behavior:

- `openai.gpt-5.4` uses `https://bedrock-mantle.<region>.api.aws/openai/v1/responses`.
- Other Mantle models use `https://bedrock-mantle.<region>.api.aws/v1/chat/completions`.

Supported behavior:

- Chat completion.
- Streaming completion.
- Multi-turn chat history.
- Tool/function definitions and tool-call extraction.
- Estimated cost calculation using per-1K token rates derived from the pricing table above.

## Installation

Install or upload this directory as a Dataiku plugin, then create a DSS code environment for the plugin.

The code environment is defined in:

```text
code-env/python/desc.json
code-env/python/spec/requirements.txt
```

The only explicit Python dependency is:

```text
boto3>=1.34.0
```

## Creating a Dataiku LLM Connection

In DSS, create a custom LLM connection and choose the `Amazon Bedrock` plugin.

For standard Bedrock models:

1. Set `Capability` to `Chat completion`.
2. Set `Type` to `Amazon Bedrock via Converse API`.
3. Configure `Region`, optional `S3 connection`, optional inference profile, and model.
4. Test the connection.

For Bedrock Mantle models:

1. Set `Capability` to `Chat completion`.
2. Set `Type` to `Amazon Bedrock via Mantle API`.
3. Configure `Region`, `Bedrock API key`, and model.
4. Test the connection.

Although Mantle may use OpenAI `Responses` internally for some models, the Dataiku-facing capability remains `Chat completion`.

## Local Test Scripts

Standalone scripts are provided under `python-lib/`.

Converse local test:

```bash
cd /path/to/custom-bedrock
PYTHONPATH=python-lib:python-llms/bedrock-via-converse \
python python-lib/test_llm.py
```

Mantle local test:

```bash
cd /path/to/custom-bedrock
export BEDROCK_API_KEY='...'
PYTHONPATH=python-lib:python-llms/bedrock-via-mantle \
python python-lib/test_llm_mantle.py
```

The test scripts mock the Dataiku `BaseLLM` class so they can be run outside DSS.

## Troubleshooting

### SSL certificate verification failures

Errors like this are environment-level trust-store problems:

```text
CERTIFICATE_VERIFY_FAILED: certificate verify failed: unable to get local issuer certificate
```

Fix the DSS runtime environment so Python and Boto3 trust the certificate chain presented by AWS or by any corporate TLS inspection proxy. Check the host/container CA certificates and environment variables such as `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, and `AWS_CA_BUNDLE`.

### Empty response from Mantle models

Mantle models do not all use the same OpenAI-compatible API path. This plugin routes:

- GPT-5.4 to the Responses API.
- GPT OSS and NVIDIA models to Chat Completions.

If a Mantle model returns an empty response after the first turn, verify the selected model ID and confirm that the deployed plugin includes the current Mantle routing logic.

### Missing API key

For Mantle connections, set the `Bedrock API key` field directly in the DSS custom LLM connection. The plugin does not read API keys from environment variables in DSS.

### Converse authentication failures

For Converse connections, verify either:

- The configured Dataiku S3 connection exposes valid AWS credentials with Bedrock permissions.
- The DSS backend runtime has ambient AWS credentials available through the Boto3 credential chain.

## Notes

- Pricing is used only for Dataiku estimated-cost reporting.
- Prices can vary by region and AWS account terms.
- Model availability depends on the target AWS account, region, and Bedrock access settings.
