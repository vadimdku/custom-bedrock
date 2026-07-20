# Amazon Bedrock Dataiku LLM Plugin

This is a custom Dataiku DSS plugin that lets LLM Mesh connect to Amazon Bedrock-hosted chat models.

The plugin provides custom LLM connection types for Bedrock models exposed through AWS Bedrock Runtime and Bedrock Mantle. It is intended for Dataiku environments that need to make Bedrock-hosted models available through the standard DSS chat-completion experience.

At a high level, the plugin handles:

- Dataiku custom LLM integration.
- Chat-completion requests and streaming responses.
- Conversion between Dataiku LLM messages and Bedrock-compatible request formats.
- Model-specific configuration through the DSS custom LLM connection UI.
- Basic usage and estimated-cost reporting where token usage is available.

The plugin code environment is defined under `code-env/python`, and the LLM implementations are under `python-llms`.

Install the plugin in DSS, create the plugin code environment, then create a custom LLM connection using the `Amazon Bedrock` plugin.
