"""Pantropic - Tool Injection.

Injects tool definitions into prompts for different model architectures.
"""

from __future__ import annotations

import json
from typing import Any

from pantropic.observability.logging import get_logger

log = get_logger("tools.injector")


class ToolInjector:
    """Inject tool definitions into prompts.

    Supports multiple model architectures with appropriate formatting.
    """

    @staticmethod
    def inject(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        architecture: str,
    ) -> list[dict[str, Any]]:
        """Inject tools into messages.

        Args:
            messages: Chat messages
            tools: Tool definitions
            architecture: Model architecture

        Returns:
            Messages with tool instructions injected
        """
        if not tools:
            return messages

        arch = architecture.lower()

        # Select injection strategy
        if "qwen" in arch:
            return ToolInjector._inject_qwen(messages, tools)
        if "deepseek" in arch:
            return ToolInjector._inject_deepseek(messages, tools)
        if "llama" in arch:
            return ToolInjector._inject_llama(messages, tools)
        return ToolInjector._inject_generic(messages, tools)

    @staticmethod
    def _inject_generic(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generic tool injection using JSON format."""
        tool_docs = []
        for tool in tools:
            func = tool.get("function", tool)
            tool_docs.append({
                "name": func["name"],
                "description": func["description"],
                "parameters": func.get("parameters", {}),
            })

        instruction = f"""You have access to the following tools:

{json.dumps(tool_docs, indent=2)}

To use a tool, respond with ONLY a JSON object in this exact format:
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}

Do not include any other text before or after the JSON."""

        # Add to system message or create one
        return ToolInjector._update_system(messages, instruction)

    @staticmethod
    def _inject_qwen(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Qwen-style tool injection with XML tags."""
        tool_xml = "<tools>\n"
        for tool in tools:
            func = tool.get("function", tool)
            tool_xml += f"""<tool>
<name>{func["name"]}</name>
<description>{func["description"]}</description>
<parameters>{json.dumps(func.get("parameters", {}))}</parameters>
</tool>
"""
        tool_xml += "</tools>"

        instruction = f"""{tool_xml}

When you need to call a tool, use this format:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg": "value"}}}}
</tool_call>"""

        return ToolInjector._update_system(messages, instruction)

    @staticmethod
    def _inject_deepseek(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """DeepSeek R1 style tool injection."""
        tool_docs = []
        for tool in tools:
            func = tool.get("function", tool)
            tool_docs.append({
                "name": func["name"],
                "description": func["description"],
                "parameters": func.get("parameters", {}),
            })

        instruction = f"""Available tools:
{json.dumps(tool_docs, indent=2)}

When calling a tool, output ONLY a valid JSON object:
{{"name": "function_name", "arguments": {{"param": "value"}}}}

No other text or explanation."""

        return ToolInjector._update_system(messages, instruction)

    @staticmethod
    def _inject_llama(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Llama 3 style tool injection."""
        tool_docs = []
        for tool in tools:
            func = tool.get("function", tool)
            tool_docs.append({
                "name": func["name"],
                "description": func["description"],
                "parameters": func.get("parameters", {}),
            })

        instruction = f"""You have access to these functions:

{json.dumps(tool_docs, indent=2)}

To call a function, respond with a JSON object:
{{"function_call": {{"name": "func_name", "arguments": {{"key": "value"}}}}}}"""

        return ToolInjector._update_system(messages, instruction)

    @staticmethod
    def _update_system(
        messages: list[dict[str, Any]],
        addition: str,
    ) -> list[dict[str, Any]]:
        """Add content to system message."""
        result = list(messages)

        # Find or create system message
        for i, msg in enumerate(result):
            if msg.get("role") == "system":
                result[i] = {
                    **msg,
                    "content": f"{msg.get('content', '')}\n\n{addition}",
                }
                return result

        # No system message, add one
        result.insert(0, {"role": "system", "content": addition})
        return result
