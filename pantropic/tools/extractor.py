"""Pantropic - Tool Extraction.

Extracts tool calls from model output.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from pantropic.observability.logging import get_logger

log = get_logger("tools.extractor")


class ToolExtractor:
    """Extract tool calls from model output.

    Supports multiple formats used by different models.
    """

    @staticmethod
    def extract(content: str, architecture: str) -> list[dict[str, Any]] | None:
        """Extract tool calls from content.

        Args:
            content: Model output
            architecture: Model architecture

        Returns:
            List of tool calls or None if no tools found
        """
        if not content or not content.strip():
            return None

        arch = architecture.lower()

        # Try architecture-specific extraction first
        if "qwen" in arch:
            result = ToolExtractor._extract_qwen(content)
            if result:
                return result

        if "llama" in arch:
            result = ToolExtractor._extract_llama(content)
            if result:
                return result

        # Try generic JSON extraction
        return ToolExtractor._extract_json(content)

    @staticmethod
    def _extract_qwen(content: str) -> list[dict[str, Any]] | None:
        """Extract Qwen-style tool calls (<tool_call> tags)."""
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        matches = re.findall(pattern, content, re.DOTALL)

        if not matches:
            return None

        calls = []
        for match in matches:
            try:
                data = json.loads(match)
                calls.append(ToolExtractor._normalize_call(data))
            except json.JSONDecodeError:
                continue

        return calls if calls else None

    @staticmethod
    def _extract_llama(content: str) -> list[dict[str, Any]] | None:
        """Extract Llama-style function_call objects."""
        # Look for {"function_call": {...}}
        pattern = r'\{\s*"function_call"\s*:\s*(\{.*?\})\s*\}'
        matches = re.findall(pattern, content, re.DOTALL)

        if matches:
            calls = []
            for match in matches:
                try:
                    func_call = json.loads(match)
                    calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": func_call.get("name", ""),
                            "arguments": json.dumps(func_call.get("arguments", {})),
                        },
                    })
                except json.JSONDecodeError:
                    continue
            return calls if calls else None

        return None

    @staticmethod
    def _extract_json(content: str) -> list[dict[str, Any]] | None:
        """Extract generic JSON tool calls."""
        # Try to find JSON objects with name and arguments
        patterns = [
            r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}',
            r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{[^{}]*\}[^{}]*\}',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                calls = []
                for match in matches:
                    try:
                        data = json.loads(match)
                        calls.append(ToolExtractor._normalize_call(data))
                    except json.JSONDecodeError:
                        continue
                if calls:
                    return calls

        # Try parsing entire content as JSON
        try:
            data = json.loads(content.strip())
            if isinstance(data, dict) and "name" in data:
                return [ToolExtractor._normalize_call(data)]
            if isinstance(data, list):
                calls = [ToolExtractor._normalize_call(d) for d in data if isinstance(d, dict) and "name" in d]
                return calls if calls else None
        except json.JSONDecodeError:
            pass

        return None

    @staticmethod
    def _normalize_call(data: dict[str, Any]) -> dict[str, Any]:
        """Normalize tool call to OpenAI format."""
        args = data.get("arguments") or data.get("parameters") or {}
        if isinstance(args, dict):
            args = json.dumps(args)

        return {
            "id": data.get("id") or f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": data.get("name", ""),
                "arguments": args,
            },
        }
