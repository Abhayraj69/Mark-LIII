"""Unit tests for core/tool_schema.py — Gemini function-declaration schema
converted to the OpenAI/Ollama and Anthropic tool-calling wire formats."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.tool_schema import (  # noqa: E402
    gemini_tools_to_anthropic,
    gemini_tools_to_openai,
)

_GEMINI_TOOLS = [
    {
        "name": "move_file",
        "description": "Move a file from one path to another.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "source":      {"type": "STRING", "description": "Source path"},
                "destination": {"type": "STRING", "description": "Destination path"},
                "overwrite":   {"type": "BOOLEAN"},
                "retries":     {"type": "INTEGER"},
                "tags":        {"type": "ARRAY", "items": {"type": "STRING"}},
            },
            "required": ["source", "destination"],
        },
    },
    {
        "name": "shutdown_jarvis",
        "description": "Shut down the assistant.",
        # No "parameters" key at all — declarations for zero-arg tools omit it.
    },
]


class TestGeminiToolsToOpenai(unittest.TestCase):
    def test_wraps_each_tool_in_function_type(self):
        out = gemini_tools_to_openai(_GEMINI_TOOLS)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["type"], "function")
        self.assertEqual(out[0]["function"]["name"], "move_file")

    def test_lowercases_schema_types_recursively(self):
        out = gemini_tools_to_openai(_GEMINI_TOOLS)
        params = out[0]["function"]["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertEqual(params["properties"]["source"]["type"], "string")
        self.assertEqual(params["properties"]["overwrite"]["type"], "boolean")
        self.assertEqual(params["properties"]["retries"]["type"], "integer")
        self.assertEqual(params["properties"]["tags"]["type"], "array")
        self.assertEqual(params["properties"]["tags"]["items"]["type"], "string")

    def test_missing_parameters_defaults_to_empty_object_schema(self):
        out = gemini_tools_to_openai(_GEMINI_TOOLS)
        params = out[1]["function"]["parameters"]
        self.assertEqual(params, {"type": "object", "properties": {}})

    def test_skips_malformed_entries(self):
        out = gemini_tools_to_openai([{"description": "no name"}, None, "junk"])
        self.assertEqual(out, [])

    def test_empty_input(self):
        self.assertEqual(gemini_tools_to_openai([]), [])
        self.assertEqual(gemini_tools_to_openai(None), [])


class TestGeminiToolsToAnthropic(unittest.TestCase):
    def test_flat_shape_no_function_wrapper(self):
        out = gemini_tools_to_anthropic(_GEMINI_TOOLS)
        self.assertEqual(len(out), 2)
        self.assertEqual(set(out[0].keys()), {"name", "description", "input_schema"})
        self.assertEqual(out[0]["name"], "move_file")
        self.assertNotIn("type", out[0])
        self.assertNotIn("function", out[0])

    def test_lowercases_schema_types_recursively(self):
        out = gemini_tools_to_anthropic(_GEMINI_TOOLS)
        schema = out[0]["input_schema"]
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["properties"]["destination"]["type"], "string")
        self.assertEqual(schema["properties"]["tags"]["items"]["type"], "string")
        self.assertEqual(schema["required"], ["source", "destination"])

    def test_missing_parameters_defaults_to_empty_object_schema(self):
        out = gemini_tools_to_anthropic(_GEMINI_TOOLS)
        self.assertEqual(out[1]["input_schema"], {"type": "object", "properties": {}})

    def test_skips_malformed_entries(self):
        out = gemini_tools_to_anthropic([{"description": "no name"}, None, "junk"])
        self.assertEqual(out, [])

    def test_empty_input(self):
        self.assertEqual(gemini_tools_to_anthropic([]), [])
        self.assertEqual(gemini_tools_to_anthropic(None), [])


if __name__ == "__main__":
    unittest.main()
