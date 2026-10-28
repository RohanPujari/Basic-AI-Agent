"""Small offline evaluations for the Weather + Calculator agent.

Run: python3 evaluate_agent.py
These checks use a fake LLM response, so they need no API key or live weather.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agentic_playground import calculator, get_weather, run_agent, run_supervisor, route_request


def tool_call(call_id: str, name: str, arguments: dict):
    """Build the shape of a tool request returned by the LLM."""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def model_reply(text: str | None = None, tool_calls=None):
    """Build the small part of an LLM response that our agent reads."""
    message = SimpleNamespace(content=text, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    """Return saved model responses and remember what the agent sent."""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return next(self.replies)


class AgentEvaluations(unittest.TestCase):
    def test_router_picks_expected_route(self):
        examples = [
            ("What's the weather in Boston?", "weather"),
            ("Calculate 25*40", "calculator"),
            ("What's the weather in Boston in Celsius?", "both"),
        ]
        for request, expected_route in examples:
            with self.subTest(request=request):
                self.assertEqual(route_request(request), expected_route)

    def test_calculator_converts_fahrenheit_to_celsius(self):
        result = calculator("(72-32)*5/9")
        self.assertAlmostEqual(result, 22.2222, places=3)

    def test_supervisor_hands_weather_result_to_calculator(self):
        replies = [
            model_reply(tool_calls=[tool_call("weather-1", "get_weather", {"city": "Boston"})]),
            model_reply("Sunny, 72°F."),
            model_reply("Weather answer checked."),
            model_reply(tool_calls=[tool_call("calculator-1", "calculator", {"expression": "(72-32)*5/9"})]),
            model_reply("22.2°C is below 25°C."),
            model_reply("72°F is 22.2°C, below 25°C. This is sample data."),
        ]
        completions = FakeCompletions(replies)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        result = run_supervisor(
            "What's the weather in Boston in Celsius, and is it below 25?",
            client=client,
        )

        self.assertEqual(result["specialists"], ["weather", "calculator"])
        self.assertEqual(
            [item["tool"] for item in result["observations"]],
            ["get_weather", "calculator"],
        )
        self.assertIn("22.2°C", result["answer"])
        self.assertIn("below 25°C", result["answer"])

        # Check what tools each specialist was allowed to see.
        weather_tools = completions.requests[0]["tools"]
        calculator_tools = completions.requests[3]["tools"]
        self.assertEqual([item["function"]["name"] for item in weather_tools], ["get_weather"])
        self.assertEqual([item["function"]["name"] for item in calculator_tools], ["calculator"])
        self.assertIn("72°F", completions.requests[3]["messages"][1]["content"])

    def test_tool_outside_route_is_blocked(self):
        replies = [
            model_reply(tool_calls=[tool_call("unexpected", "get_weather", {"city": "Boston"})]),
            model_reply("The unapproved tool was blocked."),
            model_reply("Reviewed: the tool request was blocked."),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(replies)))

        with patch("agentic_playground.get_weather") as weather_tool:
            state = run_agent("Calculate 2+2", client=client)

        weather_tool.assert_not_called()
        self.assertIn("not allowed for the calculator route", state["observations"][0]["result"])

    def test_weather_input_length_is_limited(self):
        with self.assertRaisesRegex(ValueError, "too long"):
            get_weather("x" * 101)


if __name__ == "__main__":
    unittest.main(verbosity=2)
