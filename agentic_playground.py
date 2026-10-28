"""Weather + Calculator Agent: learn tools, ReAct, state, planning, routing,
reflection, and a supervisor coordinating specialist agents.

Run: python3 agentic_playground.py
Enter a request such as: "What's the weather in Boston, and convert 72 F to C?"
"""

from __future__ import annotations

import ast
import json
import math
import operator
import os
import re

from dotenv import load_dotenv
from groq import Groq

MODEL = "openai/gpt-oss-120b"
MAX_STEPS = 5


# -------------------- The tools: ordinary Python functions --------------------

def get_weather(city: str) -> str:
    """Demo weather tool. It returns sample data, not a live forecast."""
    if not isinstance(city, str):
        raise ValueError("City name must be text.")
    if len(city) > 100:
        raise ValueError("City name is too long.")
    if not city.strip():
        raise ValueError("Please give me a city name.")
    return f"Sample weather for {city}: sunny, 72°F. This is demo data."


# The calculator accepts basic math only. It does not run arbitrary Python code.
_ALLOWED_MATH = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _calculate(node):
    """Read a math expression tree and calculate only approved operations."""
    if isinstance(node, ast.Expression):
        return _calculate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_MATH:
        left = _calculate(node.left)
        right = _calculate(node.right)
        return _ALLOWED_MATH[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _calculate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError("Use numbers and +, -, *, or / only.")


def calculator(expression: str) -> int | float:
    """Safely calculate a small arithmetic expression."""
    if not isinstance(expression, str):
        raise ValueError("The expression must be text.")
    if len(expression) > 100:
        raise ValueError("The expression is too long.")
    try:
        result = _calculate(ast.parse(expression, mode="eval"))
    except (SyntaxError, ZeroDivisionError, OverflowError) as error:
        raise ValueError(f"I could not calculate that: {error}") from error
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("The answer must be a finite number.")
    return result


# This tells the model what tools exist and what arguments each one needs.
# It describes the tools; it does not execute them.
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get sample (not live) weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate using numbers and +, -, *, or /.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


# -------------------- Topic 3: state for this request --------------------

def create_state(user_request: str) -> dict:
    """Make one dictionary that carries the agent's work as it runs."""
    return {
        "user_request": user_request,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a weather and calculator assistant. Use the calculator "
                    "tool for arithmetic. Weather is sample data, not live weather."
                ),
            },
            {"role": "user", "content": user_request},
        ],
        "observations": [],
        "completed_steps": 0,
        "plan": None,
        "route": None,
        "reflection": None,
        "human_approval": None,
        "answer": None,
    }


# -------------------- Topic 5: routing --------------------

def route_request(user_request: str) -> str:
    """Choose which tools fit the request using easy-to-read keyword rules."""
    request = user_request.lower()
    needs_weather = any(word in request for word in ("weather", "temperature", "forecast"))
    needs_calculator = any(word in request for word in (
        "calculate", "compute", "convert", "plus", "minus", "times", "divided",
        "celsius", "fahrenheit", "°c", "°f", "above", "below", "compare",
    )) or bool(re.search(r"\d\s*[+*/-]\s*\d", request))

    if needs_weather and needs_calculator:
        return "both"
    if needs_weather:
        return "weather"
    if needs_calculator:
        return "calculator"
    return "general"


def tools_for_route(route: str) -> list[dict]:
    """Give the model only tools that match the selected route."""
    if route == "weather":
        return [tools[0]]
    if route == "calculator":
        return [tools[1]]
    if route == "both":
        return tools
    return []


def create_plan(state: dict, client) -> str:
    """Ask the model for a short list of actions before any tool is run."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Make a short numbered action plan for the user's request. "
                    "List actions only. Do not solve the request or invent tool results."
                ),
            },
            {"role": "user", "content": state["user_request"]},
        ],
    )
    plan = response.choices[0].message.content or "Use the available tools as needed."

    # Save the plan in state and in messages so the acting model can see it.
    state["plan"] = plan
    state["messages"].append({"role": "assistant", "content": f"Plan:\n{plan}"})
    print("\nPlan:")
    print(plan)
    return plan


def get_client(client=None):
    """Use a supplied client for a demo, or create the real API client."""
    if client is not None:
        return client
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Add GROQ_API_KEY to your .env file first.")
    return Groq(api_key=api_key)


def reflect_on_answer(state: dict, draft_answer: str, client) -> str:
    """Check a draft answer against the request and tool results before returning it."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Review the draft answer using the user's request and tool results. "
                    "Correct any mismatch. Do not invent facts. Weather results are sample data. "
                    "Return only the final answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User request: {state['user_request']}\n"
                    f"Tool results: {json.dumps(state['observations'], ensure_ascii=False)}\n"
                    f"Draft answer: {draft_answer}"
                ),
            },
        ],
    )
    reviewed_answer = response.choices[0].message.content or draft_answer
    state["reflection"] = {
        "draft_answer": draft_answer,
        "reviewed_answer": reviewed_answer,
    }
    return reviewed_answer


# -------------------- Topics 1, 2, and 5: tools, ReAct, and routing --------------------

def run_agent(
    user_request: str,
    client=None,
    max_steps: int = MAX_STEPS,
    plan_first: bool = False,
    route_override: str | None = None,
    system_instruction: str | None = None,
) -> dict:
    """Ask the model, run its tool requests, save results, and repeat."""
    if not user_request.strip():
        raise ValueError("Please enter a request.")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")

    # A fake client can be passed for offline examples.
    client = get_client(client)

    # The same state (including message history) is used on every loop turn.
    state = create_state(user_request)
    messages = state["messages"]
    if system_instruction:
        messages[0]["content"] = system_instruction

    # Route first, then offer only the matching tools to the model.
    state["route"] = route_override or route_request(user_request)
    available_tools = tools_for_route(state["route"])
    allowed_tool_names = {
        item["function"]["name"] for item in available_tools
    }
    print(f"\nRoute selected: {state['route']}")

    # Planning is optional so we can still demonstrate the earlier lessons alone.
    if plan_first:
        create_plan(state, client)

    for step in range(1, max_steps + 1):
        state["completed_steps"] = step

        # Send the conversation and tool descriptions to the model.
        request_options = {
            "model": MODEL,
            "messages": messages,
            "tool_choice": "auto",
        }
        if available_tools:
            request_options["tools"] = available_tools
        response = client.chat.completions.create(**request_options)
        assistant_message = response.choices[0].message

        # No tool request means the assistant is ready to answer the user.
        if not assistant_message.tool_calls:
            draft_answer = assistant_message.content or "No answer was returned."
            state["answer"] = reflect_on_answer(state, draft_answer, client)
            messages.append({"role": "assistant", "content": state["answer"]})
            return state

        # Record the assistant's tool request in the required message format.
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in assistant_message.tool_calls
            ],
        })

        # Run each requested tool, then add its result as a "tool" message.
        for call in assistant_message.tool_calls:
            tool_name = call.function.name
            arguments = {}

            try:
                if tool_name not in allowed_tool_names:
                    raise ValueError(
                        f"Tool '{tool_name}' is not allowed for the {state['route']} route."
                    )
                arguments = json.loads(call.function.arguments)
                if tool_name == "get_weather":
                    result = get_weather(**arguments)
                elif tool_name == "calculator":
                    result = calculator(**arguments)
                else:
                    raise ValueError(f"That tool is not allowed: {tool_name}")
            except (TypeError, ValueError, ArithmeticError) as error:
                result = f"Tool error: {error}"

            result = str(result)
            print(f"\nAssistant requested tool: {tool_name}({arguments})")
            print(f"Tool returned: {result}")

            state["observations"].append({"tool": tool_name, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": tool_name,
                "content": result,
            })

    # Stop the tool loop at the limit. Ask for a summary without offering tools.
    final_response = client.chat.completions.create(model=MODEL, messages=messages)
    draft_answer = final_response.choices[0].message.content or "Step limit reached."
    state["answer"] = reflect_on_answer(state, draft_answer, client)
    messages.append({"role": "assistant", "content": state["answer"]})
    print(f"\nAgent stopped after {max_steps} tool steps.")
    return state


# -------------------- Named versions of the learning steps --------------------

def run_tool_calling(user_request: str, client=None) -> str:
    """Step 1: run one tool round."""
    result = run_agent(user_request, client=client, max_steps=1)
    return result["answer"]


def run_react(user_request: str, client=None) -> dict:
    """Step 2: allow repeated tool rounds, up to MAX_STEPS."""
    return run_agent(user_request, client=client, max_steps=MAX_STEPS)


def run_planned_agent(user_request: str, client=None) -> dict:
    """Make a plan, route to tools, then use the ReAct loop to complete it."""
    return run_agent(user_request, client=client, plan_first=True)


# -------------------- Topic 7: specialist agents + a supervisor --------------------

def run_weather_agent(user_request: str, client=None) -> dict:
    """A weather specialist that can only use the weather tool."""
    return run_agent(
        user_request,
        client=client,
        route_override="weather",
        system_instruction=(
            "You are the weather specialist. Use the weather tool for weather facts. "
            "The tool returns sample data, not live weather. Do not do calculations."
        ),
    )


def run_calculator_agent(user_request: str, client=None) -> dict:
    """A math specialist that can only use the calculator tool."""
    return run_agent(
        user_request,
        client=client,
        route_override="calculator",
        system_instruction=(
            "You are the calculator specialist. Use the calculator tool for arithmetic. "
            "Use only the numbers provided in the request."
        ),
    )


def run_supervisor(user_request: str, client=None, plan_first: bool = False) -> dict:
    """Optionally plan, then choose one specialist or coordinate both in order."""
    route = route_request(user_request)
    print(f"\nSupervisor selected: {route}")

    plan = None
    if plan_first:
        client = get_client(client)
        planning_state = create_state(user_request)
        plan = create_plan(planning_state, client)
        specialist_request = f"User request: {user_request}\nPlan:\n{plan}"
    else:
        specialist_request = user_request

    if route == "weather":
        result = run_weather_agent(specialist_request, client=client)
        result["user_request"] = user_request
        result["plan"] = plan
        result["specialists"] = ["weather"]
        return result

    if route == "calculator":
        result = run_calculator_agent(specialist_request, client=client)
        result["user_request"] = user_request
        result["plan"] = plan
        result["specialists"] = ["calculator"]
        return result

    if route == "both":
        # First, the weather specialist gathers the temperature.
        weather_state = run_weather_agent(specialist_request, client=client)
        weather_results = weather_state["observations"]

        # Then, the calculator specialist uses that observation for the math.
        calculator_request = (
            f"Original user request: {user_request}\n"
            f"Plan: {plan or 'Use the request to determine the needed steps.'}\n"
            f"Weather specialist result: {json.dumps(weather_results, ensure_ascii=False)}\n"
            "Use this weather result to complete any requested conversion or comparison."
        )
        result = run_calculator_agent(calculator_request, client=client)
        result["user_request"] = user_request
        result["route"] = "both"
        result["plan"] = plan
        result["observations"] = weather_results + result["observations"]
        result["specialists"] = ["weather", "calculator"]
        return result

    # For a general request, let the main assistant answer without tools.
    result = run_agent(specialist_request, client=client, route_override="general")
    result["user_request"] = user_request
    result["plan"] = plan
    result["specialists"] = ["general assistant"]
    return result


# -------------------- Topic 8: human review checkpoint --------------------

def ask_human_to_approve(state: dict, input_fn=input) -> bool:
    """Pause and let a person approve or reject the proposed final answer."""
    while True:
        choice = input_fn("Approve this answer? (y/n): ").strip().lower()
        if choice in ("y", "yes"):
            state["human_approval"] = "approved"
            return True
        if choice in ("n", "no"):
            state["human_approval"] = "rejected"
            return False
        print("Please enter y or n.")


# -------------------- Start the program --------------------

if __name__ == "__main__":
    request = input("What would you like help with? ").strip()
    result = run_supervisor(request, plan_first=True)

    print("\nMessage roles used:")
    print("system -> user -> assistant (tool request) -> tool (result) -> assistant (answer)")
    print(f"Route used: {result['route']}")
    print(f"Specialists used: {', '.join(result['specialists'])}")
    print("\nDraft answer for your review:")
    print(result["answer"])
    if ask_human_to_approve(result):
        print("\nApproved answer:")
        print(result["answer"])
    else:
        print("\nAnswer not approved. Update the request and run the agent again.")
