from typing import TypedDict
from groq import Groq
from dotenv import load_dotenv
import os
import json


def calculator(expression):
    return eval(expression)

from langgraph.graph import StateGraph, START, END


load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform a mathematical calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to calculate."
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

class AgentState(TypedDict):
    messages: list


def call_llm(state: AgentState):

    print("NODE: call_llm")

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=state["messages"],
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    return {
        "messages": [
            message
        ]
    }


def run_tool(state: AgentState):

    print("NODE: run_tool")

    last_message = state["messages"][-1]

    tool_call = last_message.tool_calls[0]

    tool_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    if tool_name == "calculator":
        result = calculator(**arguments)
    else:
        result = f"Unknown tool: {tool_name}"

    print("TOOL RESULT:", result)

    return {
        "messages": [
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": str(result)
            }
        ]
    }

def should_continue(state: AgentState):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tool"

    return "end"

graph = StateGraph(AgentState)

graph.add_node("call_llm", call_llm)
graph.add_node("run_tool", run_tool)

graph.add_edge(START, "call_llm")
graph.add_conditional_edges(
    "call_llm",
    should_continue,
    {
        "tool": "run_tool",
        "end": END
    }
)
graph.add_edge("run_tool", "call_llm")
app = graph.compile()


result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is 25 * 40?"
        }
    ]
})

print("\nFINAL STATE:")
print(result)