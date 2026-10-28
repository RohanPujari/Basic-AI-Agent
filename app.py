# # from tools import calculator

# # result  =calculator("25*40")
# # print(result)
# # ------------------------------------------------------------

# import os
# from dotenv import load_dotenv
# from groq import Groq
# import json
# from tools import calculator, get_weather

# load_dotenv()

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "calculator",
#             "description": "Calculate a mathematical expression",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "expression": {
#                         "type": "string",
#                         "description": "The mathematical expression to calculate"
#                     }
#                 },
#                 "required": ["expression"]
#             }
#         }
#     },

#     {
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "Get the current weather for a city",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {
#                     "type": "string",
#                     "description": "The name of the city"
#                 }
#             },
#             "required": ["city"]
#         }
#     }
# }
# ]

# # ------------------------------------------------------------ Tools sending to LLM
# # initial user message:
# messages=[
#         {
#             "role": "user",
#             "content": "What's the weather in New York City and what is 25*40?"
#         }
#     ]
# response = client.chat.completions.create(
#     model="openai/gpt-oss-120b",
#     messages=messages,
#     tools=tools,
#     tool_choice="auto"
# )


# message = response.choices[0].message

# print("CONTENT:", message.content)
# print("TOOL CALLS:", message.tool_calls)

# for tool_call in message.tool_calls:
#     tool_name = tool_call.function.name

# # assistant's tool call response:
# messages.append({
#     "role": "assistant",
#     "content": message.content,
#     "tool_calls": [
#         {
#             "id": tool_call.id,
#             "type": "function",
#             "function": {
#                 "name": tool_call.function.name,
#                 "arguments": tool_call.function.arguments
#             }
#         }
#     ]
# })
# # tool result:
# messages.append({
#     "role": "tool",
#     "tool_call_id": tool_call.id,
#     "name": tool_call.function.name,
#     "content": str(result)
# })

# # messages
# #    │
# #    ├── user
# #    │
# #    ├── assistant → calculator(...)
# #    │
# #    └── tool → 83810205
# # ------------------------------------------------------------ Tools sending to LLM

# print("TOOL RESULT:", result)
# print("--------------------------------")

# response2 = client.chat.completions.create(
#     model="openai/gpt-oss-120b",
#     messages=messages,
#     tools=tools,
#     tool_choice="auto"
# )



# print(get_weather("New York City"))



# from tools import calculator

# result = calculator("25*40")
# print(result)
# ------------------------------------------------------------

import os
from dotenv import load_dotenv
from groq import Groq
import json
from tools import calculator, get_weather, broken_tool

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to calculate"
                    }
                },
                "required": ["expression"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city"
                    }
                },
                "required": ["city"]
            }
        }
    },

    {
    "type": "function",
    "function": {
        "name": "broken_tool",
        "description": "A tool that intentionally fails for testing error recovery",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
}
]

# ------------------------------------------------------------ Tools sending to LLM

# initial user message:
messages = [
    {
        "role": "system",
        "content": "When a calculation is required, always use the calculator tool. Do not perform arithmetic yourself."
    },
    {
        "role": "user",
        "content": " What is the weather in New York? If the temperature is in Fahrenheit, convert it to Celsius. Then calculate how many degrees warmer 25°C is than the converted temperature."
    }
]


# print("CONTENT:", message.content)
# print("TOOL CALLS:", message.tool_calls)

# ------------------------------------------------------------

# Process every tool the LLM requested
MAX_ITERATIONS = 2
iteration = 0

while iteration < MAX_ITERATIONS:
    iteration += 1

    # Send conversation to LLM
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # If LLM has no more tools to call, we have the final answer
    if not message.tool_calls:
        print("FINAL ANSWER:", message.content)
        break

    # Add the assistant's tool call to conversation
    messages.append({
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            }
            for tool_call in message.tool_calls
        ]
    })

    # Execute every tool requested by the LLM
    for tool_call in message.tool_calls:

        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        try:
            if tool_name == "calculator":
                result = calculator(**arguments)

            elif tool_name == "get_weather":
                result = get_weather(**arguments)

            elif tool_name == "broken_tool":
                result = broken_tool()

            else:
                result = f"Unknown tool: {tool_name}"

        except Exception as e:
            result = f"Tool failed: {str(e)}"

        print("TOOL:", tool_name)
        print("RESULT:", result)

        # Add tool result to conversation
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_name,
            "content": str(result)
        })
else:
    print("Agent stopped: maximum iterations reached.")
# messages
#    │
#    ├── user
#    │
#    ├── assistant → calculator(...) / get_weather(...)
#    │
#    └── tool → results

# ------------------------------------------------------------ Tools sending to LLM


# print("--------------------------------")
# print("FINAL CONTENT:", response2.choices[0].message.content)
# print("FINAL TOOL CALLS:", response2.choices[0].message.tool_calls)