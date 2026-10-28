from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# 1. Define the state
class AgentState(TypedDict):
    city: str
    temperature_f: float
    temperature_c: float
    difference: float


# 2. Node: Get weather
def get_weather(state: AgentState):
    print("NODE: get_weather")

    return {
        "temperature_f": 95
    }


# 3. Node: Convert Fahrenheit to Celsius
def convert_to_celsius(state: AgentState):
    print("NODE: convert_to_celsius")

    temperature_f = state["temperature_f"]

    temperature_c = (temperature_f - 32) * 5 / 9

    return {
        "temperature_c": temperature_c
    }


# 4. Node: Calculate difference
def calculate_difference(state: AgentState):
    print("NODE: calculate_difference")

    difference = 25 - state["temperature_c"]

    return {
        "difference": difference
    }


def hot_weather(state: AgentState):
    print("NODE: hot_weather")
    return {
        "difference": 999
    }


def cold_weather(state: AgentState):
    print("NODE: cold_weather")
    return {
        "difference": 25 - state["temperature_c"]
    }

def freezing_weather(state: AgentState):
    print("NODE: freezing_weather")

    return {
        "difference": 0
    }

def temperature_router(state: AgentState):
    if state["temperature_c"] > 25:
        return "hot"
    elif state["temperature_c"] < 0:
        return "freezing"
    else:
        return "cold"

# 5. Create the graph
graph = StateGraph(AgentState)


# 6. Add nodes
graph.add_node("get_weather", get_weather)
graph.add_node("convert_to_celsius", convert_to_celsius)
# graph.add_node("calculate_difference", calculate_difference)
graph.add_node("hot_weather", hot_weather)
graph.add_node("cold_weather", cold_weather)
graph.add_node("freezing_weather", freezing_weather)


# 7. Connect the nodes
graph.add_edge(START, "get_weather")
graph.add_edge("get_weather", "convert_to_celsius")
graph.add_conditional_edges(
    "convert_to_celsius",
    temperature_router,
    {
        "hot": "hot_weather",
        "cold": "cold_weather",
        "freezing": "freezing_weather"
    }
)
graph.add_edge("hot_weather", END)
graph.add_edge("cold_weather", END)
graph.add_edge("freezing_weather", END)
# graph.add_edge("calculate_difference", END)


# 8. Compile the graph
app = graph.compile()


# 9. Run the graph
result = app.invoke({
    "city": "New York",
    "temperature_f": 0,
    "temperature_c": 0,
    "difference": 0
})


# 10. Print final state
print("\nFINAL STATE:")
print(result)