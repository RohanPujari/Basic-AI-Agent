"""A tiny LangGraph example using Mario's game state."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class MarioState(TypedDict):
    lives: int
    coins: int
    level: int


def play_level(state: MarioState) -> MarioState:
    """Mario completes one level and collects a coin."""
    return {
        "lives": state["lives"],
        "coins": state["coins"] + 1,
        "level": state["level"] + 1,
    }


graph_builder = StateGraph(MarioState)
graph_builder.add_node("play_level", play_level)
graph_builder.add_edge(START, "play_level")
graph_builder.add_edge("play_level", END)

graph = graph_builder.compile()

initial_state: MarioState = {"lives": 3, "coins": 0, "level": 0}
final_state = graph.invoke(initial_state)

print("Before:", initial_state)
print("After: ", final_state)
