import utils
from pprint import pprint
import random
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    upper: int
    lower: int
    guess: int
    target: int
    attempts: int
    history: List[dict]


def setup(state: AgentState) -> AgentState:
    
    state["attempts"] = 0
    state["upper"] = 20
    state["lower"] = 1
    state["target"] = random.randint(1, 20)
    state["history"] = []
    return state

def make_guess(state: AgentState) -> AgentState:

    # Make guess
    state["guess"] = random.randint(state["lower"], state["upper"])
    state["attempts"] += 1
    if state["guess"] < state["target"]:
        state["lower"] = state["guess"]+1
    elif state["guess"] > state["target"]:
        state["upper"] = state["guess"]-1
    state["history"].append({"guess": state["guess"], "attempts": state["attempts"], "lower": state["lower"], "upper": state["upper"]})
    return state


def guess_loop(state: AgentState) -> AgentState:

    # Check guess
    if state["guess"] == state["target"]:
        return "exit"
    else:
        RuntimeError("Unexpected behaviour")
    return "continue"

# Construct Graph
graph = StateGraph(AgentState)

## Add Nodes
graph.add_node("setup", setup)
graph.add_node("guessing", make_guess)

## Add edges
graph.add_edge(START,"setup")
graph.add_edge("setup", "guessing")
graph.add_conditional_edges(
    "guessing",
    guess_loop,
    {
        "continue": "guessing",
        "exit": END
    }
)
app = graph.compile()


result = app.invoke({})
pprint(result["history"])

print(f"The correct number was {result["target"]} and you got it in {result["attempts"]} attempts")