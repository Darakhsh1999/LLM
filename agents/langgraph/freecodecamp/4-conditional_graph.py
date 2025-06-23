import utils
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    operand1: int
    operand2: int
    operand3: int
    operand4: int
    operation1: str
    operation2: str
    result: int
    result2: int

def adder(state: AgentState) -> AgentState:
    state["result"] = state["operand1"] + state["operand2"]
    return state

def adder2(state: AgentState) -> AgentState:
    state["result2"] = state["operand3"] + state["operand4"]
    return state

def subtractor(state: AgentState) -> AgentState:
    state["result"] = state["operand1"] - state["operand2"]
    return state

def subtractor2(state: AgentState) -> AgentState:
    state["result2"] = state["operand3"] - state["operand4"]
    return state


def operation_router(state: AgentState) -> AgentState:

    if state["operation1"] == "+":
        return "addition_operation"
    elif state["operation1"] == "-":
        return "subtraction_operation"
    else:
        RuntimeError(f"Unavailable operation {state['operation']}")

def operation_router2(state: AgentState) -> AgentState:

    if state["operation2"] == "+":
        return "addition_operation2"
    elif state["operation2"] == "-":
        return "subtraction_operation2"
    else:
        RuntimeError(f"Unavailable operation {state['operation']}")


# Construct Graph
graph = StateGraph(AgentState)

## Add Nodes
graph.add_node("add_node", adder)
graph.add_node("sub_node", subtractor)
graph.add_node("add_node2", adder2)
graph.add_node("sub_node2", subtractor2)
graph.add_node("router", lambda state: state)
graph.add_node("router2", lambda state: state)

## Add edges
graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    operation_router,
    { # Edge -> Node
        "addition_operation": "add_node",
        "subtraction_operation": "sub_node",
    }
)
graph.add_edge("add_node", "router2")
graph.add_edge("sub_node", "router2")
graph.add_conditional_edges(
    "router2",
    operation_router2,
    { # Edge -> Node
        "addition_operation2": "add_node2",
        "subtraction_operation2": "sub_node2",
    }
)
graph.add_edge("add_node2", END)
graph.add_edge("sub_node2", END)

app = graph.compile()


result = app.invoke({"operand1": 10, "operand2": 30, "operand3": 5, "operand4": 10, "operation1": "+", "operation2": "-"})
print(result)
utils.convert_to_png(app, "conditional_graph_v2")

print(result["result"], result["result2"])