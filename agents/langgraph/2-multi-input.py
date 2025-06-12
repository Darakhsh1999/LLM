from typing import TypedDict, List
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    values: List[int]
    name: str
    operation: str
    result: str


def process_elements(state: AgentState) -> AgentState:
    """ Sums the integers to the user name

    Args:
        state (AgentState): state object

    Returns:
        AgentState: state object
    """


    if state["operation"] == "+":
        state["result"] = f"Hello {state['name']}, the sum of your values are {sum(state["values"])}"
    elif state["operation"] == "*":
        res = 1
        for element in state["values"]:
            res *= element
        state["result"] = f"Hello {state['name']}, the sum of your values are {res}"
    else:
        state["result"] = f"Unknown operation: '{state["operation"]}'"
    return state



# Construct Graph
graph = StateGraph(AgentState)

## Add Nodes
graph.add_node("summer", process_elements)
graph.set_entry_point("summer")
graph.set_finish_point("summer")

app = graph.compile()




result = app.invoke({"name": "Nicólo", "values": [1,3,53,213,424,3], "operation": "+"})

print(result["result"])