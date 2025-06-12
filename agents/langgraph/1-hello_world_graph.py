from typing import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    name: str


def printing_node(state: AgentState) -> AgentState:
    """ Adds compliment to the state

    Args:
        state (AgentState): _description_

    Returns:
        AgentState: _description_
    """

    state["name"] = f"{state["name"]}, you are so good making pizzas."

    return state


# Construct Graph
graph = StateGraph(AgentState)

## Add Nodes
graph.add_node("print", printing_node)
graph.set_entry_point("print")
graph.set_finish_point("print")

app = graph.compile()


result = app.invoke({"name": "Nicólo"})

print(result["name"])