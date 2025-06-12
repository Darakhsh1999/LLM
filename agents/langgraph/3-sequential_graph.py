import utils
from typing import TypedDict, List
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    name: str
    age: str
    skills: List[str]
    result: str

def name_node(state: AgentState)-> AgentState:
    """ Adds gretting to the state

    Args:
        state (AgentState): agent state

    Returns:
        AgentState: agent state
    """

    state["result"] = f"Hello {state["name"]}."
    return state


def age_node(state: AgentState):
    """ Adds age information to the state

    Args:
        state (AgentState): agent state

    Returns:
        AgentState: agent state
    """

    state["result"] = state["result"] + f" You are {state['age']} year old. "
    return state

def skills_node(state: AgentState):
    """ Adds age information to the state

    Args:
        state (AgentState): agent state

    Returns:
        AgentState: agent state
    """

    skills = "Your skills are:\n- " + "\n- ".join(state["skills"])
    state["result"] = state["result"] + skills
    return state


# Construct Graph
graph = StateGraph(AgentState)

## Add Nodes
graph.add_node("name_node", name_node)
graph.add_node("age_node", age_node)
graph.add_node("skills_node", skills_node)

## Add edges
graph.set_entry_point("name_node")
graph.add_edge("name_node", "age_node")
graph.add_edge("age_node", "skills_node")
graph.set_finish_point("skills_node")

app = graph.compile()

utils.convert_to_png(app, "graph3")

result = app.invoke({"name": "Nicólo", "age": "13", "skills": ["robotics", "crafting", "gaming"]})

print(result["result"])