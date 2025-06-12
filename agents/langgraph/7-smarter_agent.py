import utils
from typing import TypedDict, List, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv


load_dotenv()

client = ChatOpenAI(
    model="gpt-4o-mini",
)

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# Node functions
def send_message(state: AgentState) -> AgentState:

    response = client.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state


# Construct Graph
graph = StateGraph(AgentState)

## Add Nodes
graph.add_node("send_message", send_message)

## Add edges
graph.add_edge(START,"send_message")
graph.add_edge("send_message", END)

# Compile Graph
app = graph.compile()
utils.convert_to_png(app, "smart_agent")

# Invoke Graph
conversation_history = []

while True:
    user_input = input(">> ")
    if user_input.lower() in ["q","quit","exit","stop"]: break

    conversation_history.append(HumanMessage(user_input))
    result: AgentState = app.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    print("AI:", result["messages"][-1].content)