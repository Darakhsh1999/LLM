import utils
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv


load_dotenv()

client = ChatOpenAI(
    model="gpt-4o-mini",
)

class AgentState(TypedDict):
    messages: List[BaseMessage]

# Node functions
def send_message(state: AgentState) -> AgentState:

    response = client.invoke(state["messages"])
    state["messages"].append(AIMessage(response.content))
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

# Invoke Graph
user_input = input("Enter message: ")
result = app.invoke({"messages": [HumanMessage(user_input)]})

print(result["messages"][-1].content)