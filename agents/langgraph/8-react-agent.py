import utils
from pprint import pprint
from typing import TypedDict, Annotated, Sequence
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv


load_dotenv()

SYSTEM_PROMPT = """
You are my AI assistant, please answer my queries to the best of your abilitiy, use the available tools if needed!
"""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Tools
@tool
def add(a: int, b: int) -> int:
    """Addition function that adds two integers together

    Args:
        a (int): first integer
        b (int): second integer

    Returns:
        int_sum int: the sum of the integers
    """

    int_sum = a + b
    return int_sum

@tool
def subtract(a: int, b: int) -> int:
    """Subtraction function that subtracts two integers

    Args:
        a (int): first integer
        b (int): second integer

    Returns:
        int_diff int: the difference of the integers
    """

    int_diff = a - b
    return int_diff

@tool
def multiply(a: int, b: int) -> int:
    """Multiply function that multiplies two integers

    Args:
        a (int): first integer
        b (int): second integer

    Returns:
        int_prod int: the product of the integers
    """

    int_prod = a - b
    return int_prod


tools = [add, subtract, multiply]


client = ChatOpenAI(
    model="gpt-4o-mini",
).bind_tools(tools)


# Node functions
def send_message(state: AgentState) -> AgentState:

    response = client.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}

def message_loop_condition(state: AgentState) -> bool:

    last_message = state["messages"][-1]
    return (not last_message.tool_calls) # True -> END, False -> continue


# Construct Graph
graph = StateGraph(AgentState)

## Add Nodes
graph.add_node("agent", send_message)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

## Add edges
graph.add_edge(START,"agent")
graph.add_conditional_edges(
    "agent",
    message_loop_condition,
    {
        True: END,
        False: "tools"
    }
)
graph.add_edge("tools","agent")

# Compile Graph
app = graph.compile()
utils.convert_to_png(app, "react_agent")

# Invoke Graph
input = {"messages": [("user", "Add 30+55. From the result subtract 22 and multiply the output with 2")]}
result = app.invoke(input)
pprint([msg.content for msg in result["messages"]])

# while True:
#     user_input = input(">> ")
#     if user_input.lower() in ["q","quit","exit","stop"]: break

#     conversation_history.append(HumanMessage(user_input))
#     result: AgentState = app.invoke({"messages": conversation_history})
#     conversation_history = result["messages"]
#     print("AI:", result["messages"][-1].content)