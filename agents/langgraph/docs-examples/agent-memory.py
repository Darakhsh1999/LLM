from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
load_dotenv()

memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = init_chat_model("openai:gpt-4o-mini")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile(checkpointer=memory)

def invoke_chatbot(user_input: str, config: dict):
    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config
    )
    print("Assistant:", result["messages"][-1].content)



if __name__ == "__main__":

    config1 = {"configurable": {"thread_id": "1"}}
    config2 = {"configurable": {"thread_id": "2"}}
    
    invoke_chatbot("Hello my Name is Thomas", config1)

    # sleep 5 seconds
    import time
    time.sleep(2)

    invoke_chatbot("What is my name again?", config1)

    invoke_chatbot("Wait I forgot my name, what was it again?", config2)