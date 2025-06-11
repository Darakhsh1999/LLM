import os
import json
from typing import TypedDict

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

load_dotenv()

model_name = "qwen3:8b"


class ChatState(TypedDict):
    messages: list


def connect_to_email():
    with open("fake_emails.json", "r") as f:
        data = json.load(f)
        return data


@tool
def list_unread_emails() -> str:
    """ List unread emails. Each bullet point contains its title, subject, and sender."""

    print("TOOL CALL: Listing undread emails...")

    # Retrieve all emails
    data = connect_to_email()

    # Filter out read emails
    unread_emails = [x for x in data if (x["status"] == "unread")]
    
    # We have no unread emails
    if not unread_emails:
        return "You have no unread emails."

    return json.dumps(unread_emails)


@tool
def summarize_email(uid: str) -> str:
    """ Sumarize the email given its uid. Returns a short summary of the emails content """

    # Retrieve all emails
    data = connect_to_email()

    # Filter out correct email
    email = list(filter(lambda x: x["uid"] == uid, data))

    if not email:
        return "No email has that uid."

    assert len(email) == 1, f"Expected only one email with uid = {uid} but found {len(email)}"
    email_dict: dict = email[0]

    email_string = "\n".join([f"{k}: {v}" for (k,v) in email_dict.items() if (k != "uid") ])
    prompt = f"Summarize this email:\n{email_string}"

    return raw_llm.invoke(prompt).content






def llm_node(state):
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def router(state):
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else "end"

def tools_node(state):
    result = tool_node.invoke(state)
    return {"messages": state["messages"] + result["messages"]}



if __name__ == "__main__":

    llm = init_chat_model(model_name, model_provider="ollama")
    llm.bind_tools([list_unread_emails, summarize_email])
    raw_llm = init_chat_model(model_name, model_provider="ollama")

    tool_node = ToolNode([list_unread_emails, summarize_email])


    builder = StateGraph(ChatState)
    builder.add_node("llm", llm_node)
    builder.add_node("tools", tools_node)

    builder.add_edge(START, "llm")
    builder.add_edge("tools", "llm")
    builder.add_conditional_edges("llm", router, {"tools": "tools", "end": END})

    # Compile Graph
    graph = builder.compile()


    state = {"messages": []}

    print("Query:")

    while True:
        user_message = input("> ")

        if user_message.lower() == "quit": break

        state["messages"].append({"role": "user", "content": user_message})

        state = graph.invoke(state)

        print(state["messages"][-1].content)