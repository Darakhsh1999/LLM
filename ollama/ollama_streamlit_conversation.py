import streamlit as st
from langchain.tools import Tool
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage
)


# DuckDuckGo
web_search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=web_search.run,
    description="Search the web for information"
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant."),
    ]
    st.session_state.messages_streamlit = []


llm = ChatOllama(model="qwen3:4b").bind_tools([search_tool])


def process_text(user_input):
    # Only handle LLM conversation history
    st.session_state.messages.append(HumanMessage(content=user_input))
    while True:
        response = llm.invoke(st.session_state.messages)
        print(response)
        if response.tool_calls:
            # call tool
            args = response.tool_calls[0]['args']
            id = response.tool_calls[0]['id']
            tool_response = search_tool.invoke({"args": args, "id": id, "name": "search", "type": "tool_call"})
            print(tool_response)
            st.session_state.messages.append(ToolMessage(
                content=tool_response,
                tool_call_id=response.tool_calls[0].id,
            ))
        else:
            break
    st.session_state.messages.append(AIMessage(content=response.content))
    return response.content


st.title("Ollama Chat")

# Render messages
for message in st.session_state.messages_streamlit:
    if message[0] == "user":
        st.chat_message("user").write(message[1])
    elif message[0] == "assistant":
        st.chat_message("assistant").write(message[1])

user_input = st.chat_input("Enter your text here")

if user_input:
    # Add user message to display and show it
    st.session_state.messages_streamlit.append(("user", user_input))
    st.chat_message("user").write(user_input)
    
    # Process with LLM and show spinner
    with st.spinner('Generating response...'):
        response = process_text(user_input)
    
    # Add assistant response to display and show it
    st.session_state.messages_streamlit.append(("assistant", response))
    st.chat_message("assistant").write(response)
