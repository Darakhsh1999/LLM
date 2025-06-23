from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama

import asyncio

async def main():

    llm = ChatOllama(model="qwen3:8b")


    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["math_server.py"],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    agent = create_react_agent(llm, tools)

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit": break
        response = await agent.ainvoke({"messages": user_input})
        print("Assistant:", response)

if __name__ == "__main__":
    asyncio.run(main())
    