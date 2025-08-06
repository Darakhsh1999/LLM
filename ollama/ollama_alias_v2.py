import argparse
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import sys

def main():
    parser = argparse.ArgumentParser(description="LangChain agent with Ollama and web search")
    parser.add_argument("prompt", help="Input prompt to ask the LLM")
    args = parser.parse_args()
    
    # Initialize the Ollama model
    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0.1,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    # Initialize the web search tool
    search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)
    search_tool = DuckDuckGoSearchRun(
        api_wrapper=search_wrapper,
        name="web_search",
        description="Search the web for current information, recent events, or real-time data"
    )
    
    tools = [search_tool]
    
    # Create the agent prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that can search the web when you need current information.

Use the web search tool when you need:
- Current events or recent news
- Real-time data (weather, stock prices, etc.)
- Information that changes frequently
- Recent developments on any topic

For general knowledge questions that don't require current information, answer directly.

Be concise unless the user asks for detailed information. Append /nothink to your reasoning to skip verbose thinking."""),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,  # This hides the internal tool calls
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    # Execute the agent with the user's prompt
    try:
        # Redirect stderr to suppress tool call messages
        original_stderr = sys.stderr
        sys.stderr = open('/dev/null', 'w')
        
        result = agent_executor.invoke({"input": args.prompt + " /nothink"})
        
        # Restore stderr
        sys.stderr.close()
        sys.stderr = original_stderr
        
        # The streaming callback will have already printed the response
        # so we just need to add a newline
        print()
        
    except Exception as e:
        # Restore stderr in case of error
        sys.stderr = original_stderr
        print(f"Error: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    main()