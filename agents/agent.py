import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
api_key2 = os.getenv("CLAUDE_API_KEY")


class ResearchRespone(BaseModel):
    topic: str
    summary: str
    source: list[str]
    tools_used: list[str]




llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
#llm2 = ChatAnthropic(model_name="claude-3-5-sonnet-20241022", api_key=api_key2)

parser = PydanticOutputParser(pydantic_object=ResearchRespone)


# LLM prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a history assistant that will help generate a research paper.
            Answer the provided user query and use neccessary tools if needed.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """
        ),
        ("placeholder","{chat_history}"),
        ("human","{query}"),
        ("placeholder","{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())


# Agent
tools = [search_tool, wiki_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)


# Executor object
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
user_query = input("Query:")
respone = executor.invoke({"query":user_query})

structured_respone = parser.parse(respone["output"])
print(structured_respone)