import os
import utils
from pprint import pprint
from typing import TypedDict, Annotated, Sequence
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv


load_dotenv()


embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

SYSTEM_PROMPT = """
You are my AI assistant helping with RAG. Please answer the query based on the retrieved document context from the DB. You can look up context using the built in tools
"""

# Load data
pdf_data_path = os.path.join("..","..","data","pdf","attention is all you need.pdf")
pdf = PyPDFLoader(pdf_data_path)

try:
    pages = pdf.load()
    print(f"PDF loaded with {len(pages)} pages")
except Exception as e:
    raise IOError(f"Error loading PDF: {e}")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
)

chunks = text_splitter.split_documents(pages)

db_path = "chroma"
db_collection = "attention"

try:
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path,
        collection_name=db_collection
    )
    print(f"Created DB {db_collection}")
except Exception as e:
    print(f"Error creating DB: {str(e)}")

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# Tools
@tool
def retriever_tool(query: str) -> str:
    """Returns relevant context 

    Args:
        query (str): user query

    Returns:
        str: Document context if present in database
    """

    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found given the query "

    return "\n\n".join([f"Document {idx}: \n{doc.page_content}" for idx,doc in enumerate(docs)])


tools = [retriever_tool]

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
).bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

tools_dict = {_tool.name: _tool for _tool in tools}

# Node functions
def send_message(state: AgentState) -> AgentState:

    messages = list(state["messages"])
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    message = model.invoke(messages)
    return {"messages": [message]}

def message_loop_condition(state: AgentState) -> bool:

    last_message = state["messages"][-1]
    return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0

# Retriever agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

# Construct Graph
graph = StateGraph(AgentState)

## Add Nodes
graph.add_node("llm", send_message)
graph.add_node("retriever_agent", take_action)

## Add edges
graph.set_entry_point("llm")
graph.add_edge("retriever_agent", "llm")
graph.add_conditional_edges(
    "llm",
    message_loop_condition,
    {
        True: "retriever_agent",
        False: END
    }
)

# Compile Graph
rag_agent = graph.compile()
utils.convert_to_png(rag_agent, "rag_agent")


def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

if __name__ == "__main__":
    running_agent()