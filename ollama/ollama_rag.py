import ollama
import os.path as osp

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


# Load in text data
pdf_path = osp.join("..","data","pdf")
pdf_data = PyPDFDirectoryLoader(path=pdf_path).load()

#print(pdf_data[0].page_content[:300]) # test print content


# Split text and chunk it 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(pdf_data)


#print(text_chunks[0]) # print chunk


# Create vector database from text chunks
db = Chroma.from_documents(
    documents=text_chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    persist_directory="chroma_db"
)


# Retrieval
llm = ChatOllama(model="llama3.2")
query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)
retriever = MultiQueryRetriever.from_llm(db.as_retriever(), llm, prompt=query_prompt)


rag_template = """Answer the user question based only on the following context:
{context}
----
Question {question}
"""
rag_prompt = ChatPromptTemplate.from_template(template=rag_template)


# Langchain chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)



# Query
respone = chain.invoke(input=("In the Attention is all you need paper, what optimzer was used for training?"))
print(respone)
print(10*"---")
respone = chain.invoke(input=("Who are the authors for the article that proposed the Generative Adversarial Network (GAN)?"))
print(respone)
print(10*"---")
respone = chain.invoke(input=("Give a short summary what ImageNet is and what it is mainly used for."))
print(respone)
print(10*"---")