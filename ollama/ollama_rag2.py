import os.path as osp
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


### Data loading ###
def load_data(_type: str = "pdf"):
    """ Load in data from disc """
    assert _type in ["pdf","txt"], f"Unsupported data type {_type}"
    pdf_path = osp.join("..","data","pdf")
    pdf_data = PyPDFDirectoryLoader(path=pdf_path).load()
    return pdf_data

def chunk_data(data):
    """ Chunks the data with overlap """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(data)
    return text_chunks

### Data base ###
def create_db(chunked_data, embed_model):
    db_path = osp.join(".","chroma_db")
    db = Chroma.from_documents(
        documents=chunked_data,
        collection_name="pdf_vector_store",
        persist_directory=db_path,
        embedding=embed_model)
    return db


if __name__ == "__main__":


    # Load in data
    data = load_data()

    # Process data
    chunked_data = chunk_data(data)

    # Load in model
    model = OllamaLLM(model="llama3.1")
    embed_model = OllamaEmbeddings(model="mxbai-embed-large")

    # Create DB 
    db = create_db(chunked_data=chunked_data, embed_model=embed_model)

    # DB retriever
    retriever = db.as_retriever(search_kwargs={"k": 5})


    # Query prompt
    prompt_template = """
    You are and LLM assistant to aid the user with Retrieval Augmented Generation (RAG).
    The user will ask you question and your job is to answer it based on the relevant context.
    If the question can be answered in a short form, do so. Do not include too much details.

    Here is some relevant context to the question: {context}

    Here is the question the user has asked: {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)


    # Main loop
    chain = prompt | model # prompt -> model
    while (True):

        question = input("Query:") 
        if question in ["q","quit"]:
            exit()

        # Vector DB search
        try:
            context = retriever.invoke(question)
        except:
            print("Retrive failed, no context provided")
            context = ""

        output_result = chain.invoke({"question": question, "context": context})
        print(output_result)

