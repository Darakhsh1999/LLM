import os
import openai 

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

from load_in_data import load_data, chunk_data

PROMPT_TEMPLATE = """
You are an assistan that will perform Retrieval Augmented Generation (RAG).
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Load in API key
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

def word_to_vector(word:str):
    embed_fn = OpenAIEmbeddings()
    return embed_fn.embed_query(word)


def create_vector_db():
    
    pdf_path = os.path.join("..","data","pdf")
    pdf_documents = load_data(pdf_path)
    data_chunks = chunk_data(pdf_documents)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        data_chunks, OpenAIEmbeddings(), persist_directory="chroma"
    )
    db.persist()

def load_vector_db():
    embed_fn = OpenAIEmbeddings()
    db = Chroma(persist_directory="chroma", embedding_function=embed_fn)
    return db

def db_similarity_search(query, db:Chroma):
    results = db.similarity_search_with_relevance_scores(query, k=3)
    if len(results) == 0 or results[0][1] < 0.75:
        return None
    return results

def parse_db_results(results, query_text):
    context_text = "\n\n---\n\n".join([doc.page_content for (doc,_) in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    return prompt

if __name__ == "__main__":
    create_vector_db()