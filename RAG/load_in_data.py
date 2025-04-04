import os
import os.path as osp
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_data(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf")
    documents = loader.load()
    return documents

def load_single_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    return document

def chunk_data(documents):
    text_chunker = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    chunks = text_chunker.split_documents(documents)
    return chunks



if __name__ == "__main__":

    ## Chunking multiple document
    #pdf_path = osp.join("..","data","pdf")
    #pdf_documents = load_data(pdf_path)
    #chunked_data = chunk_data(pdf_documents)
    #print(len(pdf_documents))
    #print(len(chunked_data))

    print("------")

    # Chunking single PDF document
    pdf_path = osp.join("..","data","pdf","attention is all you need.pdf")
    pdf_document = load_single_pdf(pdf_path)
    chunked_data = chunk_data(pdf_document)
    print("Documents:",len(pdf_document))
    print("Chunks:",len(chunked_data))
