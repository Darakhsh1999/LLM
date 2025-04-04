import os
import chromadb
import tempfile
import tkinter as tk
import streamlit as st
from tkinter import filedialog
from langchain_chroma import Chroma
from pdf2image import convert_from_path
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import load_in_data


def create_file_selector():
    """
    Opens a file dialog to select a PDF file using tkinter
    Returns the selected file path
    """
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    file_path = filedialog.askopenfilename(
        title="Select PDF Document",
        filetypes=[("PDF files", "*.pdf")]
    )
    root.destroy()
    return file_path if file_path else None

def get_pdf_preview(pdf_path):
    """
    Creates a preview image of the first page of the PDF
    Returns the image for display in Streamlit
    """
    try:
        # Convert first page of PDF to image
        with tempfile.TemporaryDirectory() as temp_dir:
            images = convert_from_path(pdf_path, first_page=1, last_page=1, 
                                      output_folder=temp_dir, dpi=100)
            if images:
                return images[0]
    except Exception as e:
        st.sidebar.error(f"Error generating preview: {e}")
    return None

def main():

    ### Streamlit GUI ###

    # App title
    st.set_page_config(page_title="RAG by Arash Darakhsh", layout="wide")

    # Session state initialization
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    
    if 'sidebar_expanded' not in st.session_state:
        st.session_state.sidebar_expanded = True
    
    if 'db' not in st.session_state:

        # Load in models
        model = OllamaLLM(model="llama3.2")
        embed_model = OllamaEmbeddings(model="nomic-embed-text")

        # Load DB
        db = Chroma(
            client=chromadb.PersistentClient(),
            collection_name="RAG-pdf",
            embedding_function=embed_model,
        )

        # Load in model, db and retriever object
        st.session_state.model = model
        st.session_state.db = db
        st.session_state.retriever = db.as_retriever(search_kwargs={"k": 5})

        print("Loaded DB into memory")
    
    if 'prompt' not in st.session_state:

        # Query prompt
        prompt_template = """
        You are and LLM assistant to aid the user with Retrieval Augmented Generation (RAG).
        The user will ask you question and your job is to answer it based on the relevant context.
        If the question can be answered in a short form, do so. Do not include too much details.

        Here is some relevant context to the question: {context}

        Here is the question the user has asked: {question}
        """
        st.session_state.prompt = ChatPromptTemplate.from_template(prompt_template)

    
    # Sidebar widget
    with st.sidebar:

        st.header("Document Selection") # Document selection 
        
        # Large red file select button
        if st.button("📄 SELECT PDF FILE", use_container_width=True, type="primary"):
            selected_file = create_file_selector()
            if selected_file:
                st.session_state.current_file = selected_file
                filename = os.path.basename(selected_file)
                st.success(f"Selected: {filename}")
                
                # Load in and chunk PDF
                pdf_document = load_in_data.load_single_pdf(selected_file)
                pdf_chunks = load_in_data.chunk_data(pdf_document)

                # Add PDF chunks to database
                st.session_state.db.add_documents(pdf_chunks)
        
        # Display current file if one is selected
        if st.session_state.current_file:
            st.markdown("---")
            st.subheader("Current Document")
            st.write(os.path.basename(st.session_state.current_file))
            st.markdown("---")
            st.subheader("Front page preview")

            # Image preview
            preview_image = get_pdf_preview(st.session_state.current_file)
            if preview_image:
                st.image(preview_image, caption="PDF Preview", use_container_width=True)
    
    # Conversation title
    st.title("Message dialogue")
    
    # Display conversation
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # User input
    user_input = st.chat_input("Ask a question about your document...")
    
    # Reading and Writing text messages to text box
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Parse user input
        if not st.session_state.current_file:
            response = "Please select a PDF document first."
        else:

            # Get context
            try:
                context = st.session_state.retriever.invoke(user_input)
            except:
                context = ""

            # Create chain
            chain = st.session_state.prompt | st.session_state.model # prompt -> model

            # Invoke chain
            response = chain.invoke({"question": user_input, "context": context})
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)

if __name__ == "__main__":
    main()