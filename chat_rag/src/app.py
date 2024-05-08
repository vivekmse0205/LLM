import os
import logging
from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import ContextualCompressionRetriever, BM25Retriever, EnsembleRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings, FastEmbedEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Configuration
SAVE_BASE_PATH = "data"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
os.makedirs(SAVE_BASE_PATH,exist_ok=True)
# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO)


def load_data_from_web(web_url):
    """
    Load data from a web URL
    """
    documents = []
    try:
        if web_url:
            document = WebBaseLoader(web_url).load()
            documents.extend(document)
    except Exception as e:
        st.error(f"Error loading data from web URL: {e}")
        logging.error(f"Error loading data from web URL: {e}")
    return documents

def load_data_from_files(file_bytes):
    """
    Load data from uploaded files
    """
    documents = []
    try:
        for file_data in file_bytes:
            file_save_path = os.path.join(SAVE_BASE_PATH, file_data.name)
            with open(file_save_path, mode='wb') as w:
                w.write(file_data.getvalue())
            loader = PyPDFLoader(file_save_path)
            document = loader.load_and_split()
            documents.extend(document)
    except Exception as e:
        st.error(f"Error loading data from uploaded files: {e}")
        logging.error(f"Error loading data from uploaded files: {e}")
    return documents

def ingest_data(file_bytes, web_url):
    """
    Data ingestion pipeline
    """
    documents = []
    documents = load_data_from_web(web_url)
    documents.extend(load_data_from_files(file_bytes))
    if documents:
        document_chunks = split_documents(documents)
        return document_chunks
    return None

def split_documents(documents, chunk_size=400, overlap=50):
    """
    Split the large document into chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)



def insert_data(doc_chunks,use_hf=False):
    """
    Create embeddings and upload to vector db
    """
    embeddings = FastEmbedEmbeddings()
    if use_hf:
        embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=st.session_state.hf_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
        )
    vector_store = Chroma.from_documents(doc_chunks, embeddings)
    return vector_store

def get_retriever(vector_store, chunks, num_results=10):
    """
    Return an initialized retriever
    """
    vectorstore_retriever = vector_store.as_retriever(search_kwargs={"k": num_results})
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = num_results
    ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retriever, keyword_retriever],
                                           weights=[0.5, 0.5])
    return ensemble_retriever

def get_response(user_query):
    """
    For a given user query, return response
    """
    try:
        response = st.session_state.conversation({'question': user_query})
        st.session_state.chat_history = response['chat_history']
        st.write(response['answer'])
    except Exception as e:
        st.error("Error in application, restart the app")
        logging.error(f"Error in application: {e}")

def get_conversation_chain(retriever, is_rerank=False):
    # LLM for generating output - we can use open source llm from huggingface also.
    llm = ChatOpenAI()
    if is_rerank:
        compressor = FlashrankRerank()
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain

def main():
    text_HF_API_key = "HuggingFace API key - [Get an API key](https://huggingface.co/settings/tokens)"
    text_OAI_API_key = "OpenAI API key"
    st.set_page_config("Personal assistant")
    st.header("Get information about the candidate from the resume")
    st.session_state.hf_api_key = ""

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("Ask questions..")
    if user_question:
        get_response(user_question)

    with st.sidebar:
        st.title("Menu:")
        st.session_state.hf_api_key = st.text_input(
            text_HF_API_key,
            type="password",
            placeholder="insert your API key",
        )
        st.session_state.openai_api_key = st.text_input(
            text_OAI_API_key,
            type="password",
            placeholder="insert your API key",
        )
        st.session_state.portfolio = st.text_input(
            "Candidate portfolio",
            placeholder="https://example_portfolio"
        )
        pdf_docs = st.file_uploader("Upload pdf file about the candidate and Click Process button",
                                    accept_multiple_files=True, type=["pdf", "docx"])
        if st.button("Process"):
            if OPENAI_API_KEY:
                st.session_state.openai_api_key = OPENAI_API_KEY
            if HF_API_KEY:
                st.session_state.hf_api_key = HF_API_KEY
            if not st.session_state.hf_api_key  or not st.session_state.openai_api_key:
                st.session_state.error_message = "Please enter api keys for processing"
                st.error("Please enter the HuggingFace API key for processing")
            else:
                os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
                with st.spinner("Storing Data..."):
                    # Data ingestion
                    document_chunks = ingest_data(pdf_docs, st.session_state.portfolio)
                    vector_store = insert_data(document_chunks)
                    retriever = get_retriever(vector_store, document_chunks)
                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(retriever)

if __name__ == "__main__":
    main()
