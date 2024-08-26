import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader,TextLoader
import os
import tempfile
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
st.set_page_config(layout="wide")


with st.sidebar:
    api_token = st.text_input("API TOKEN", type='password')

# --- Streamlit Session State Management ---
def initialize_session_state():
    """Initializes Streamlit session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'context' not in st.session_state:
        st.session_state.context = []
    if 'mode_of_chat' not in st.session_state:
        st.session_state.mode_of_chat = "Generic"
    if 'context_url' not in st.session_state:
        st.session_state.context_url = ""
    if 'context_text' not in st.session_state:
        st.session_state.context_text = ""

@st.cache_resource
def initialize_llm_and_embeddings():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro", 
        temperature=0.7, 
        streaming=True, 
    )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return llm, embeddings

@st.cache_resource
def create_combine_docs_chain(_llm):
    template = """
    Given the following context, answer the user's question only from the context consisely. 
    
    Context:'''
    {context}
    '''

    Question:'''
    {input}
    '''

    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    return create_stuff_documents_chain(_llm, prompt)

def load_and_process_docs(file_path_or_url, doc_type):
    try:
        if doc_type == 'url':
            loader = UnstructuredURLLoader(urls=[file_path_or_url])
            documents = loader.load()
        elif doc_type == 'doc':
            # documents=None
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file_path_or_url.read())
                temp_file_path = temp_file.name
            if file_path_or_url.type == 'application/pdf':
                loader = PyPDFLoader(temp_file_path)
                documents = loader.load()
            if file_path_or_url.type =='text/plain':
                loader = TextLoader(temp_file_path)
                documents = loader.load()
        else: 
            documents = [Document(page_content=file_path_or_url)] 
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        return docs
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return None

# @st.cache_resource  
def get_retrieval_chain(_docs,_embeddings, _combine_docs_chain):
    db = FAISS.from_documents(_docs, _embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 10}) 
    retrieval_chain = create_retrieval_chain(retriever, _combine_docs_chain)
    return retrieval_chain

def stream_response(response):
    for chunk in response:
        if answer_chunk := chunk.get("answer"):
            yield answer_chunk

def write_messages(prompt, retriever=None, mode='generic'):
    new_message = {"user": prompt, "ai": ""}
    st.session_state.chat_history.append(new_message)
    
    with messages:
        for message in st.session_state.chat_history[:-1]:
            with st.chat_message("user"):
                st.write(message['user'])
            with st.chat_message("ai"):
                st.write(message['ai'])


        with messages.chat_message("user"):
            st.write(prompt)

        with messages.chat_message("ai"): 
            if mode == 'generic':
                with st.spinner("Thinking..."):
                    stream = llm.stream(prompt)
                    full_response = st.write_stream(stream)
                    st.session_state.chat_history[-1]["ai"] = full_response

            else:
                with st.spinner("Searching for relevant information..."):
                    stream = retriever.stream({"input": prompt})
                    full_response = st.write_stream(stream_response(stream))
                    st.session_state.chat_history[-1]["ai"] = full_response


# Initialize session state
initialize_session_state()

if api_token:
    os.environ["GOOGLE_API_KEY"] = api_token
    llm, embeddings = initialize_llm_and_embeddings()
    combine_docs_chain = create_combine_docs_chain(llm)

    st.title("Gemini Pro 1.5 RAG Chatbot")

    def handle_context(change_widget):
        if 'context_url' == change_widget:
            url = load_and_process_docs(st.session_state['context_url'], 'url')
            st.session_state.context = url
        if 'context_pdf' == change_widget:
            if st.session_state['context_pdf'] is not None:
                doc = load_and_process_docs(st.session_state['context_pdf'], 'doc')
                st.session_state.context = doc
        if 'context_text' == change_widget:
            text = load_and_process_docs(st.session_state['context_text'], 'text')
            st.session_state.context = text
                

    expander = st.expander("Provide Context (Optional)")
    document_url = expander.text_input("Website URL",value=st.session_state.context_url ,key="context_url",on_change=handle_context,args=('context_url',))
    uploaded_file = expander.file_uploader("PDF Document", type=["pdf","txt"],key="context_pdf",on_change=handle_context,args=('context_pdf',))
    text_content = expander.text_area("Or Paste Text Here",value=st.session_state.context_text ,key="context_text",on_change=handle_context,args=('context_text',))
        
    messages = st.container(height=250)
    prompt = st.text_area("Your Question:", height=100)
    button = st.button("Send")
    st.session_state.mode_of_chat = st.radio(
        "Conversation Mode", 
        ["Generic", "Context-Aware"], 
        index=0 if st.session_state.mode_of_chat == "Generic" else 1, 
        horizontal=True
    )

    if prompt and button:
        if st.session_state.mode_of_chat == 'Context-Aware':
            if not st.session_state.context:
                st.warning("Please provide context (URL, PDF,TXT or text) for Context-Aware mode.")
                st.stop() 
            retriever = get_retrieval_chain(st.session_state.context, embeddings, combine_docs_chain)
            write_messages(prompt, retriever, mode='context-aware')
        else:
            write_messages(prompt, mode='generic') 

else:
    st.info("Please provide the API_TOKEN IN the side bar before proceeding..")
