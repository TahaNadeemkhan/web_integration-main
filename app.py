import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from dotenv import load_dotenv
import os

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "gemini-2.0-flash" 
# --- Initialization ---
# Check for API Key
if not GOOGLE_API_KEY:
    st.error("🔴 Error: GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")
    st.stop() # Stop execution if key is missing

# Embedding model (initialize only once)
@st.cache_resource # Cache the embedding model resource
def get_embedding_model():
    print("Loading embedding model...") 
    try:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    except Exception as e:
        st.error(f"🔴 Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        st.stop()
embed_model = get_embedding_model()

# Streamlit setup
st.set_page_config(page_title="Website Insights", layout="wide")
st.title("Chat with Websites 💬")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! Please enter a website URL in the sidebar to begin.")]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_url" not in st.session_state:
    st.session_state.processed_url = None

# --- Core Functions ---

# Function to load website, split text, and create an in-memory vector store
def get_vector_store_from_url(url):
    """Loads content from URL, splits it, and creates an in-memory FAISS vector store."""
    if not url:
        return None, "URL is empty."
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()

        if not documents:
             return None, "Could not load any content from the URL. Check if the URL is correct and accessible."

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        doc_chunks = text_splitter.split_documents(documents)

        if not doc_chunks:
            return None, "Loaded content but could not split it into chunks."

        # Create FAISS vector store in memory
        vector_store = FAISS.from_documents(doc_chunks, embed_model)
        return vector_store, None # Return store and no error

    except Exception as e:
        return None, f"Failed to process URL '{url}': {str(e)}"

# Function to create a retriever chain that considers chat history
def get_context_retriever_chain(vector_store):
    """Creates a history-aware retriever chain."""
    try:
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.2) # Lower temp for query generation
        retriever = vector_store.as_retriever()
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name='chat_history'),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up relevant information from the website content.")
        ])
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
        return retriever_chain
    except Exception as e:
        st.error(f"🔴 Error creating retriever chain: {e}")
        return None

# Function to set up the main RAG (Retrieval-Augmented Generation) chain
def get_conversational_rag_chain(retriever_chain):
    """Creates the main conversational RAG chain."""
    try:
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.7)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for answering questions about a specific website. Use the provided context (website content) to answer the user's questions. If the information is not in the context, state that you cannot answer based on the provided website content. Be concise and helpful.\n\nContext:\n{context}"),
            MessagesPlaceholder(variable_name='chat_history'),
            ('user', "{input}"),
        ])
        stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever_chain, stuff_documents_chain)
    except Exception as e:
        st.error(f"🔴 Error creating RAG chain: {e}")
        return None

# Function to get the bot's response
def get_response(user_input):
    """Gets the response from the RAG chain based on user input and history."""
    if st.session_state.vector_store is None:
        return "⚠️ Please submit a valid website URL first using the sidebar."

    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    if not retriever_chain:
        return "🔴 Error: Could not create the retriever chain. Cannot process query."

    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    if not conversation_rag_chain:
         return "🔴 Error: Could not create the conversation chain. Cannot process query."

    try:
        response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })

        answer = response.get('answer', '').strip()

        # Check if the answer is meaningful or a fallback/refusal
        # (This check might need refinement based on the LLM's typical refusal patterns)
        common_refusals = [
            "I cannot answer based on the provided website content",
            "information is not in the context",
            "don't have information about that",
            "based on the provided context",
        ]
        is_refusal = not answer or any(refusal in answer.lower() for refusal in common_refusals)

        # Optional: Add a fallback if the LLM gives a poor answer despite having context
        # if not answer and response.get('context'):
        #     answer = "I found some relevant information but couldn't formulate a specific answer. Please try rephrasing your question."
        # elif is_refusal:
        #     # Optional: Check if context was actually retrieved
        #     if response.get('context'):
        #          answer = "I found some context on the website, but it doesn't seem to contain the specific answer to your question."
        #     else:
        #          answer = "I couldn't find relevant information on the website to answer your question."


        return answer if answer else "I couldn't find relevant information on the website to answer your question."

    except Exception as e:
        return f"🔴 Error processing your query: {str(e)}"

# --- Streamlit UI Elements ---

# Sidebar for URL input
with st.sidebar:
    st.header("Settings")
    web_url = st.text_input("Enter Website URL", key="url_input")
    button_clicked = st.button("Load Website", type="primary")

    if button_clicked:
        if not web_url:
            st.warning("Please enter a website URL.")
        # Process only if the URL is new
        elif web_url != st.session_state.get("processed_url"):
            with st.spinner(f"Processing {web_url}... This may take a moment."):
                vector_store, error_message = get_vector_store_from_url(web_url)
                if error_message:
                    st.error(f"🔴 Failed: {error_message}")
                    st.session_state.vector_store = None
                    st.session_state.processed_url = None
                    st.session_state.chat_history = [AIMessage(content="Failed to load the website. Please try another URL.")]
                elif vector_store:
                    st.session_state.vector_store = vector_store
                    st.session_state.processed_url = web_url
                    # Clear chat history for the new site
                    st.session_state.chat_history = [AIMessage(content=f"Website '{web_url}' loaded! How can I help you?")]
                    st.success("✅ Website loaded successfully!")
                    # Force rerun to clear chat input and display new history
                    st.rerun()
        elif web_url == st.session_state.processed_url:
            st.info("This URL has already been loaded.")


# Display chat history
st.subheader("Chat History")
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Chat input area
user_query = st.chat_input("Ask a question about the website...")

if user_query:
    # Add user message to history immediately
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.write(user_query)

    # Get response (handle potential errors)
    with st.spinner("Thinking..."):
        response_content = get_response(user_query)

    # Add AI response to history and display
    st.session_state.chat_history.append(AIMessage(content=response_content))
    with st.chat_message("AI"):
        st.write(response_content)