import os
import datetime
import traceback
import streamlit as st
import torch
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- Config ---
DB_FAISS_PATH = "vectorstore/db_faiss"
LOG_FILE = "chatbot_log.txt"

# Page configuration
st.set_page_config(page_title="🔍 Cybercrime Investigation Assistant", page_icon="🤖", layout="wide")

# Force CPU usage to avoid meta tensor issues
torch.set_default_device('cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize models with caching and proper error handling
@st.cache_resource
def initialize_models():
    """Initialize FAISS database and ChatGroq LLM with error handling"""
    try:
        # Get GROQ API key from environment or secrets
        groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
        
        if not groq_api_key:
            st.error("⚠ GROQ_API_KEY not found. Please set it in environment or secrets.")
            return None, None
            
        # Check if vector store exists
        if not os.path.exists(DB_FAISS_PATH):
            st.error(f"❌ Vector store not found at {DB_FAISS_PATH}")
            st.info("👉 Please create the vector store first using the setup script.")
            return None, None
        
        # Initialize embeddings with explicit CPU configuration
        st.info("🧠 Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': False
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 1,
                'show_progress_bar': False
            }
        )
        
        # Load FAISS database
        st.info("🔍 Loading vector database...")
        db = FAISS.load_local(
            DB_FAISS_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        
        # Initialize GROQ LLM
        st.info("🤖 Initializing GROQ LLM...")
        llm = ChatGroq(
            model="openai/gpt-oss-120b", 
            api_key=groq_api_key,
            temperature=0.1,
            max_tokens=1000
        )
        
        return db, llm
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Model initialization failed: {error_msg}")
        
        # Specific error handling for meta tensor issues
        if "meta tensor" in error_msg.lower():
            st.warning("""
            *Meta Tensor Error Detected!* Try these fixes:
            1. Update dependencies: pip install --upgrade torch transformers sentence-transformers
            2. Clear cache: Delete ~/.cache/huggingface/ folder
            3. Restart Streamlit app
            """)
        
        st.code(traceback.format_exc())
        return None, None

def log_interaction(user_query, context_text, prompt, answer, error=None):
    """Log interactions and errors into a text file"""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(f"User Query: {user_query}\n")
            if context_text:
                f.write(f"Context Extracted:\n{context_text[:1000]}...\n")
            f.write(f"Final Prompt:\n{prompt}\n")
            if answer:
                f.write(f"Answer:\n{answer}\n")
            if error:
                f.write(f"Error:\n{error}\n")
            f.write("=" * 80 + "\n")
    except Exception as e:
        st.warning(f"Failed to log interaction: {e}")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main UI
st.title("🔍 Cybercrime Investigation Assistant")
st.markdown("Ask questions about cybercrime investigation procedures, UAE laws, and digital evidence handling.")

# Initialize models
with st.spinner("Initializing models..."):
    db, llm = initialize_models()

if db is None or llm is None:
    st.error("❌ Failed to initialize models. Please check the configuration and try again.")
    st.stop()

st.success("✅ Models initialized successfully!")

# Sidebar
with st.sidebar:
    st.header("⚙ Controls")
    
    # Clear chat button
    if st.button("🗑 Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared!")
    
    # Model info
    st.info("""
    *Model Info:*
    - Embeddings: all-MiniLM-L6-v2
    - LLM: llama-3.3-70b-versatile (GROQ)
    - Vector Store: FAISS
    """)

# Main chat interface
st.subheader("💬 Ask Your Question")

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.expander(f"Q{i+1}: {message['query'][:50]}..."):
        st.write(f"*Question:* {message['query']}")
        st.write(f"*Answer:* {message['answer']}")

# Input form
with st.form("query_form", clear_on_submit=True):
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_query = st.text_area(
            "Enter your cybercrime investigation query:",
            height=100,
            placeholder="e.g., What are the legal procedures for digital evidence collection in UAE?"
        )
    
    with col2:
        st.write("") # Spacer
        st.write("") # Spacer
        custom_template = st.text_area(
            "Custom template (optional):",
            height=70,
            placeholder="Use {context} and {question}"
        )
    
    submit_button = st.form_submit_button("🔍 Ask", use_container_width=True)

# Process query
if submit_button:
    if not user_query.strip():
        st.warning("⚠ Please enter a query.")
    else:
        with st.spinner("🤔 Processing your query..."):
            try:
                # Retrieve relevant documents
                docs = db.similarity_search(user_query, k=3)
                context_text = "\n\n".join(doc.page_content for doc in docs)
                
                # Build prompt
                if custom_template.strip():
                    prompt = custom_template.format(context=context_text, question=user_query)
                else:
                    prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\nAnswer:"
                
                # Get response from LLM
                response = llm.invoke(prompt)
                answer = response.content if hasattr(response, "content") else str(response)
                
                # Log interaction
                log_interaction(user_query, context_text, prompt, answer)
                
                # Add to session state
                st.session_state.messages.append({
                    "query": user_query,
                    "answer": answer
                })
                
                # Display response
                st.success("✅ Response generated!")
                
                # Show answer
                st.subheader("📋 Answer")
                st.write(answer)
                
                # Show sources
                with st.expander("📚 Sources Used"):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"*Source {i}:*")
                        st.text(doc.page_content[:300] + "...")
                        st.markdown("---")
                
            except Exception as e:
                error_msg = traceback.format_exc()
                log_interaction(user_query, None, None, None, error=error_msg)
                st.error(f"❌ An error occurred: {str(e)}")
                
                with st.expander("🔍 Error Details"):
                    st.code(error_msg)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🔍 Cybercrime Investigation Assistant<br>
    Powered by GROQ API, LangChain & FAISS<br>
    UAE Legal Framework Compliant
</div>
""", unsafe_allow_html=True)