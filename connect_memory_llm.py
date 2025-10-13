import os
import datetime
import traceback
import streamlit as st
import torch
from dotenv import load_dotenv

import pytesseract
from PIL import Image
from PIL.ExifTags import TAGS
from stegano import lsb
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠ GROQ_API_KEY not found. Please set it in your .env file.")
else:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    print("🔍 DEBUG: GROQ_API_KEY loaded successfully.")

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"
CHAT_LOG_FILE = "chatbot_log.txt"
FEEDBACK_LOG_FILE = "feedback_log.txt"

st.set_page_config(page_title="🔍 Cybercrime Investigation Assistant", page_icon="🤖", layout="wide")

torch.set_default_device("cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def extract_image_metadata(image_path):
    image = Image.open(image_path)
    image.load()

    metadata_output = []
    metadata_output.append(f"File: {image.filename}")
    metadata_output.append(f"Format: {image.format}")
    metadata_output.append(f"Size: {image.size}")
    metadata_output.append(f"Mode: {image.mode}")
    metadata_output.append(f"Is animated: {getattr(image, 'is_animated', False)}")
    metadata_output.append(f"Frames in image: {getattr(image, 'n_frames', 1)}")

    exif_data = image.getexif()
    if exif_data:
        metadata_output.append("\nEXIF Metadata:")
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if isinstance(value, bytes):
                try:
                    value = value.decode()
                except:
                    pass
            metadata_output.append(f"  {tag}: {value}")
    else:
        metadata_output.append("\nNo EXIF metadata found.")

    if hasattr(image, "info") and image.info:
        metadata_output.append("\nPNG Metadata Chunks:")
        for k, v in image.info.items():
            metadata_output.append(f"  {k}: {v}")

    return "\n".join(metadata_output)


def extract_steganography(image_path):
    secret_message = lsb.reveal(image_path)
    if secret_message:
        return f"Secret message detected:\n{secret_message}"
    else:
        return "No secret message detected."


def load_uploaded_file(file_path: str):
    ext = file_path.lower().split(".")[-1]

    if ext == "pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif ext == "txt":
        loader = TextLoader(file_path)
        return loader.load()
    elif ext in ["docx", "doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
        return loader.load()
    elif ext == "pptx":
        loader = UnstructuredPowerPointLoader(file_path)
        return loader.load()
    elif ext in ["jpg", "jpeg", "png"]:
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
            if not text.strip():
                st.warning(f"⚠ No readable text found in image.")
                return []
            return [Document(page_content=text, metadata={"source": os.path.basename(file_path)})]
        except Exception as e:
            st.warning(f"❌ Failed to process image: {e}")
            return []
    else:
        raise ValueError(f"❌ Unsupported file type: {ext}")


def process_uploaded_file(file_path: str):
    documents = load_uploaded_file(file_path)
    if not documents:
        st.warning("⚠ No content extracted from the file.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(DB_FAISS_PATH):
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(text_chunks, embeddings=embeddings)
    else:
        db = FAISS.from_documents(text_chunks, embeddings)

    db.save_local(DB_FAISS_PATH)
    return db


@st.cache_resource
def initialize_models():
    try:
        groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
        if not groq_api_key:
            st.error("⚠ GROQ_API_KEY not found. Please set it in .env or Streamlit secrets.")
            return None, None

        if not os.path.exists(DB_FAISS_PATH):
            st.warning("ℹ Vector store not found. Please upload a document to create one.")
            return None, None

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        db = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True,
        )

        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=groq_api_key,
            temperature=0.1,
            max_tokens=1000,
        )

        return db, llm

    except Exception as e:
        st.error(f"❌ Model initialization failed: {e}")
        st.code(traceback.format_exc())
        return None, None


def log_interaction(user_query, context_text, prompt, answer, error=None):
    try:
        with open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
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


def log_feedback(user_query, answer, feedback_type):
    try:
        with open(FEEDBACK_LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(f"User Query: {user_query}\n")
            f.write(f"Answer: {answer[:500]}...\n")
            f.write(f"Feedback: {feedback_type}\n")
            f.write("=" * 60 + "\n")
    except Exception as e:
        st.warning(f"Failed to log feedback: {e}")


st.title("🔍 Cybercrime Investigation Assistant")
st.markdown("Ask questions about cybercrime investigation procedures, UAE laws, and digital evidence handling.")

uploaded_file = st.file_uploader(
    "📂 Upload a document (PDF, DOCX, PPTX, TXT, JPG, PNG)",
    type=["pdf", "docx", "pptx", "txt", "jpg", "jpeg", "png"]
)

metadata_text = ""
steganography_text = ""

if uploaded_file:
    temp_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Uploaded: {uploaded_file.name}")
    db = process_uploaded_file(temp_path)
    if db:
        st.success("✅ Document processed and added to knowledge base!")
    else:
        st.error("❌ Failed to process uploaded document.")

    # If image, show metadata and detect stego
    if uploaded_file.type in ['image/png', 'image/jpeg']:
        with st.expander("🔍 Image Metadata and Steganography Analysis"):
            metadata_text = extract_image_metadata(temp_path)
            st.text(metadata_text)

            steganography_text = extract_steganography(temp_path)
            st.text(steganography_text)

db, llm = initialize_models()

if db is None or llm is None:
    st.info("📌 Upload a document to start chatting.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙ Controls")
    if st.button("🗑 Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()
    st.info(
        """
    *Model Info:*
    - *Embeddings:* all-MiniLM-L6-v2
    - *LLM:* openai/gpt-oss-120b (GROQ)
    - *Vector Store:* FAISS
    """
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("🤔 Thinking..."):
            try:
                docs = db.similarity_search(user_query, k=3)
                context_text = "\n\n".join(doc.page_content for doc in docs)
                prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\nAnswer:"
                response = llm.invoke(prompt)
                answer = response.content if hasattr(response, "content") else str(response)

                log_interaction(user_query, context_text, prompt, answer)

                full_response = answer
                with st.expander("📚 Sources Used"):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"*Source {i}:*")
                        st.text(doc.page_content)
                        st.markdown("---")

                message_placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"❌ Error during response generation: {e}")
                st.code(traceback.format_exc())
                full_response = "Sorry, an error occurred while generating the answer."

            if f"feedback_{len(st.session_state.messages)}" not in st.session_state:
                st.session_state[f"feedback_{len(st.session_state.messages)}"] = None

            col1, col2 = st.columns(2)

            with col1:
                if st.session_state[f"feedback_{len(st.session_state.messages)}"] is None:
                    if st.button("👍 Satisfied", key=f"satisfied_{len(st.session_state.messages)}"):
                        st.session_state[f"feedback_{len(st.session_state.messages)}"] = "Satisfied"
                        st.success("Thanks for your feedback! ✅ (Satisfied)")
                        log_feedback(user_query, answer, "Satisfied")

            with col2:
                if st.session_state[f"feedback_{len(st.session_state.messages)}"] is None:
                    if st.button("👎 Unsatisfied", key=f"unsatisfied_{len(st.session_state.messages)}"):
                        st.session_state[f"feedback_{len(st.session_state.messages)}"] = "Unsatisfied"
                        st.warning("Thanks for your feedback! ❌ (Unsatisfied)")
                        log_feedback(user_query, answer, "Unsatisfied")

            if st.session_state[f"feedback_{len(st.session_state.messages)}"] == "Satisfied":
                st.markdown("*Feedback Recorded:* 👍 Satisfied")
            elif st.session_state[f"feedback_{len(st.session_state.messages)}"] == "Unsatisfied":
                st.markdown("*Feedback Recorded:* 👎 Unsatisfied")

            st.session_state.messages.append({"role": "assistant", "content": full_response})


st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Powered by GROQ, LangChain & FAISS</div>",
    unsafe_allow_html=True,
)