import os
import datetime
import traceback
import json
import uuid
import subprocess
# import threading
# import wave

import streamlit as st
import pytesseract
# import pyaudio
from PIL import Image
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
# import whisper
# from gtts import gTTS

# --- Load Environment Variables ---
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠ GROQ_API_KEY not found. Please set it in your .env file.")
else:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    print("🔍 DEBUG: GROQ_API_KEY loaded successfully.")


# --- Config ---
DB_FAISS_PATH = "vectorstore/db_faiss"
CHAT_HISTORY_FILE = "chat_history.json"
FEEDBACK_LOG_FILE = "feedback_log.txt"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# RECORDINGS_DIR = "recordings"
# RECORDING_PATH = os.path.join(RECORDINGS_DIR, "voice_input.wav")

# --- Page Configuration ---
st.image("images/logo.png", width=100)
st.set_page_config(page_title="🔍 Assistant ", page_icon="🤖", layout="wide")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------
# FFmpeg Check
# -------------------------------
def is_ffmpeg_installed():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False

FFMPEG_OK = is_ffmpeg_installed()
if not FFMPEG_OK:
    st.warning("⚠ FFmpeg not found. Voice input/output may be limited. Install FFmpeg and add it to PATH.")

# -------------------------------
# Chat History Management
# -------------------------------
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def new_chat_session(user_query=""):
    chat_id = str(uuid.uuid4())
    title = user_query[:40] if user_query else f"Chat {len(st.session_state.chat_history) + 1}"
    st.session_state.chat_history[chat_id] = {"title": title, "messages": []}
    save_chat_history(st.session_state.chat_history)
    st.session_state.active_chat = chat_id

def switch_chat(chat_id):
    st.session_state.active_chat = chat_id

def delete_chat(chat_id):
    if chat_id in st.session_state.chat_history:
        del st.session_state.chat_history[chat_id]
        save_chat_history(st.session_state.chat_history)
        if st.session_state.active_chat == chat_id:
            st.session_state.active_chat = None

# -------------------------------
# Feedback Logging
# -------------------------------
def log_feedback(user_query, answer, feedback_type, reason=None):
    try:
        with open(FEEDBACK_LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(f"User Query: {user_query}\n")
            f.write(f"Answer: {answer[:500]}...\n")
            f.write(f"Feedback: {feedback_type}\n")
            if reason:
                f.write(f"Reason: {reason}\n")
            f.write("=" * 60 + "\n")
    except Exception as e:
        st.warning(f"Failed to log feedback: {e}")

# -------------------------------
# File Handling
# -------------------------------
def load_uploaded_file(file_path: str):
    ext = file_path.lower().split(".")[-1]
    if ext == "pdf":
        return PyPDFLoader(file_path).load()
    elif ext == "txt":
        return TextLoader(file_path, autodetect_encoding=True).load()
    elif ext in ["docx", "doc"]:
        return UnstructuredWordDocumentLoader(file_path).load()
    elif ext == "pptx":
        return UnstructuredPowerPointLoader(file_path).load()
    elif ext in ["jpg", "jpeg", "png"]:
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
            if not text.strip():
                st.warning("⚠ No readable text found in image.")
                return None
            st.info(f"📝 OCR Extracted Text Preview:\n\n{text[:500]}")
            return [Document(page_content=text, metadata={"source": os.path.basename(file_path), "file_type": "image"})]
        except Exception as e:
            st.error(f"❌ Failed to process image {file_path}: {e}")
            return None
    else:
        raise ValueError(f"❌ Unsupported file type: {ext}")

def process_uploaded_file(file_path: str):
    documents = load_uploaded_file(file_path)
    if not documents:
        st.warning("⚠ No content extracted from the file.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})

    if os.path.exists(DB_FAISS_PATH):
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(text_chunks)
    else:
        db = FAISS.from_documents(text_chunks, embeddings)

    db.save_local(DB_FAISS_PATH)
    if 'db_instance' in st.session_state:
        del st.session_state['db_instance']
    if 'llm_instance' in st.session_state:
        del st.session_state['llm_instance']

    st.success(f"📚 File processed successfully.")
    return db
# -------------------------------
# Document Summarization
# -------------------------------
def summarize_document(file_path: str, llm):
    """Extract text from a document and return a concise summary."""
    try:
        documents = load_uploaded_file(file_path)
        if not documents:
            st.warning("⚠ No content to summarize.")
            return None

        # Combine all pages/texts into one string
        full_text = "\n".join([doc.page_content for doc in documents])

        # Limit text for performance (optional, can handle 10k+ tokens)
        full_text = full_text[:15000]

        prompt = f"""
You are a professional legal summarizer for cyber law and cybersecurity documents.

Summarize the following document in clear, concise language. 
Highlight key legal points, definitions, and procedures relevant to cybercrime or digital evidence.

Document Text:
{full_text}

Summary:
"""
        summary_response = llm.invoke(prompt)
        summary = summary_response.content if hasattr(summary_response, "content") else str(summary_response)
        return summary.strip()

    except Exception as e:
        st.error(f"❌ Summarization failed: {e}")
        st.code(traceback.format_exc())
        return None

# -------------------------------
# Model Initialization
# -------------------------------
def initialize_models():
    try:
        if 'db_instance' in st.session_state and 'llm_instance' in st.session_state:
            return st.session_state['db_instance'], st.session_state['llm_instance']

        groq_api_key = GROQ_API_KEY
        if not groq_api_key:
            st.error("⚠ GROQ_API_KEY not set.")
            return None, None

        if not os.path.exists(DB_FAISS_PATH):
            st.warning("ℹ Vector store not found. Upload a document to create one.")
            return None, None

        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        llm = ChatGroq(model="openai/gpt-oss-120b", api_key=groq_api_key, temperature=0.2, max_tokens=2000)

        st.session_state['db_instance'] = db
        st.session_state['llm_instance'] = llm
        return db, llm

    except Exception as e:
        st.error(f"❌ Model initialization failed: {e}")
        st.code(traceback.format_exc())
        return None, None

# -------------------------------
# Audio Recording
# -------------------------------
# if "is_recording" not in st.session_state:
#     st.session_state.is_recording = False

# def record_audio(filename=RECORDING_PATH, rate=16000, chunk=1024):
#     os.makedirs(RECORDINGS_DIR, exist_ok=True)
#     p = pyaudio.PyAudio()
#     try:
#         stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)
#     except Exception as e:
#         st.error(f"Could not open microphone stream: {e}")
#         p.terminate()
#         return None

#     frames = []
#     while st.session_state.is_recording:
#         try:
#             data = stream.read(chunk, exception_on_overflow=False)
#             frames.append(data)
#         except Exception as e:
#             print("Stream read error:", e)

#     stream.stop_stream()
#     stream.close()
#     p.terminate()

#     try:
#         wf = wave.open(filename, "wb")
#         wf.setnchannels(1)
#         wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
#         wf.setframerate(rate)
#         wf.writeframes(b"".join(frames))
#         wf.close()
#         return filename
#     except Exception as e:
#         print("❌ Failed to save recording:", e)
#         return None

# def start_recording():
#     st.session_state.is_recording = True
#     thread = threading.Thread(target=record_audio)
#     thread.start()
#     st.session_state.recording_thread = thread

# def stop_recording():
#     st.session_state.is_recording = False
#     if "recording_thread" in st.session_state:
#         st.session_state.recording_thread.join()

# -------------------------------
# STT & TTS
# -------------------------------
# @st.cache_resource
# def load_whisper_model():
#     try:
#         return whisper.load_model("base")
#     except Exception as e:
#         st.error(f"Could not load Whisper model: {e}")
#         return None

# whisper_model = load_whisper_model()

# def transcribe_audio(audio_path):
#     if whisper_model is None:
#         return ""
#     try:
#         result = whisper_model.transcribe(audio_path, task="transcribe")
#         return result.get("text", "").strip()
#     except Exception as e:
#         st.warning(f"Whisper transcription failed: {e}")
#         return ""

# def speak_text_to_file(text, lang="en"):
#     os.makedirs("tts", exist_ok=True)
#     try:
#         filename = os.path.join("tts", f"tts_{uuid.uuid4().hex}.mp3")
#         tts = gTTS(text=text, lang=lang)
#         tts.save(filename)
#         return filename
#     except Exception as e:
#         st.warning(f"TTS failed: {e}")
#         return None

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("📂 Upload & Controls")
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "pptx", "txt", "jpg", "jpeg", "png"]
    )
    if uploaded_file:
        temp_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        with st.spinner("Processing document..."):
            process_uploaded_file(temp_path)
        st.info("✨ Vector store updated! You can now ask queries based on this document.")

    st.markdown("---")
    st.subheader("💬 Chat History")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    if "active_chat" not in st.session_state:
        st.session_state.active_chat = None

    if st.button("➕ New Chat"):
        new_chat_session()
        st.rerun()

    if st.session_state.chat_history:
        for chat_id, chat in list(st.session_state.chat_history.items()):
            col1, col2 = st.columns([8, 2])
            with col1:
                if st.button(chat["title"], key=f"open_{chat_id}"):
                    switch_chat(chat_id)
            with col2:
                if st.button("🗑", key=f"del_{chat_id}"):
                    delete_chat(chat_id)
                    st.rerun()
    else:
        st.write("No saved chats yet.")

    if st.button("🧹 Clear All History"):
        st.session_state.chat_history = {}
        save_chat_history({})
        st.session_state.active_chat = None
        st.rerun()
if uploaded_file:
    temp_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Uploaded: {uploaded_file.name}")

    with st.spinner("Processing document..."):
        process_uploaded_file(temp_path)
    st.info("✨ Vector store updated! You can now ask queries based on this document.")

    # --- Summarization Feature ---
    st.markdown("---")
    st.subheader("🧠 Document Summary")
    db, llm = initialize_models()
    if llm:
        if st.button("📄 Generate Summary"):
            with st.spinner("Summarizing document..."):
                summary = summarize_document(temp_path, llm)
            if summary:
                st.success("✅ Summary generated successfully!")
                st.markdown(f"### 📘 Summary of *{uploaded_file.name}*")
                st.write(summary)
                st.download_button(
                    label="💾 Download Summary",
                    data=summary,
                    file_name=f"summary_{uploaded_file.name.split('.')[0]}.txt",
                    mime="text/plain"
                )
    else:
        st.warning("⚠ LLM not initialized. Please set your GROQ_API_KEY.")

# -------------------------------
# Main Chat
# -------------------------------
db, llm = initialize_models()
if db is None or llm is None:
    st.info("📌 Upload a document to create the vector store and start chatting.")
    st.stop()

st.title("🔍 Cyber law enforcement assistant ")
st.markdown("Ask questions about cybercrime investigation, cybersecurity laws, and digital evidence handling.")

if not st.session_state.active_chat:
    new_chat_session()

chat_id = st.session_state.active_chat
messages = st.session_state.chat_history[chat_id]["messages"]

# display history
for message in messages:
    role = message.get("role", "user")
    with st.chat_message(role):
        st.markdown(message.get("content", ""))

st.markdown("🎙 **Voice Input / Language Selection**")
lang_options = {
    "English": "en", "Arabic": "ar", "Hindi": "hi", "Marathi": "mr", "Gujarati": "gu",
    "Bengali": "bn", "Tamil": "ta", "Telugu": "te", "Kannada": "kn", "Malayalam": "ml", "Punjabi": "pa"
}
if "detected_lang" not in st.session_state:
    st.session_state.detected_lang = "en"

selected_lang = st.selectbox("Select language", options=list(lang_options.keys()), index=0)
selected_lang_code = lang_options[selected_lang]

user_query = None
# col1, col2 = st.columns(2)
# if col1.button("🎤 Start Recording"):
#     start_recording()
#     st.info("🎙 Recording... Press Stop when done.")
# if col2.button("⏹ Stop Recording"):
#     stop_recording()
#     if os.path.exists(RECORDING_PATH):
#         with open(RECORDING_PATH, "rb") as f:
#             st.audio(f.read(), format="audio/wav")
#         user_query = transcribe_audio(RECORDING_PATH)
#         st.session_state.detected_lang = selected_lang_code
#         if user_query:
#             st.success(f"🗣 Detected ({selected_lang_code}): {user_query}")
#         else:
#             st.warning("⚠ Could not detect speech.")
#     else:
#         st.warning("⚠ No audio file found.")

if not user_query:
    user_query = st.chat_input("What is your question?")

# -------------------------------
# Chat Processing
# -------------------------------
if user_query:
    if len(messages) == 0:
        st.session_state.chat_history[chat_id]["title"] = user_query[:40]
    messages.append({"role": "user", "content": user_query})
    save_chat_history(st.session_state.chat_history)

    with st.chat_message("user"):
        st.markdown(user_query)
    with st.chat_message("assistant"):
        placeholder = st.empty()

    with st.spinner("🤔 Thinking..."):
        try:
            db, llm = initialize_models()
            docs = db.similarity_search(user_query, k=5)
            context_text = "\n\n".join([doc.page_content for doc in docs])

            # UPDATED PROMPT
            prompt = f"""You are an expert Cyber Law Enforcement Assistant specializing in:
- UAE Cyber Laws
- India Cyber Laws
- Cybercrime investigation
- Cybersecurity procedures
- Digital forensics and evidence handling

Use the following document context if relevant:
{context_text}

Guidelines:
1. If the user's question relates to cybersecurity, cybercrime, or cyber law enforcement — use both your knowledge and the context above to answer clearly and accurately.
2. If the context lacks sufficient data, rely on your general knowledge — but stay strictly within the cyber law, cybercrime, or cybersecurity domains.
3. If the question is unrelated to these topics, respond exactly with:
   "❌ I can only provide guidance on cyber law enforcement, cybersecurity, or cybercrime topics."
4. Always answer factually and concisely, referencing UAE laws where possible.

User Question: {user_query}

Answer:
"""

            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)

            # Display answer and save
            placeholder.markdown(answer)
            messages.append({"role": "assistant", "content": answer})
            save_chat_history(st.session_state.chat_history)

            # Mark response ready so feedback UI only appears after full response is generated and saved
            idx = len(messages) - 1
            key_prefix = f"{chat_id}_{idx}"
            st.session_state[f"response_ready_{key_prefix}"] = True

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.code(traceback.format_exc())
            answer = "Sorry, an error occurred."
            messages.append({"role": "assistant", "content": answer})
            save_chat_history(st.session_state.chat_history)

    # -------------------------------
    # Feedback Section (robust)
    # -------------------------------
    # Use keys derived from chat_id and message index so each assistant reply has its own feedback UI/state
    idx = len(messages) - 1
    key_prefix = f"{chat_id}_{idx}"
    response_ready_flag = st.session_state.get(f"response_ready_{key_prefix}", False)
    already_done_flag = st.session_state.get(f"fb_done_{key_prefix}", False)

    if response_ready_flag and not already_done_flag:
        st.markdown("---")
        st.subheader("💬 Feedback")

        # Use a selectbox with a neutral default so something isn't pre-selected.
        fb_options = ["Select an option", "👍 Satisfied", "👎 Unsatisfied"]
        fb_key = f"fb_choice_{key_prefix}"
        # Ensure the key exists in session_state to preserve selection across reruns
        if fb_key not in st.session_state:
            st.session_state[fb_key] = fb_options[0]

        fb_choice = st.selectbox("Was this response helpful?", fb_options, key=fb_key)

        # AUTO-log satisfied feedback once (so user sees immediate thank you and we record it)
        if fb_choice == "👍 Satisfied":
            # show thank you immediately
            st.markdown("✅ Thank you for your feedback!")
            # ensure we only log once
            auto_flag_key = f"fb_auto_logged_{key_prefix}"
            if not st.session_state.get(auto_flag_key, False):
                # Log feedback (no reason)
                try:
                    log_feedback(user_query, answer, "Satisfied", reason=None)
                except Exception as e:
                    st.warning(f"Failed to log feedback: {e}")
                st.session_state[auto_flag_key] = True
                st.session_state[f"fb_done_{key_prefix}"] = True
                st.success("🙏 Feedback recorded. Thank you!")
                # no rerun here to avoid immediate loop; UI will reflect the 'done' state on next interaction

        elif fb_choice == "👎 Unsatisfied":
            # show reason textbox and Submit button
            reason_key = f"fb_reason_{key_prefix}"
            if reason_key not in st.session_state:
                st.session_state[reason_key] = ""
            reason = st.text_area(
                "Please tell us why you are unsatisfied:",
                key=reason_key,
                placeholder="Describe what was wrong or missing in the answer..."
            )

            if st.button("Submit Feedback", key=f"fb_submit_{key_prefix}"):
                # require non-empty reason
                if not reason or not reason.strip():
                    st.warning("Please provide a reason before submitting.")
                else:
                    try:
                        log_feedback(user_query, answer, "Unsatisfied", reason=reason.strip())
                    except Exception as e:
                        st.warning(f"Failed to log feedback: {e}")
                    st.session_state[f"fb_done_{key_prefix}"] = True
                    st.success("✅ Your feedback has been submitted. Thank you!")
                    st.rerun()
        else:
            st.info("Select '👍 Satisfied' or '👎 Unsatisfied' to provide feedback.")

    elif st.session_state.get(f"fb_done_{key_prefix}", False):
        st.info("✅ Feedback already submitted for this response.")

    # -------------------------------
    # TTS
    # -------------------------------
    # if FFMPEG_OK:
    #     tts_file = speak_text_to_file(answer, lang=st.session_state.get("detected_lang", "en"))
    #     if tts_file and os.path.exists(tts_file):
    #         with open(tts_file, "rb") as f:
    #             st.audio(f.read(), format="audio/mp3")

st.markdown("---")
# st.markdown("<div style='text-align:center;color:#666;'>Powered by GROQ, LangChain, FAISS, Whisper & gTTS</div>", unsafe_allow_html=True)
