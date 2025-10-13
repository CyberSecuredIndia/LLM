import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Base directories
BASE_DIR = "data"
VECTORDB_DIR = "vectorstore/db_faiss"

# ---------- Helper Functions ---------- #
def load_with_metadata(file_path: str, jurisdiction: str):
    """
    Load a document with the right loader and attach jurisdiction + filename metadata.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".pptx":
        loader = UnstructuredPowerPointLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        print(f"⚠ Skipping unsupported file: {file_path}")
        return []

    docs = loader.load()
    for doc in docs:
        doc.metadata["jurisdiction"] = jurisdiction
        doc.metadata["source_file"] = os.path.basename(file_path)
    return docs


def gather_all_docs():
    """
    Traverse data/ folder and load docs with jurisdiction tagging.
    Expecting subfolders: data/UAE/ and data/India/
    """
    all_docs = []
    for jurisdiction in ["UAE", "India"]:
        folder = os.path.join(BASE_DIR, jurisdiction)
        if not os.path.exists(folder):
            continue
        for root, _, files in os.walk(folder):
            for f in files:
                file_path = os.path.join(root, f)
                all_docs.extend(load_with_metadata(file_path, jurisdiction))
    return all_docs


# ---------- Main Script ---------- #
if __name__ == "__main__":
    print("📂 Loading documents...")
    documents = gather_all_docs()

    print(f"✅ Loaded {len(documents)} documents")

    # Split docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(documents)

    print(f"✂ Split into {len(split_docs)} chunks")

    # Create FAISS index
    db = FAISS.from_documents(split_docs, embeddings)

    # Save FAISS DB
    if not os.path.exists(VECTORDB_DIR):
        os.makedirs(VECTORDB_DIR)
    db.save_local(VECTORDB_DIR)

    print(f"💾 Vector DB saved to {VECTORDB_DIR}")