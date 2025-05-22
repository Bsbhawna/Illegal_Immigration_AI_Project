# src/rag_chatbot/build_vectorstore.py

import os
import csv
from datetime import datetime
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# === CONFIG ===
PROJECT_ROOT = "C:/Users/sharmbha/Downloads/Bhawna_project/Illegal_Immigration_AI_Project"

DOMAIN_KNOWLEDGE_PATH = os.path.join(PROJECT_ROOT, "data", "domain_knowledge")
AUTO_SUMMARY_PATH = os.path.join(PROJECT_ROOT, "data", "auto_summaries")
CUSTOM_QA_PATH = os.path.join(PROJECT_ROOT, "data", "custom_qa")
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "src", "rag_chatbot", "faiss_index")
LOG_PATH = os.path.join(PROJECT_ROOT, "data", "logs", "chat_log.csv")

# ✅ Load all .txt files from given folders
def load_documents_from_folders(folders):
    docs = []
    for folder in folders:
        if not os.path.isdir(folder): continue
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder, filename)
                try:
                    loader = TextLoader(filepath, encoding="utf-8")
                    loaded = loader.load()
                    docs.extend(loaded)
                    print(f"📄 Loaded {filename} ({len(loaded)} chunks)")
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
    return docs

# ✅ Feedback logger
def log_chat(query, answer):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), query, answer])

# ✅ Main vectorstore builder
def main():
    print("⏳ Loading documents from all knowledge sources...")
    documents = load_documents_from_folders([
        DOMAIN_KNOWLEDGE_PATH,
        AUTO_SUMMARY_PATH,
        CUSTOM_QA_PATH  # ✅ Added this
    ])

    print(f"📚 Total loaded documents: {len(documents)}")
    print(f"⏳ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = splitter.split_documents(documents)

    docs_split = [doc for doc in docs_split if doc.page_content.strip() != ""]
    print(f"✅ Chunks ready for embedding: {len(docs_split)}")

    if not docs_split:
        raise ValueError("❌ No valid non-empty documents found to embed.")

    print("🧠 Loading HuggingFace embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"⚙️ Creating FAISS index...")
    vectorstore = FAISS.from_documents(docs_split, embeddings)

    print(f"💾 Saving FAISS index to {FAISS_INDEX_PATH} ...")
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)

    print("✅ FAISS vectorstore build complete and saved!")

if __name__ == "__main__":
    main()
