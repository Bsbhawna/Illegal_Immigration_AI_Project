import os
import csv
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings

# === CONFIGURATION ===
try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Jupyter or interactive environment
    CURRENT_DIR = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..",".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "src", "rag_chatbot", "faiss_index")
PROMPT_PATH = os.path.join(PROJECT_ROOT, "configs", "system_prompt.txt")
LOG_PATH = os.path.join(PROJECT_ROOT, "data", "logs", "chat_log.csv")

# === LOAD SYSTEM PROMPT ===
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    base_prompt = f.read()

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=base_prompt + "\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)
# === SETUP EMBEDDINGS + LLM ===
print("⏳ Loading HuggingFace embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("🧠 Loading local Mistral LLM...")
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,
    verbose=False
)

# === LOAD VECTORSTORE ===
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError("❌ FAISS index not found! Run build_vectorstore.py first.")

vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# === CREATE RETRIEVAL CHAIN WITH CUSTOM PROMPT ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # map_reduce ke bajaye stuff use kar rahe hain
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=False
)
# === CHAT + LOGGING ===
def chat_with_bot(query: str) -> str:
    try:
        response = qa_chain.run(query)
        return response.strip()
    except Exception as e:
        return f"❌ Error generating answer: {e}"

def log_chat(query, answer, success_flag="N/A"):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "user_query", "llm_answer", "success_flag"])
        writer.writerow([datetime.now().isoformat(), query, answer, success_flag])

# === MAIN INTERFACE ===
if __name__ == "__main__":
    print("\n🚀 RAG Chatbot (Illegal Immigration Project) Ready!")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("👤 You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        if not user_input:
            print("⚠️ Please enter a valid question.")
            continue

        response = chat_with_bot(user_input)
        print(f"🤖 Bot:\n{response}\n")

        feedback = input("✅ Was this helpful? (y/n/maybe): ").strip().lower()
        if feedback == "y":
            success_flag = "Yes"
        elif feedback == "n":
            success_flag = "No"
        else:
            success_flag = "Maybe"

        log_chat(user_input, response, success_flag)