import os
import csv
import subprocess
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# === CONFIGURATION ===
try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
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

# === LOAD EMBEDDINGS ===
print(f"{Fore.YELLOW}⏳ Loading HuggingFace embeddings...{Style.RESET_ALL}")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === LOAD LLM ===
print(f"{Fore.YELLOW}🧠 Loading local Mistral LLM...{Style.RESET_ALL}")
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,
    temperature=0.2,
    top_p=0.95,
    repeat_penalty=1.1,
    max_tokens=512,
    verbose=False
)

# === LOAD VECTORSTORE ===
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"{Fore.RED}❌ FAISS index not found! Run build_vectorstore.py first.{Style.RESET_ALL}")

vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# === CREATE RETRIEVAL CHAIN ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=False
)

# === CHAT FUNCTION ===
def chat_with_bot(query: str) -> str:
    try:
        response = qa_chain.run(query)
        return response.strip()
    except Exception as e:
        with open("error_log.txt", "a") as f:
            f.write(f"{datetime.now()} | {query} | {str(e)}\n")
        return f"{Fore.RED}❌ Error generating answer: {e}{Style.RESET_ALL}"

# === LOGGING FUNCTION ===
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
    print(f"\n{Fore.GREEN}🚀 RAG Chatbot (Illegal Immigration Project) Ready!")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input(f"{Fore.CYAN}👤 You: {Style.RESET_ALL}").strip()
        if user_input.lower() in ["exit", "quit"]:
            print(f"{Fore.MAGENTA}👋 Goodbye!{Style.RESET_ALL}")
            break
        if not user_input:
            print(f"{Fore.YELLOW}⚠️ Please enter a valid question.{Style.RESET_ALL}")
            continue

        response = chat_with_bot(user_input)
        print(f"{Fore.GREEN}🤖 Bot:\n{Style.RESET_ALL}{response}\n")

        feedback = input(f"{Fore.CYAN}✅ Was this helpful? (y/n/maybe): {Style.RESET_ALL}").strip().lower()
        if feedback == "y":
            success_flag = "Yes"
        elif feedback == "n":
            success_flag = "No"
        else:
            success_flag = "Maybe"

        log_chat(user_input, response, success_flag)
