from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
import faiss
import os


file_path = "ukpga_20250022_en.pdf"
# ------------------------------
# Step 1: Load and split document
# ------------------------------
def load_and_split(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=160
    )
    texts = text_splitter.split_documents(docs)
    return texts


# ------------------------------
# Step 2: Build or load FAISS
# ------------------------------
def build_vector_store(texts, embeddings, index_path="rag_index"):
    embedding_size = len(embeddings.embed_query("hello"))
    index = faiss.IndexFlatL2(embedding_size)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_documents(texts)
    vector_store.save_local(index_path)
    return vector_store


# ------------------------------
# Step 3: RAG Response Function
# ------------------------------
def rag_answer(vector_store, query, model_name="qwen3:1.7b"):
    # Retrieve relevant context
    results = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    # Build prompt for LLM
    prompt = f"""
You are a helpful AI assistant.
Use the provided context to answer the question clearly and concisely.

Context:
{context}

Question:
{query}

Answer:
"""

    # Generate answer from Ollama
    model = ChatOllama(model=model_name)
    response = model.invoke(prompt)

    print("\n🧠 Query:", query)
    print("\n📘 Retrieved Context:\n", context[:400], "...\n")
    print("🤖 Answer:\n", response.content)


# ------------------------------
# Step 4: Main entry point
# ------------------------------
if __name__ == "__main__":
    
    index_path = "rag_index"

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(index_path):
        print("📂 Loading existing FAISS index...")
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("⚙️ Building FAISS index...")
        texts = load_and_split(pdf_path)
        vector_store = build_vector_store(texts, embeddings, index_path)
        print("✅ Index created successfully!")

    # Example query
    rag_answer(vector_store, " Summarize the entire Act in 5–10 bullet points focusing on: - Purpose - Key definitions - Eligibility - Obligations - Enforcement elements ")