import os
from src.utils.loader import load_procqa_jsonl
from src.utils.splitter import split_documents
from src.utils.vectorstore import create_vectorstore, load_vectorstore
from src.rag_engine import build_rag_chain
from src.config import DATA_C, DATA_CPP, DATA_CSH, DB_DIR

def load_all_documents():
    docs = []
    if os.path.exists(DATA_C): docs += load_procqa_jsonl(DATA_C)
    if os.path.exists(DATA_CPP): docs += load_procqa_jsonl(DATA_CPP)
    if os.path.exists(DATA_CSH): docs += load_procqa_jsonl(DATA_CSH)
    return docs

def check_vectorstore_nonempty(vectordb):
    try:
        return len(vectordb.similarity_search("test", k=1)) > 0
    except Exception:
        return False

def main():
    if not os.path.exists(DB_DIR):
        docs = load_all_documents()
        if not docs:
            print("❌ No hay documentos para indexar.")
            return
        chunks = split_documents(docs)
        vectordb = create_vectorstore(chunks, DB_DIR)
    else:
        vectordb = load_vectorstore(DB_DIR)

    if not check_vectorstore_nonempty(vectordb):
        print("❌ Vectorstore no funcional.")
        return

    rag_chain = build_rag_chain(vectordb)

    print("\n🩺 Asistente médico listo. Escribe tu pregunta o 'salir'.")
    while True:
        query = input("Tú: ")
        if query.lower() in ["salir", "exit", "quit"]:
            break
        print("\nAsistente está pensando...\n")
        result = rag_chain.invoke(query).get("result", "").strip()
        print("Asistente:", result or "No encontré información relevante.")

if __name__ == "__main__":
    main()
