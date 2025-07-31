from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from src.config import LLM_MODEL_NAME

def build_rag_chain(vectorstore):
    print(f"[INFO] Usando modelo Ollama: {LLM_MODEL_NAME}")

    llm = Ollama(model=LLM_MODEL_NAME)

    retriever = vectorstore.as_retriever()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs = {
            "prompt": None,
        }
    )
