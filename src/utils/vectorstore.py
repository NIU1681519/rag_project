import uuid
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL_NAME, BATCH_SIZE
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Dispositivo para embeddings: {device}")

def create_vectorstore(docs, persist_directory):
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device}
    )
    vectordb = Chroma(embedding_function=embedding, persist_directory=persist_directory)

    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Indexando"):
        batch = docs[i:i + BATCH_SIZE]
        vectordb.add_texts(
            texts=[doc.page_content for doc in batch],
            metadatas=[doc.metadata for doc in batch],
            ids=[str(uuid.uuid4()) for _ in batch]
        )
    return vectordb


def load_vectorstore(persist_directory):
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device}
    )
    return Chroma(persist_directory=persist_directory, embedding_function=embedding)
