import json, os, time
from langchain.schema import Document

def load_plain_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": os.path.basename(filepath)})]

def load_qa_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Document(
            page_content=f"Pregunta: {item['question']}\nRespuesta: {item['answer']}",
            metadata={"source": os.path.basename(filepath)}
        ) for item in data if item.get("question") and item.get("answer")
    ]

def load_squad_format_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    total_qa = 0
    for entry in data.get("data", []):
        for para in entry.get("paragraphs", []):
            context = para.get("context", "").strip()
            for qa in para.get("qas", []):
                total_qa += 1
                question = qa.get("question", "").strip()
                for ans in qa.get("answers", []):
                    answer_text = ans.get("text", "").strip()
                    content = f"PREGUNTA: {question}\nRESPUESTA: {answer_text}"
                    docs.append(Document(page_content=content, metadata={"context": context}))
    print(f"[INFO] Total preguntas leídas en SQuAD JSON: {total_qa}")
    return docs


def load_procqa_jsonl(filepath):
    """
    Carga un archivo JSONL estilo ProCQA: cada línea es un objeto con 'title', 'question' y 'answer'.
    Retorna una lista de Document para usar con RAG.
    """
    documents = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue  # Ignora líneas vacías
            try:
                item = json.loads(line)
                title = item.get("title", "").strip()
                question = item.get("question", "").strip()
                answer = item.get("answer", "").strip()

                if question and answer:
                    content = f"TÍTULO: {title}\nPREGUNTA: {question}\nRESPUESTA: {answer}"
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": os.path.basename(filepath),
                                  "question_id": item.get("question_id"),
                                  "answer_id": item.get("answer_id")}
                    ))
            except json.JSONDecodeError as e:
                print(f"[WARN] Línea inválida en JSON: {e}")

    print(f"[INFO] Total documentos cargados: {len(documents)}")
    return documents

