
import os
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


# 1. Project paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "chroma_db")

MODEL_NAME = "llama3.2:latest"


# 2. Initialize ChromaDB

embedding_function = DefaultEmbeddingFunction()

client = chromadb.PersistentClient(path=DB_PATH)


# 3. Extract text from PDF

def extract_pdf_text(pdf_path):

    reader = PdfReader(pdf_path)

    extracted_text = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text:
            extracted_text.append(text)

    return "\n".join(extracted_text), len(reader.pages)


# 4. Split text into chunks

def split_text(text, chunk_size=500, chunk_overlap=100):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_text(text)

    return chunks


# 5. Create ChromaDB collection

def create_collection(collection_name, chunks):

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    for start in range(0, len(chunks), 100):

        batch = chunks[start:start + 100]

        ids = [
            f"chunk_{i}"
            for i in range(start, start + len(batch))
        ]

        collection.add(
            ids=ids,
            documents=batch
        )

    return collection


# 6. Retrieve Top-5 relevant chunks

def retrieve_chunks(collection, question):

    results = collection.query(
        query_texts=[question],
        n_results=min(5, collection.count())
    )

    documents = results["documents"][0]

    return documents


# 7. Initialize Ollama

llm = OllamaLLM(
    model=MODEL_NAME,
    temperature=0.2
)


# 8. Generate answer without retrieval

def generate_without_retrieval(question):

    prompt = f"""
    Answer the following question using your knowledge.

    Question: {question}

    Answer:
    """

    return llm.invoke(prompt)


# 9. Generate answer with retrieval

def generate_with_retrieval(question, retrieved_chunks):

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    You are a PDF Question Answering Assistant.

    Answer the question using only the given PDF context.

    If the answer is not available in the context,
    clearly say that the information is not available
    in the uploaded PDF.

    PDF Context:
    {context}

    Question:
    {question}

    Answer:
    """

    return llm.invoke(prompt)


# 10. Complete PDF processing

def process_pdf(pdf_path, collection_name,
                chunk_size=500, chunk_overlap=100):

    text, total_pages = extract_pdf_text(pdf_path)

    if not text.strip():
        raise ValueError("No readable text found in PDF.")

    chunks = split_text(
        text,
        chunk_size,
        chunk_overlap
    )

    collection = create_collection(
        collection_name,
        chunks
    )

    return {
        "text": text,
        "pages": total_pages,
        "chunks": chunks,
        "collection": collection
    }
