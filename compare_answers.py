
import csv
import glob
import os

from pdf_qa import (
    process_pdf,
    retrieve_chunks,
    generate_without_retrieval,
    generate_with_retrieval
)

# 1. Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "dataset", "uploaded_files")
QUESTIONS_FILE = os.path.join(BASE_DIR, "dataset", "questions.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset", "comparison_results.csv")

# 2. Find uploaded PDF
pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF found in uploaded_files folder.")

pdf_path = max(pdf_files, key=os.path.getmtime)

print("PDF:", os.path.basename(pdf_path))

# 3. Process PDF
result = process_pdf(
    pdf_path,
    "comparison_test_collection",
    chunk_size=500,
    chunk_overlap=100
)

collection = result["collection"]

print("PDF processed successfully!")
print("Total chunks:", len(result["chunks"]))

# 4. Read questions
with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    questions = [row["Question"] for row in reader]

# 5. Generate RAG and Non-RAG answers
results = []

for index, question in enumerate(questions, start=1):

    print(f"\nProcessing question {index}/{len(questions)}")

    # Non-RAG answer
    non_rag_answer = generate_without_retrieval(question)

    # Retrieve Top-5 chunks
    retrieved_chunks = retrieve_chunks(collection, question)

    # RAG answer
    rag_answer = generate_with_retrieval(
        question,
        retrieved_chunks
    )

    results.append({
        "Question": question,
        "Non_RAG_Answer": non_rag_answer,
        "RAG_Answer": rag_answer,
        "Retrieved_Chunks": len(retrieved_chunks)
    })

# 6. Save comparison results
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
    fieldnames = [
        "Question",
        "Non_RAG_Answer",
        "RAG_Answer",
        "Retrieved_Chunks"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("\nAll questions processed successfully!")
print("Comparison CSV saved at:", OUTPUT_FILE)
