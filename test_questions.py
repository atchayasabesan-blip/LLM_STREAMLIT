
import csv

questions = [
    "What is the main topic of this document?",
    "What are the important points discussed?",
    "Explain the introduction.",
    "What is the main objective?",
    "List the key concepts.",
    "What are the advantages mentioned?",
    "What are the disadvantages mentioned?",
    "Explain the methodology.",
    "What are the important findings?",
    "What examples are provided?",
    "What are the applications discussed?",
    "Explain the conclusion.",
    "What are the challenges mentioned?",
    "Summarize the document.",
    "What are the future developments discussed?"
]

with open("dataset/questions.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Question"])
    for question in questions:
        writer.writerow([question])

print("15 questions saved successfully!")
