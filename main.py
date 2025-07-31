import os
import re
import ollama
import chromadb
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings


# Global Variables
DB_PATH = "chroma_db"
vector_db = None
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 1. Extract Text from PDF (Ignoring Images)
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text.strip()

# 2. Title-Aware Chunking
def split_by_titles(text):
    sections = re.split(r'\n\s*(Chapter|Section|Unit|Lesson|Topic|Subsection)\s*\d*[:.]?', text)
    return [s.strip() for s in sections if s.strip()]

# 3. Recursive Character Splitting
def refine_chunks(chunks):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    final_chunks = []
    for chunk in chunks:
        final_chunks.extend(text_splitter.split_text(chunk))
    return final_chunks

# 4. Process PDF and Store Embeddings in ChromaDB
def process_and_store_pdf(pdf_path):
    global vector_db
    print(f"\U0001F4C4 Processing file: {pdf_path} ...")
    raw_text = extract_text_from_pdf(pdf_path)
    structured_chunks = split_by_titles(raw_text)
    final_chunks = refine_chunks(structured_chunks)
    vector_db = Chroma.from_texts(final_chunks, embeddings, persist_directory=DB_PATH)
    vector_db.persist()
    print(f"\u2705 PDF processed & stored with {len(final_chunks)} chunks!")

# 5. Load ChromaDB
def load_vector_db():
    global vector_db
    if os.path.exists(DB_PATH):
        vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        print("\u26A0 No database found. Please upload a file first.")

# Direct Prompting with LLaMA 3
def direct_query(prompt):
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    ai_response = response['message']['content']
    print("\n\U0001F916 AI Response:\n", ai_response)
    return ai_response

# 6. Query Processing with Embedded Query
def query_pdf(question):
    if vector_db is None:
        message = "\u26A0 No database found. Please upload a file first."
        print(message)
        return message

    print(f"\U0001F50D Embedding query & searching...\n")
    query_embedding = embeddings.embed_query(question)
    results = vector_db.similarity_search_by_vector(query_embedding, k=5)
    context = "\n".join([r.page_content for r in results])
    prompt = f"Based on the textbook content:\n\n{context}\n\nAnswer the question: {question}"
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    ai_response = response['message']['content']
    print("\n\U0001F916 AI Response:\n")
    print(ai_response)
    return ai_response

# 7. Main Menu with Switch-Case
def main():
    load_vector_db()
    while True:
        print("\n\U0001F4DA Textbook RAG Chatbot")
        print("1\uFE0F⃣ Upload PDF")
        print("2\uFE0F⃣ Ask a Question")
        print("3\uFE0F⃣ Exit")
        choice = input("Enter your choice: ")

        match choice:
            case "1":
                pdf_path = input("\U0001F4C2 Enter PDF path: ").strip()
                if os.path.exists(pdf_path):
                    process_and_store_pdf(pdf_path)
                else:
                    print("\u274C File not found!")
            case "2":
                query = input("\u2753 Enter your question: ").strip()
                query_pdf(query)
            case "3":
                print("\U0001F44B Exiting...")
                break
            case _:
                print("\u26A0 Invalid choice! Please select again.")

if __name__ == "__main__":
    main()
