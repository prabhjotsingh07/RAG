import os
import io
import fitz
import pytesseract
from PIL import Image
import ollama
import chromadb
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import TextProcessorForm, UserCreationForm
from datetime import datetime

# Constants
COLLECTION_NAME = "pdfs_collection"
# Add this at the document processing stage to store document metadata
doc_chunks = {}  # Store chunk-to-document mapping
PDF_FOLDER_PATH = "/home/prabh/Desktop/chunking/pdfs"
collection = None

from django.http import HttpResponse

# Path to the text file where history is stored
HISTORY_FILE_PATH = "/home/prabh/Desktop/llm/history.txt"

from django.shortcuts import render

def history(request):
    try:
        with open(HISTORY_FILE_PATH, "r") as file:
            history_content = file.read()
    except FileNotFoundError:
        history_content = "No history available."

    return render(request, 'text_processor/history.html', {'history': history_content})


def initialize_pdf_collection():
    print("Initializing PDF collection...")
    client = ollama.Client()
    chroma_client = chromadb.Client()

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"Error deleting collection: {e}")

    global collection
    collection = chroma_client.create_collection(COLLECTION_NAME)
    print(f"Created new collection: {COLLECTION_NAME}")

    for filename in os.listdir(PDF_FOLDER_PATH):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER_PATH, filename)
            print(f"Processing PDF: {filename}")

            pdf_text = extract_text_from_pdf(pdf_path)
            if not pdf_text.strip():
                print(f"No text found in {filename}. Skipping.")
                continue

            chunks = split_text(pdf_text)
            print(f"Split PDF into {len(chunks)} chunks")

            # Add each chunk's text and embedding to the collection
        for i, chunk_text in enumerate(chunks, 1):
            chunk_id = f"{filename}_chunk{i}"
            embedding = get_embedding(chunk_text, client)
            collection.add(
                documents=[chunk_text],
                embeddings=[embedding],
                ids=[chunk_id],
                metadatas=[{"source": filename}]  # Add metadata with source filename
            )
            doc_chunks[chunk_text] = filename  # Store the mapping
            print(f"Added chunk {i} of {filename} to ChromaDB.")

    print("PDF collection initialization complete")

def split_text(text: str, max_chunk_size: int = 200) -> list:
    print("Starting text splitting...")
    chunks = []
    current_chunk = ""

    for char in text:
        current_chunk += char
        if len(current_chunk) >= max_chunk_size:
            if '.' in current_chunk:
                split_index = current_chunk.rfind('.') + 1
            elif ',' in current_chunk:
                split_index = current_chunk.rfind(',') + 1
            elif ' ' in current_chunk:
                split_index = current_chunk.rfind(' ')
            else:
                split_index = max_chunk_size

            chunks.append(current_chunk[:split_index].strip())
            current_chunk = current_chunk[split_index:].strip()

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    print(f"Text split into {len(chunks)} chunks")
    return chunks

def extract_text_from_pdf(pdf_path: str) -> str:
    print(f"Extracting text from PDF: {pdf_path}")
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, 1):
                print(f"Processing page {page_num}")
                text += page.get_text()
                for img_index, img in enumerate(page.get_images(full=True), start=1):
                    print(f"Processing image {img_index} on page {page_num}")
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    image_text = pytesseract.image_to_string(image)
                    text += f"\n[Image {img_index} Text]: {image_text}"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

def get_embedding(text: str, client) -> list:
    print("Generating embedding...")
    result = client.embeddings(model="all-minilm", prompt=text)
    print("Embedding generated successfully")
    return result['embedding']

def get_llama_response(context: str, query: str, output_language: str, client) -> str:
    print("Generating Llama response...")
    prompt = f"""You are a helpful AI assistant. Use the following context to answer the question provided.
    Give the response in {output_language} language.
   
    Context:
    {context}
   
    Question:
    {query}
   
    Answer based on the context:"""
    try:
        response = client.chat(
            model="llama3.2:1b",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        print("Llama response generated successfully")
        return response.get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"Error with Llama3.2 API: {e}")
        return "No response received from Llama3.2."

HISTORY_FILE_PATH = "/home/prabh/Desktop/llm/history.txt"

def log_history(query, output_language, output):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current date and time
    with open(HISTORY_FILE_PATH, "a") as file:
        file.write(f"Timestamp: {current_time}\n")  # Write the timestamp
        file.write(f"Query: {query}\n")
        file.write(f"Output Language: {output_language}\n")
        file.write(f"Output: {output}\n")
        file.write("-" * 50 + "\n")


@login_required
def text_processor(request):
    global collection
    if collection is None:
        initialize_pdf_collection()
       
    print("Processing text request...")
    if request.method == 'POST':
        form = TextProcessorForm(request.POST)
        if form.is_valid():
            print("Form is valid, processing input...")
            user_text = form.cleaned_data['user_text']
            output_language = form.cleaned_data['output_language']

            client = ollama.Client()
            print("Ollama client initialized")

            query_chunks = split_text(user_text)
            results = []

            for i, query_text in enumerate(query_chunks, 1):
                print(f"Processing chunk {i} of {len(query_chunks)}")
                query_embedding = get_embedding(query_text, client)
                query_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=2,
                    include=["documents", "distances", "metadatas"]
                )
                print(f"Retrieved {len(query_results['documents'][0])} matches for chunk {i}")
               
                closest_matches = query_results["documents"][0]
                distances = query_results["distances"][0]
                metadatas = query_results["metadatas"][0]

                for idx, (match_doc, distance, metadata) in enumerate(zip(closest_matches, distances, metadatas), start=1):
                    print(f"\nClosest Match {idx}:")
                    print(f"Matched Chunk: {match_doc}")
                    print(f"Distance: {distance}")
                    source_pdf = metadata.get("source", "Unknown source")
                    print(f"Source PDF: {source_pdf}")

                context = " ".join(closest_matches)
                llama_response = get_llama_response(context, user_text, output_language, client)

                results.append({
                    'query': query_text,
                    'response': llama_response,
                    'matches': list(zip(closest_matches, distances)),
                    'source_pdf': source_pdf  # Include source_pdf here
                })

            print("Rendering results page")

            # Log the history
            log_history(query_text, output_language, llama_response)

            return render(request, 'text_processor/result.html', {
                'results': results,
                'original_query': user_text
            })
    else:
        form = TextProcessorForm()
    return render(request, 'text_processor/process.html', {'form': form})

def signup(request):
    print("Processing signup request...")
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            print("Form is valid, creating user...")
            user = form.save()
            login(request, user)
            print("User created and logged in successfully")
            return redirect('text_processor')
        else:
            print("Form validation failed")
    else:
        form = UserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})

def logout_view(request):
    print("Processing logout request...")
    logout(request)
    messages.success(request, 'Logged out successfully!')
    print("User logged out successfully")
    return redirect('login')
	
