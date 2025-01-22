import os
import fitz  # PyMuPDF for handling PDFs
import ollama
import chromadb
import pandas as pd

import pytesseract       #these for text as images and tables
from PIL import Image
import fitz  # PyMuPDF
import io

def get_embedding(text, client):
    """Get embedding vector for a text using Ollama."""
    result = client.embeddings(
        model="all-minilm",
        prompt=text
    )
    return result['embedding']

output_language=""

# Create Ollama client
client = ollama.Client()

def get_llama_response(context: str, query: str, client):
    """Get response from the Llama3.2 model using context and query."""
    prompt = f"""You are a helpful AI assistant. Use the following context to answer the question provided. 
    Give the response in {output_language} language

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
        return response.get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"Error with Llama3.2 API: {e}")
        return "No response received from Llama3.2."

def split_text(text: str, max_chunk_size: int = 500) -> list:
    """Split text into chunks with prioritized breaks: first by full stop, then comma, then space."""
    
    chunks = []
    current_chunk = ""

    for char in text:
        # Add the current character to the chunk
        current_chunk += char

        # Check if the current chunk exceeds the max size
        if len(current_chunk) >= max_chunk_size:
            # Try to break by full stop
            if '.' in current_chunk:
                split_index = current_chunk.rfind('.') + 1
            # If no full stop, try to break by comma
            elif ',' in current_chunk:
                split_index = current_chunk.rfind(',') + 1
            # If no comma, break by the last space
            elif ' ' in current_chunk:
                split_index = current_chunk.rfind(' ')
            else:
                # No suitable delimiter; split at max_chunk_size
                split_index = max_chunk_size

            # Add the chunk to the list
            chunks.append(current_chunk[:split_index].strip())
            # Start a new chunk with the remaining text
            current_chunk = current_chunk[split_index:].strip()

    # Add the last chunk if any
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file, including tables and text in images."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                # Extract text from the page
                text += page.get_text()
               
                # Check for images in the page
                for img_index, img in enumerate(page.get_images(full=True), start=1):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                   
                    # Open image using PIL
                    image = Image.open(io.BytesIO(image_bytes))
                   
                    # Perform OCR on the image
                    image_text = pytesseract.image_to_string(image)
                    text += f"\n[Image {img_index} Text]: {image_text}"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Create or get the collection
collection_name = "pdfs_collection"

# Delete the collection if it exists and create a new one
try:
    chroma_client.delete_collection(collection_name)
except Exception as e:
    print(f"Error deleting collection: {e}")

# Create a new collection
collection = chroma_client.create_collection(collection_name)
print(f"Created new '{collection_name}' collection.")

# Folder containing PDF files
pdf_folder_path = "/home/prabh/Desktop/chunking/pdfs"  # Update this path to your PDF folder path

# Add this at the document processing stage to store document metadata
doc_chunks = {}  # Store chunk-to-document mapping

# When adding documents to the collection, modify this part:
for filename in os.listdir(pdf_folder_path):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder_path, filename)
        print(f"Processing {filename}...")

        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text.strip():
            print(f"No text found in {filename}. Skipping.")
            continue

        # Split text into chunks
        chunks = split_text(pdf_text)

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

# Query input loop
while True:
    user_text = input("\nEnter a query text (or -1 to exit): ")
    output_language = input("\nEnter the language in which you want the response: ")
    
    # Check if user wants to exit
    if user_text == '-1':
        print("Exiting the program...")
        break
        
    # Process the query
    query_chunks = split_text(user_text)
    
    for query_text in query_chunks:
        # Get the embedding for the query chunk
        query_embedding = get_embedding(query_text, client)
        
        # Use ChromaDB's query method to find closest matches
        query_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,  # Retrieve top 2 closest matches
            include=["documents", "distances", "metadatas"]
        )
        
        print(f"\nQuery Chunk: {query_text}")
        closest_matches = query_results["documents"][0]
        distances = query_results["distances"][0]
        metadatas = query_results["metadatas"][0]
        
        # Print matches and distances
        for idx, (match_doc, distance, metadata) in enumerate(zip(closest_matches, distances, metadatas), start=1):
            print(f"\nClosest Match {idx}:")
            print(f"Matched Chunk: {match_doc}")
            print(f"Distance: {distance}")
            source_pdf = metadata.get("source", "Unknown source")
            print(f"Source PDF: {source_pdf}")
        
        # Combine the closest matches as context for Llama3.2
        context = " ".join(closest_matches)
        print("\nGetting response from Llama3.2...")
        llama_response = get_llama_response(context, user_text, client)
        print("\nLlama3.2 Response:")
        print(llama_response)
        print("-" * 50)
