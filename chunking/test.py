from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout
from django.contrib import messages
#from .forms import TextProcessorForm
import ollama
import chromadb
import os
import fitz
import pandas as pd
import pytesseract
from PIL import Image
import io


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

def get_embedding(text, client):
    """Get embedding vector for a text using Ollama."""
    result = client.embeddings(
        model="all-minilm",
        prompt=text
    )
    return result['embedding']

def get_llama_response(context: str, query: str, output_language: str, client):
    """Get response from the Llama3.2 model using context and query."""
    prompt = f"""You are a helpful AI assistant. Use the following context to answer the question provided. Give the response in {output_language} language
   
    Context: {context}
   
    Question: {query}
   
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

def text_processor(request):
    if request.method == 'POST':
        form = TextProcessorForm(request.POST)
        if form.is_valid():
            user_text = form.cleaned_data['user_text']
            output_language = form.cleaned_data['output_language']
           
            # Initialize clients
            client = ollama.Client()
            chroma_client = chromadb.Client()
           
            # Get or create collection
            collection_name = "pdfs_collection"
            try:
                collection = chroma_client.get_collection(collection_name)
            except Exception:
                collection = chroma_client.create_collection(collection_name)
           
            # Process the query
            query_chunks = split_text(user_text)
            results = []
           
            for query_text in query_chunks:
                # Get the embedding for the query chunk
                query_embedding = get_embedding(query_text, client)
               
                # Use ChromaDB's query method to find closest matches
                query_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=2,
                    include=["documents", "distances"]
                )
               
                closest_matches = query_results["documents"][0]
               
                # Combine the closest matches as context for Llama3.2
                context = " ".join(closest_matches)
                llama_response = get_llama_response(context, user_text, output_language, client)
               
                results.append({
                    'query': query_text,
                    'response': llama_response,
                    'matches': list(zip(closest_matches, query_results["distances"][0]))
                })
           
            return render(request, 'text_processor/result.html', {
                'results': results,
                'original_query': user_text
            })
    else:
        form = TextProcessorForm()
   
    return render(request, 'text_processor/process.html', {'form': form})

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('text_processor')
    else:
        form = UserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})

def logout_view(request):
    logout(request)
    messages.success(request, 'Logged out successfully!')
    return redirect('login')

