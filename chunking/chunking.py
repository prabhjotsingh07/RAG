import pandas as pd
import chromadb
import requests

class OllamaEmbedder:
    def __init__(self, model="all-minilm"):
        self.model = model
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        return [
            requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": self.model, "prompt": text}
            ).json()["embedding"]
            for text in input
        ]

# Read CSV file
df = pd.read_csv('./ollamademo/Articles.csv', encoding='ISO-8859-1')
print("Found columns:", df.columns.tolist())

# Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection(
    name="paragraphs_collection",
    embedding_function=OllamaEmbedder()
)

# Process each paragraph and store with embeddings
all_chunks = []
chunk_ids = []

for index, row in df.iterrows():
    # Get text from first column
    text = str(row[df.columns[0]])
    
    # Break into smaller chunks (500 characters each)
    current_position = 0
    while current_position < len(text):
        # Find the end of current chunk
        chunk_end = min(current_position + 500, len(text))
        
        # Try to find a period or newline for natural break
        if chunk_end < len(text):
            next_period = text.find('.', chunk_end - 50, chunk_end + 50)
            next_newline = text.find('\n', chunk_end - 50, chunk_end + 50)
            
            if next_period != -1:
                chunk_end = next_period + 1
            elif next_newline != -1:
                chunk_end = next_newline
        
        # Get the chunk
        chunk = text[current_position:chunk_end].strip()
        if chunk:  # Only add non-empty chunks
            all_chunks.append(chunk)
            chunk_ids.append(f"chunk_{index}_{len(chunk_ids)}")
        
        current_position = chunk_end

# Store all chunks and get embeddings in one go
if all_chunks:
    collection.add(
        documents=all_chunks,
        ids=chunk_ids
    )

print(f"Processed {len(df)} documents into {len(all_chunks)} chunks")

# Get user query and find similar chunks
query = input("\nEnter your search text: ")

# Query the collection for similar chunks
results = collection.query(
    query_texts=[query],
    n_results=2
)

# Print the results
print("\nTop 2 most similar chunks:")
for i, chunk in enumerate(results['documents'][0], 1):
    print(f"\n{i}. {chunk}")
    print("-" * 50)