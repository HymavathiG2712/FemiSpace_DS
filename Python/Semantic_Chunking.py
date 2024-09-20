import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Initialize the sentence-transformers model
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# Read the content of the text file
with open('/Users/hymavathigummudala/VS/nutri_output.txt', 'r') as file:
    text = file.read()

# Create a semantic text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # Define max chunk size
    chunk_overlap=20  # Overlap to ensure context continuity
)

# Split the text into chunks
chunks = text_splitter.split_text(text)

# Generate embeddings for each chunk
chunk_embeddings = [model.encode(chunk) for chunk in chunks]

# Save chunked text to a file
with open('chunked_text.txt', 'w', encoding='utf-8') as chunk_file:
    for i, chunk in enumerate(chunks):
        chunk_file.write(f"Chunk {i+1}:\n{chunk}\n\n")

# Save embeddings to a file (using numpy for saving arrays)
np.save('chunk_embeddings.npy', chunk_embeddings)

print("Chunks and embeddings have been saved to 'chunked_text.txt' and 'chunk_embeddings.npy'.")
