
import numpy as np

# Load the saved embeddings from the .npy file
chunk_embeddings = np.load('/Users/hymavathigummudala/chunk_embeddings.npy', allow_pickle=True)

# Inspect the shape and first few embeddings
print(f"Shape of embeddings: {chunk_embeddings.shape}")

# Loop through and inspect the first few embeddings
for i, embedding in enumerate(chunk_embeddings[:3]):  # Inspecting first 3 embeddings
    print(f"Embedding {i+1}: {embedding[:5]}...")  # Print the first 5 dimensions for readability
