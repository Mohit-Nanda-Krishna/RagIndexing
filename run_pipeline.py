import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# Make folders to save our work
os.makedirs("data", exist_ok=True)
os.makedirs("indexes", exist_ok=True)

print("Step 1: Downloading a small dataset...")
ds = load_dataset("ms_marco", "v2.1", split="train[:2000]")

print("Step 2: Cleaning it into a table...")
passages = []
for i, row in enumerate(ds):
    if row["passages"]["passage_text"]:
        passages.append({"id": i, "text": row["passages"]["passage_text"][0]})
df = pd.DataFrame(passages)
df.to_csv("data/passages.csv", index=False)
print(f"Saved {len(df)} passages to data/passages.csv")

print("Step 3: Generating embeddings (this takes a few minutes)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
np.save("data/embeddings.npy", embeddings)
print("Saved embeddings to data/embeddings.npy")

print("Step 4: Building the search index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "indexes/flat.index")
print("Saved index to indexes/flat.index")

print("Step 5: Testing with a real search query...")
query = "What causes inflation?"
q_embedding = model.encode([query])
distances, indices = index.search(q_embedding, k=5)

print("\nTop 5 results for query:", query)
for idx in indices[0]:
    print("-", df.iloc[idx]['text'][:150])

print("\nDONE. Pipeline works end to end.")