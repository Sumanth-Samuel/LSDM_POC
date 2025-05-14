import openai
import faiss
import numpy as np
import pickle
import re
import os
import time
from pathlib import Path
from tqdm import tqdm
import os
from dotenv import load_dotenv
load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
SCHEMA_PATH = r"C:/Users/SumanthJSamuel/POC to Host/schema.sql"
OUTPUT_DIR = r"C:/Users/SumanthJSamuel/POC to Host/index"
CACHE_FILE = os.path.join(OUTPUT_DIR, "embeddings_cache.pkl")
INDEX_FILE = os.path.join(OUTPUT_DIR, "schema_index.faiss")
CHUNKS_FILE = os.path.join(OUTPUT_DIR, "schema_chunks.pkl")
EMBEDDING_MODEL = "text-embedding-3-small"


def extract_tables(schema: str):
    """Parse CREATE TABLE blocks into individual table chunks."""
    tables = re.findall(r"CREATE TABLE.*?\);", schema, re.DOTALL | re.IGNORECASE)
    parsed_chunks = []
    for table in tables:
        name_match = re.search(r"CREATE TABLE (\w+)", table, re.IGNORECASE)
        table_name = name_match.group(1) if name_match else "UnknownTable"
        fields = "\n".join(line.strip() for line in table.split("\n")[1:-1])
        parsed_chunks.append(f"TABLE: {table_name}\n{fields}")
    return parsed_chunks

def get_embedding(text: str, max_retries=3, base_wait=2) -> list:
    """Get embedding with retry and backoff."""
    for attempt in range(max_retries):
        try:
            response = openai.Embedding.create(input=[text], model=EMBEDDING_MODEL)
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(base_wait * (2 ** attempt))
            else:
                raise RuntimeError(f"Failed to embed after {max_retries} attempts.")


# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load schema
with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema_text = f.read()

schema_chunks = extract_tables(schema_text)
print(f"Extracted {len(schema_chunks)} schema chunks.")

# Load or init cache
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
else:
    cache = {}

# Embed and cache
print("Generating embeddings with caching and retries...")
embeddings = []
for chunk in tqdm(schema_chunks, desc="Embedding chunks"):
    if chunk in cache:
        embeddings.append(cache[chunk])
    else:
        emb = get_embedding(chunk)
        cache[chunk] = emb
        embeddings.append(emb)

# Save updated cache
with open(CACHE_FILE, "wb") as f:
    pickle.dump(cache, f)

# Store in FAISS
embedding_dim = len(embeddings[0])
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings).astype("float32"))

faiss.write_index(index, INDEX_FILE)
with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(schema_chunks, f)

print(f"FAISS index saved to: {INDEX_FILE}")
print(f"Schema chunks saved to: {CHUNKS_FILE}")
print(f"Embedding cache saved to: {CACHE_FILE}")
