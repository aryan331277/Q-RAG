from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import json
import numpy as np

model = SentenceTransformer('yiyanghkust/finbert-pretrain')
chunks_dir = '/content/drive/MyDrive/chunks'
embeddings_dir = '/content/drive/MyDrive/embeddings'

texts = []
metadata = []

for file in os.listdir(chunks_dir):#similar logic as extracting text just that here we are using chunk_id instead of page number
    if file.endswith("_chunks.json"):
        with open(os.path.join(chunks_dir, file), "r", encoding="utf-8") as f:
            chunks = json.load(f)
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                metadata.append({
                    "filename": file.replace("_chunks.json", ""),
                    "chunk_id": i,
                    "text": chunk
                })
print(f"{len(texts)}")

embeddings = model.encode(texts,convert_to_numpy=True)#generating embeddings

np.save(os.path.join(embeddings_dir,"quant_embeddings.npy"),embeddings)#saving




