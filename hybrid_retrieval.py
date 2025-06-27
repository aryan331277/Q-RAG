from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json


embeddings = np.load("/content/drive/MyDrive/embeddings/quant_embeddings.npy")#loading embeddings
with open("/content/drive/MyDrive/embeddings/quant_metadata.json", "r", encoding="utf-8") as f:#opening the fike
    texts = [item["text"] for item in json.load(f)]#converting the text field into a pair of strings

embedder = SentenceTransformer('yiyanghkust/finbert-pretrain')#used in embedding
semantic = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')#semantic reranking
bm25 = BM25Okapi([nltk.word_tokenize(t.lower()) for t in texts])#fast token based finder to find relevant text because of overlapping

def retrieve(query,top_k=2):#top2,here top2 is picked as doc size is 6
    # BM25 scores
    bm25_scores = bm25.get_scores(nltk.word_tokenize(query.lower()))#lowers the words and tokenises them then provides a score based on bm25
    bm25_top = np.argsort(bm25_scores)[::-1][:top_k]#takes top5 of descending order

    q_vec = embedder.encode(query)#capture semantic meaning of word
    dense_scores = np.dot(embeddings, q_vec)/(np.linalg.norm(embeddings, axis=1)*np.linalg.norm(q_vec) + 1e-8)#cosine similarity with query vector and document vector.
  #here embeddings is matrix of shape (num_docs,vector_dim),q_vec is of vector_dim.1e-8 to avoid division with 0
    dense_top = np.argsort(dense_scores)[::-1][:top_k]#top2 of document

    candidates=list({*bm25_top, *dense_top})#merging top and unique results
    candidate_texts=[texts[i] for i in candidates]#pulls out the text from the document

    # Rerank
    q_rerank = semantic.encode(query, convert_to_tensor=True)#encodes
    c_rerank = semantic.encode(candidate_texts, convert_to_tensor=True)#encodes
    scores = util.cos_sim(q_rerank, c_rerank)[0]#cosine similarity
    top_results = [text for text, _ in sorted(zip(candidate_texts, scores), key=lambda x: x[1], reverse=True)[:top_k]]#sort and get topk results

    return top_results
