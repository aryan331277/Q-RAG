# Q-RAG: Retrieval-Augmented Generation for Quantitative Finance 🧠📈

**Q-RAG** is a domain-specific Retrieval-Augmented Generation (RAG) pipeline built for processing and understanding technical documents in **quantitative finance** — including research papers, textbooks, and quant strategy docs.

This isn't a generic RAG. It’s designed for:
- Preserving LaTeX/math-heavy structure
- Using **FinBERT** embeddings
- Performing **hybrid retrieval**
- Answering **real quant questions** using a lightweight LLM backend

---

##What It Does

###Key Features:
- **Math-Aware Chunking**  
  Handles overlapping tokens while preserving LaTeX/math symbols to maintain semantic and mathematical integrity.

- **FinBERT Embeddings**  
  Finetuned on financial corpora for better representation of quant-specific terminology and modeling terms.

- **Hybrid Retrieval Stack**  
  Combines:
  - `BM25` for keyword-based retrieval
  - `FinBERT` for semantic search
  - `MiniLM` for reranking relevance

- **LLM Backend**  
  Uses **Mistral-7B** via API (Hugging Face or local) for context-aware response generation.

- **Evaluation Framework (WIP)**  
  Implements metrics like `ROUGE`, `BLEU`, `Recall@K`, and real quant QA tasks.

---

## 📦 Libraries Used

