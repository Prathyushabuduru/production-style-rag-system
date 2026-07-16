# Production-Style RAG System — Academic Policy Q&A Assistant

A Retrieval-Augmented Generation (RAG) pipeline built from scratch in Python to answer natural-language questions over academic policy documents, reducing hallucinated answers compared to an LLM-only baseline.

## Problem

Academic policy documents (GPA requirements, attendance rules, academic probation criteria) are long and hard to search manually. A standalone LLM answering from general knowledge alone will hallucinate specifics — for example, stating a GPA requirement of 2.0 when the actual policy requires 2.5. This project builds a RAG system that grounds every answer in the source document instead of the model's memorized knowledge.

## How It Works

1. **Ingestion** — PDF policy documents parsed with PyMuPDF
2. **Cleaning** — removes boilerplate (e.g., "Question"/"Answer" scaffolding)
3. **Chunking** — two strategies implemented and compared: sentence-based and paragraph-based
4. **Embedding** — chunks embedded using Sentence-Transformers (`all-mpnet-base-v2`)
5. **Retrieval** — manual cosine similarity search with NumPy (no vector database) to surface top-2 relevant chunks
6. **Generation** — retrieved context passed to an open-source LLM (`TinyLlama-1.1B`) to generate a grounded answer
7. **Comparison UI** — side-by-side view of RAG output vs. LLM-only output for the same query

## Key Result

| Query | LLM-Only (Hallucinated) | RAG (Grounded) |
|---|---|---|
| GPA requirement | 2.0 (incorrect) | 2.5 (correct) |
| Attendance below 75% | Generic explanation | Academic probation next semester (correct) |
| Academic probation | Generic definition | Warning that student isn't meeting program requirements (correct) |

**Finding:** Paragraph-based chunking outperformed sentence-based chunking on context retention, producing more complete, accurate answers.

## Tech Stack

`Python` · `PyMuPDF` · `Sentence-Transformers` · `NumPy` · `TinyLlama-1.1B` · `Gradio`

## Business Impact

- **Accuracy** — responses grounded in real source documents, not model memory
- **Efficiency** — faster retrieval of policy information than manual search
- **Reliability** — reduced hallucination improves trust in automated Q&A
- **Scalability** — same architecture applies to HR, legal, and support knowledge bases

## Author

Prathyusha Buduru — M.S. Analytics, Saint Louis University
[LinkedIn](https://www.linkedin.com/in/prathyushabuduru/) · [GitHub](https://github.com/Prathyushabuduru)

## References

- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*. https://arxiv.org/abs/2005.11401
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks. *EMNLP*. https://arxiv.org/abs/1908.10084
- Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. https://arxiv.org/abs/2302.13971
