# Responsible AI RAG Assistant

## AI Governance, GDPR, and EU AI Act Knowledge Support

This project is a source-grounded **Responsible AI Retrieval-Augmented Generation (RAG) assistant** for questions related to the **EU AI Act**, **GDPR**, and **NIST AI Risk Management Framework (AI RMF)**.

The assistant retrieves relevant information from selected authoritative public sources, formats the retrieved evidence, applies responsible-use guardrails, and generates a structured answer in a reproducible fallback mode. The project also includes an optional OpenAI-assisted generation path, which is disabled by default for safe and reproducible portfolio demonstration.

---

## Project Status

**Portfolio-ready prototype**

The project includes:

- Public-source collection and source inventory
- Text extraction from official web pages and PDF sources
- Document chunking
- Sentence-transformer embedding generation
- Chroma vector-store creation
- Multi-source retrieval
- Source-aware retrieval evaluation
- Context-grounded fallback answer generation
- Optional OpenAI LLM generation pathway
- Responsible-use guardrails
- CLI testing interface
- Streamlit demo application

---

## Important Responsible-Use Notice

This project is an educational and portfolio-ready prototype.

It does **not** provide legal advice.  
It does **not** replace qualified legal, privacy, compliance, or regulatory review.  
Real compliance decisions should always be reviewed by qualified professionals.

The assistant is designed to demonstrate responsible AI system design patterns, including:

- Source-grounded retrieval
- Transparent retrieved evidence
- Explicit disclaimers
- Fallback behavior when LLM generation is disabled or unavailable
- Separation between local secrets and public repository files

---

## Key Features

### 1. Multi-source knowledge base

The project uses selected public and authoritative sources covering:

- EU AI Act overview and implementation guidance
- Regulation (EU) 2024/1689 Artificial Intelligence Act
- Regulation (EU) 2016/679 General Data Protection Regulation
- NIST AI Risk Management Framework AI RMF 1.0
- NIST AI RMF overview material

### 2. Retrieval-Augmented Generation pipeline

The project builds a structured RAG workflow:

1. Load source inventory
2. Extract and clean source text
3. Chunk documents into retrieval-ready segments
4. Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
5. Store chunks and embeddings in Chroma
6. Retrieve relevant chunks for user questions
7. Format retrieved context
8. Generate source-grounded answers
9. Display retrieved sources and confidence information

### 3. Source-aware retrieval

The retrieval pipeline includes lightweight source-awareness logic. For example:

- GDPR questions are routed toward GDPR source material
- NIST AI risk-management questions are routed toward NIST sources
- EU AI Act questions are routed toward EU AI Act sources

This improves retrieval relevance and makes the system easier to inspect.

### 4. Responsible AI guardrails

The assistant includes guardrails such as:

- No legal advice
- No unsupported certainty beyond retrieved context
- Source-aware answer generation
- Retrieved chunk references
- Responsible-use disclaimer
- Fallback mode when API generation is unavailable or disabled

### 5. Reproducible fallback mode

The repository is designed to run without paid OpenAI API usage.

By default:

```env
ENABLE_OPENAI_GENERATION=False
