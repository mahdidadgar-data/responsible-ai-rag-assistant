# Project Plan

## Project Title

Responsible AI RAG Assistant for AI Governance, GDPR, and EU AI Act Knowledge Support

## Project Objective

The objective of this project is to build a Retrieval-Augmented Generation assistant that answers questions from selected public AI governance documents.

The assistant will retrieve relevant document chunks, generate source-grounded answers, display citations or retrieved context, and include responsible AI guardrails.

## Problem Statement

AI teams often need to understand responsible AI, GDPR, and EU AI Act concepts when designing or evaluating AI systems. However, governance documents can be long, complex, and difficult to search manually.

This project explores whether a RAG-based assistant can support AI project teams by providing grounded, source-aware answers from selected governance documents.

## Intended Users

- Junior data scientists
- AI engineers
- Data analysts transitioning into AI
- Business stakeholders involved in AI projects
- Portfolio reviewers and recruiters

## Out-of-Scope

This assistant does not provide legal advice.

It should not be used as the final authority for GDPR, EU AI Act, or compliance decisions.

It should not process private, confidential, or sensitive personal data.

## Main Technical Components

1. Document ingestion
2. Text cleaning
3. Chunking
4. Embedding generation
5. Vector search
6. RAG answer generation
7. Source display
8. Responsible AI guardrails
9. Retrieval and answer evaluation
10. Streamlit app interface

## Responsible AI Design Principles

The assistant should:

- Answer only from retrieved sources where possible
- Say when the answer is not supported by the available documents
- Show retrieved context or citations
- Include a legal/compliance disclaimer
- Avoid pretending to be a lawyer or regulator
- Refuse requests for private legal conclusions
- Encourage users to consult qualified professionals for real compliance decisions