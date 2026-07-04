# Source Register

This file records the public sources planned for the Responsible AI RAG Assistant project.

The assistant is designed as a knowledge-support prototype, not as a legal advice system. Regulatory and compliance content should always be checked against current official sources before real-world use.

## Source Selection Principles

The project uses public and authoritative sources only.

The project should not use:

* private company documents
* confidential documents
* personal data
* API keys or credentials
* copyrighted books or paywalled documents
* unverified blog content as primary source material

Raw source files should remain local and should not be uploaded to GitHub. Processed text extracts, summaries, metadata files, and evaluation examples may be uploaded only when legally and ethically appropriate.

## Planned Sources

| Source ID | Source Name                                                  | Publisher                | Source Type              | Main Topic                                                                       | Planned Use                                            | Access Date | Upload Raw Copy to GitHub? |
| --------- | ------------------------------------------------------------ | ------------------------ | ------------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------ | ----------- | -------------------------- |
| SRC-001   | AI Act                                                       | European Commission      | Public official web page | EU AI Act overview, risk-based approach, trustworthy AI                          | Governance overview and user-facing source explanation | 2026-06-30  | No                         |
| SRC-002   | Regulation (EU) 2024/1689 Artificial Intelligence Act        | EUR-Lex / European Union | Official legal text      | AI Act legal provisions, definitions, obligations, risk categories               | Grounding source for AI Act questions                  | 2026-06-30  | No                         |
| SRC-003   | Regulation (EU) 2016/679 General Data Protection Regulation  | EUR-Lex / European Union | Official legal text      | GDPR principles, lawful basis, data protection obligations                       | Grounding source for GDPR-related questions            | 2026-06-30  | No                         |
| SRC-004   | Artificial Intelligence Risk Management Framework AI RMF 1.0 | NIST                     | Public framework / PDF   | AI risk management, trustworthy AI, governance, measurement, mapping, management | Responsible AI and risk-management grounding           | 2026-06-30  | No                         |
| SRC-005   | AI Risk Management Framework overview page                   | NIST                     | Public official web page | AI risk management framework overview                                            | Supporting source metadata and framework explanation   | 2026-06-30  | No                         |

## Source URLs

| Source ID | URL                                                                       |
| --------- | ------------------------------------------------------------------------- |
| SRC-001   | https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai |
| SRC-002   | https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX%3A32024R1689    |
| SRC-003   | https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng                         |
| SRC-004   | https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf                    |
| SRC-005   | https://www.nist.gov/itl/ai-risk-management-framework                     |

## Initial Scope for Version 1

Version 1 of this project will focus on a limited set of governance topics:

1. Responsible AI risk management
2. EU AI Act risk-based approach
3. High-risk AI system concepts
4. Transparency and human oversight
5. GDPR principles relevant to AI systems
6. Lawfulness, fairness, transparency, and data minimisation
7. Documentation and accountability
8. Limitations of RAG systems for legal/regulatory support

## Important Responsible-Use Note

The final assistant should clearly state that it does not provide legal advice.

The assistant should help users explore relevant concepts from public documents, but real compliance decisions should be reviewed by qualified legal, privacy, or compliance professionals.
