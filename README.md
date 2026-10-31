# 🚗 Insurance Claim Intelligence System

A multimodal AI-based insurance claim assessment system that combines **Computer Vision, NLP, Multimodal Fusion, Retrieval-Augmented Generation (RAG), and Large Language Models (LLMs)** to assist with vehicle insurance claim analysis.

The system analyzes both a **damaged vehicle image** and a **claim description**, compares predictions from independent AI models, retrieves relevant insurance policy clauses, and generates an evidence-grounded claim assessment report.

---

## 📌 Project Overview

Traditional insurance claim assessment often requires manual inspection of:

- Vehicle damage images
- Customer claim descriptions
- Insurance policy documents
- Coverage conditions and exclusions

This project aims to assist this process by bringing these sources of information into a single AI-assisted pipeline.

The system accepts:

1. **Vehicle damage image**
2. **Customer claim description**

It then:

- Detects vehicle damage from the image using **EfficientNetB3**
- Classifies damage described in the claim text using **DistilBERT**
- Combines predictions from both modalities using a **multimodal fusion pipeline**
- Measures agreement and disagreement between the models
- Retrieves relevant clauses from insurance policy documents using **RAG**
- Uses **Gemini 2.5 Flash** to generate a structured, evidence-grounded claim report
- Displays the complete analysis through a **Streamlit** interface

---

# 🏗️ System Architecture

```text
                    USER INPUT
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
       Vehicle Image       Claim Description
              │                 │
              ▼                 ▼
       EfficientNetB3        DistilBERT
              │                 │
              ▼                 ▼
       Image Damage       Text Damage
       Probabilities      Probabilities
              │                 │
              └────────┬────────┘
                       ▼
                Multimodal Fusion
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
      Agreement     Confidence    Conflict
       Score           Gap          Level
          │            │            │
          └────────────┼────────────┘
                       ▼
                Claim Assessment
                       │
                       ▼
              Policy Query / Context
                       │
                       ▼
                RAG Retrieval
                       │
                ┌──────┴──────┐
                │             │
                ▼             ▼
            Embeddings    ChromaDB
                │             │
                └──────┬──────┘
                       ▼
                  MMR Retrieval
                       │
                       ▼
              Relevant Policy Clauses
                       │
                       ▼
             Gemini 2.5 Flash
                       │
                       ▼
             Structured Claim Report
                       │
                       ▼
                  Streamlit UI