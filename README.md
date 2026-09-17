# Explainable, Conflict-Aware, and Feedback-Adaptive RAG for Opinion Summarization

An academic research and engineering framework for grounded opinion summarization.
Developed for B.E. Capstone Project (Group 46, Computer Engineering, Vishwakarma Institute of Technology).

---

## 🌟 Core Contributions

1. **Uncorrupted Global Sentiment Proportions ($P(S|A)$)**: 
   - Decouples population distribution calculation from retrieval. 
   - Proportions are queried directly from the full corpus in SQLite rather than estimated from a biased top-$k$ semantic search pool.
2. **Aspect Extraction & Canonicalization**:
   - Clusters noisy, fragmented user terms (*"battery drain"*, *"charging speed"*, *"backup"*) into canonical taxonomy categories (*"Battery"*).
3. **Proportional Stratified Exemplar Sampling**:
   - Samples context evidence to mirror true corpus sentiment ratios while guaranteeing a minority quota ($\ge 15\%$) so that conflicting viewpoints remain visible with verifiable receipts.
4. **Auditable Pipeline Decision Trace**:
   - Replaces ungrounded LLM chain-of-thought with a deterministic, step-by-step pipeline execution trace logging timings, aspect counts, conflict flags, and verification results.
5. **Atomic Proposition NLI Verification**:
   - Splits compound generated sentences into atomic claims and cross-examines each against cited evidence spans using DeBERTa-v3 MNLI.
6. **Quantifier Calibration Error (QCE)**:
   - Validates that verbal quantifiers (*"most"*, *"all"*, *"a minority"*) mathematically correspond to the empirical review counts.
7. **Feedback-Adaptive Memory Store**:
   - Persists user corrections (aspect reclassifications, polarity flips, upvoted citations) into a SQLite/vector exemplar store for interactive self-improvement.

---

## 🏗️ Architecture Flow

```
Raw Review Corpus ──> MinHash Deduplication ──> ABSA & Canonicalizer ──> SQLite (Exact Counts)
                                                                     └──> ChromaDB (Dense) + BM25 (Sparse)

Query ──> Canonical Aspect Parser ──> Exact Global P(S|A)
      ──> Hybrid RRF Retrieval ──> Proportional Stratifier (Minority Quota)
      ──> Decision Trace Logger
      ──> Constrained Generator (Injected Global Stats + Citations [E#])
      ──> Atomic Claim Splitter ──> DeBERTa NLI Verification Gate
      ──> Quantifier Calibration (QCE Gate)
      ──> Opinion Intelligence Dashboard
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```env
LLM_PROVIDER=gemini       # Options: 'gemini', 'groq', 'ollama', 'mock'
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

### 3. Run Interactive Dashboard
```powershell
streamlit run app/dashboard.py
```

### 4. Run REST API Server
```powershell
uvicorn app.api:app --reload --port 8000
```

### 5. Run Automated Test Suite
```powershell
pytest tests/ -v
```

---

## 📊 Evaluation & Baselines
- **Benchmark Datasets**: 
  - `SPACE`: Hotel reviews with multi-aspect gold human summaries.
  - `Amazon Electronics`: Real-world noisy product reviews.
- **Baselines Supported**:
  - Extractive: LexRank
  - Naive RAG: Standard dense top-$k$
  - Long-Context LLM: Full review concatenation
  - Proposed Framework
