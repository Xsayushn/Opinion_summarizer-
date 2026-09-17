# 🧠 Explainable, Conflict-Aware, and Feedback-Adaptive RAG for Opinion Summarization

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![Vector Store](https://img.shields.io/badge/Vector%20DB-ChromaDB%20%7C%20BM25-green.svg)](https://www.trychroma.com/)
[![Tests](https://img.shields.io/badge/Tests-100%25%20Passed-brightgreen.svg)](tests/)

An academic research and engineering framework for grounded, aspect-aware, and conflict-preserving opinion summarization.
Developed for **B.E. Final Year Capstone Project (Group 46, Computer Engineering, Vishwakarma Institute of Technology)**.

---

## 📑 Core Documentation Sitemap

| Document | Purpose |
|---|---|
| [**`ARCHITECTURE.md`**](ARCHITECTURE.md) | Complete system architecture diagrams, mathematical equations ($P(S\|A)$, RRF, QCE), database schemas, and data pipelines. |
| [**`PROMPTS.md`**](PROMPTS.md) | All prompt templates, dialectical prompt construction, few-shot in-context exemplars, and anti-hallucination guardrails. |
| [**`Implementation_plan.md`**](Implementation_plan.md) | The approved v2.0 research roadmap, review milestones, and academic review evaluations (7/10 $\rightarrow$ 9.3/10). |

---

## 🌟 The Core Problem We Solve

When a standard LLM or naive RAG pipeline summarizes thousands of customer reviews, it encounters three catastrophic failure modes:

1. **Volume & Sentiment Distortion:** Naive semantic search retrieves the 10 most emotionally polarized reviews (often vocal complaints), causing the summary to claim *"users widely report dissatisfaction"* when 80% of customers were actually pleased.
2. **Conflict Eradication:** Standard summarizers average out conflicting viewpoints into a generic consensus, concealing polarized feedback (e.g., great daytime photography vs. terrible low-light performance).
3. **Quantifier Hallucination:** LLMs fabricate quantifiers like *"most reviewers"* or *"everyone agreed"* without checking empirical proportions.

---

## 🔬 Our Research Contributions

1. **Uncorrupted Global Corpus Statistics ($P(S|A)$):**
   - Decoupled population distribution calculation from retrieval. Proportions are computed directly across all indexed units in SQLite, guaranteeing true mathematical ground truth immune to retrieval bias.
2. **Aspect Canonicalization Layer:**
   - Standardizes fragmented customer vocabularies (*"battery drain"*, *"charging speed"*, *"screen-on-time"*) into canonical taxonomy categories (*"Battery"*).
3. **Proportional Stratified Exemplar Sampling:**
   - Samples context evidence to mirror true corpus sentiment ratios while guaranteeing a **$\ge 15\%$ minority quota** so that opposing viewpoints are never suppressed.
4. **Auditable Pipeline Decision Trace:**
   - Replaces ungrounded LLM chain-of-thought with a deterministic, millisecond-accurate pipeline execution log.
5. **Atomic Proposition NLI Verification:**
   - Decomposes compound generated sentences into atomic claims and cross-examines each against cited evidence spans `[E#]` using DeBERTa-v3 MNLI.
6. **Quantifier Calibration Error (QCE):**
   - Introduces a formal calibration metric:
     $$\text{QCE} = \frac{1}{|C|} \sum_{i=1}^{|C|} \left| Q_{\text{verbal}}(c_i) - P_{\text{empirical}}(a_i, s_i) \right|$$
7. **Feedback-Adaptive Memory Store:**
   - Persists user corrections (aspect reclassifications, polarity flips, upvotes) into a SQLite exemplar store for active learning without expensive model retraining.

---

## 🏗️ System Architecture Overview

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
*(For complete mathematical details, see [ARCHITECTURE.md](ARCHITECTURE.md))*

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/Xsayushn/Opinion_summarizer-.git
cd Opinion_summarizer-

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)
Create a `.env` file in the root directory:
```env
LLM_PROVIDER=gemini       # Options: 'gemini', 'groq', 'ollama', 'mock'
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```
*(Note: If no API key is set, the system automatically uses a deterministic mock generator for offline testing and viva presentations).*

### 3. Launch Interactive Dashboard
```powershell
.\.venv\Scripts\streamlit run app/dashboard.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Launch REST API Server
```powershell
.\.venv\Scripts\uvicorn app.api:app --reload --port 8000
```
API Documentation (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Run Automated Test Suite
```powershell
.\.venv\Scripts\pytest tests/ -v
```

---

## 🎓 Review & Viva Presentation Guide (VIT Pune)

When presenting to your project guide and evaluation panel:

1. **Step 1: Ingest Corpus**
   - Click **"📥 Load Sample Reviews"** in the sidebar. Point out that duplicate reviews were removed via MinHash LSH and opinion units were extracted into SQLite.
2. **Step 2: Execute Analysis**
   - Query: `"What do customers think about cleanliness, service, and location?"`
   - Click **"🚀 Analyze Opinions"**.
3. **Step 3: Show the Decision Trace**
   - Expand the **"⚡ Pipeline Decision Trace"** drawer. Walk the panel through the milestones: intent parsing $\rightarrow$ global distribution $\rightarrow$ conflict detection $\rightarrow$ proportional sampling $\rightarrow$ NLI verification.
4. **Step 4: Highlight Conflict Preservation**
   - Show the **Amber Conflict Box** for *Service*: explain that front desk staff was praised, but room service was criticized. Show how naive RAG would have hidden this, while your system separated it into Consensus vs. Contentions.
5. **Step 5: Demonstrate Feedback Learning**
   - In the **Evidence Receipts Explorer**, click **"Flag Mislabeled Sentiment"** on an evidence card. Show how the correction is stored in SQLite and injected as a few-shot exemplar for future queries.
