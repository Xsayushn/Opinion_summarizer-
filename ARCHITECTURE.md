# System Architecture: Explainable, Conflict-Aware, and Feedback-Adaptive Opinion Summarizer

This document provides the complete technical architecture, mathematical formulations, database schemas, and dataflow specifications for the **Explainable Opinion Summarization Framework** (Group 46, Computer Engineering, Vishwakarma Institute of Technology).

---

## 1. High-Level Architectural Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    OFFLINE INGESTION STAGE                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                  [ RAW REVIEWS CORPUS ]
                                (SPACE / Amazon Electronics)
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ MinHash LSH Deduplication │
                               │   (Jaccard threshold=0.85)│
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Sentence / Proposition    │
                               │ Segmentation & Cleaning   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Aspect Canonicalization   │
                               │ Synonyms ──> Standard Cat │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Polarity Classification   │
                               │ DeBERTa-v3 / Rule Engine  │
                               └─────────────┬─────────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
       ┌───────────────────────────┐                   ┌───────────────────────────┐
       │     SQLite Database       │                   │    Dual Retrieval Index   │
       │ Exact Global Counts P(S|A)│                   │ • ChromaDB (Dense Vector) │
       │ & Opinion Unit Records    │                   │ • BM25 (Sparse Keyword)   │
       └───────────────────────────┘                   └───────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   ONLINE QUERY PIPELINE                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                       [ USER QUERY ]
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Step 1: Query Parser &    │
                               │ Canonical Aspect Extractor│
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Step 2: Global Counts     │
                               │ Exact P(S|A) from SQLite  │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Step 3: Conflict Detector │
                               │ Polarity Bifurcation Flag │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Step 4: Hybrid RRF Search │
                               │ Dense BGE + BM25 Fusion   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Step 5: Proportional      │
                               │ Stratified Exemplar Sample│
                               │ (15% Minority Guarantee)  │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Step 6: Pipeline Decision │
                               │ Trace Logger (Timings/Log)│
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Step 7: Constrained Gen   │
                               │ Dialectical Prompt + [E#] │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Step 8: Atomic Claim      │
                               │ Decomposition             │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Step 9: NLI & Quantifier  │
                               │ Verification Gate (QCE)   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Step 10: Opinion Dashboard│
                               │ Receipts & Decision Trace │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Step 11: Feedback Memory  │
                               │ Exemplar Cache & Boost    │
                               └───────────────────────────┘
```

---

## 2. Mathematical Formulations & Algorithms

### A. Unbiased Global Distribution ($P(S|A)$)
Standard RAG estimates sentiment proportions from the top-$k$ retrieved candidates ($K \approx 10-20$), which introduces severe retrieval bias because emotional rants and complaints cluster densely in semantic embedding space.

We compute exact population parameters directly across all indexed units in SQLite:
$$P(S = s \mid A = a) = \frac{\sum_{u \in \mathcal{U}} \mathbb{I}(u_A = a \land u_S = s)}{\sum_{u \in \mathcal{U}} \mathbb{I}(u_A = a)}$$

Where:
- $\mathcal{U}$ is the global corpus of opinion units.
- $a$ is the canonical aspect (e.g., Cleanliness, Service).
- $s \in \{+1 \text{ (Positive)}, 0 \text{ (Neutral)}, -1 \text{ (Negative)}\}$.

### B. Conflict & Polarity Bifurcation Detection
An aspect $a$ is flagged with a **Polarity Conflict Alert** if both positive and negative empirical proportions exceed a bifurcation threshold $\tau_{\text{conflict}} = 20\%$:
$$\text{Conflict}(a) = \mathbb{I}\Big(P(S = +1 \mid A = a) \ge \tau_{\text{conflict}} \;\land\; P(S = -1 \mid A = a) \ge \tau_{\text{conflict}}\Big)$$

### C. Hybrid Reciprocal Rank Fusion (RRF)
To balance semantic conceptual relevance (Dense BGE-base) with exact keyword/aspect matches (BM25):
$$\text{RRF}(d) = \frac{1}{k_{\text{rrf}} + \text{rank}_{\text{dense}}(d)} + \frac{1}{k_{\text{rrf}} + \text{rank}_{\text{bm25}}(d)}$$
Where default smoothing constant $k_{\text{rrf}} = 60$.

### D. Proportional Stratified Exemplar Sampling
To ensure evidence exemplars mirror true sentiment ratios while preventing the erasure of minority viewpoints:
1. Context budget per aspect: $K_a = \lfloor K_{\text{context}} / |\mathcal{A}_{\text{query}}| \rfloor$.
2. Target quota for polarity $s$:
   $$Q(a, s) = \max\Big(\lfloor K_a \cdot \tau_{\text{minority}} \rfloor \cdot \mathbb{I}\big(P(s|a) \ge \tau_{\text{minority}}\big), \; \text{round}(K_a \cdot P(s|a))\Big)$$
   Where default minority quota guarantee $\tau_{\text{minority}} = 0.15$ (15%).

### E. Quantifier Calibration Error (QCE)
Let $C = \{c_1, c_2, \dots, c_m\}$ be the set of atomic claims extracted from the generated summary. Each claim with a verbal quantifier $Q_{\text{verbal}}(c_i)$ (e.g., *"most"*, *"all"*, *"a minority"*) is mapped to an expected numeric interval $[\alpha_{\min}, \alpha_{\max}]$ with midpoint $\mu(c_i)$.

$$\text{QCE} = \frac{1}{|C|} \sum_{i=1}^{|C|} \left| \mu(c_i) - P_{\text{empirical}}(a_i, s_i) \right|$$

---

## 3. Database & Storage Schemas

### SQLite Schema (`data/opinions.db`)

#### Table: `reviews`
| Column | Type | Description |
|---|---|---|
| `review_id` | `TEXT PRIMARY KEY` | Unique review identifier |
| `entity_id` | `TEXT` | Entity or product identifier |
| `text` | `TEXT` | Full raw review body |
| `rating` | `REAL` | Original star rating (1.0 - 5.0) |
| `date` | `TEXT` | Timestamp / ISO date string |
| `metadata_json` | `TEXT` | Serialized JSON metadata |

#### Table: `opinion_units`
| Column | Type | Description |
|---|---|---|
| `unit_id` | `TEXT PRIMARY KEY` | Unique opinion unit ID (`U_{review_id}_{idx}`) |
| `review_id` | `TEXT` | Foreign key referencing `reviews(review_id)` |
| `entity_id` | `TEXT` | Entity identifier |
| `canonical_aspect` | `TEXT` | Standardized aspect tag (e.g., `Cleanliness`) |
| `raw_aspect` | `TEXT` | Raw extracted aspect phrase |
| `sentiment` | `INTEGER` | Numeric polarity (`+1`, `0`, `-1`) |
| `sentiment_label` | `TEXT` | Human-readable label (`positive`, `neutral`, `negative`) |
| `confidence` | `REAL` | Polarity classification confidence score |
| `text_span` | `TEXT` | Exact evidence sentence or proposition span |

*Indices: `idx_units_entity_aspect ON (entity_id, canonical_aspect)`, `idx_units_sentiment ON (sentiment)`.*

### ChromaDB Schema (`data/chroma_db/`)
- **Collection Name**: `opinion_units`
- **Embedding Function**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Distance Metric**: Cosine Distance ($d_{\cos} = 1 - \frac{u \cdot v}{\|u\|\|v\|}$)
- **Payload Metadata**: `unit_id`, `review_id`, `entity_id`, `canonical_aspect`, `sentiment`, `sentiment_label`, `confidence`.

### Feedback Memory Schema (`data/feedback/feedback.db`)

#### Table: `user_feedback`
| Column | Type | Description |
|---|---|---|
| `feedback_id` | `TEXT PRIMARY KEY` | Unique feedback submission ID |
| `unit_id` | `TEXT` | Associated opinion unit ID |
| `original_aspect` | `TEXT` | Original aspect assigned by model |
| `corrected_aspect` | `TEXT` | Human-corrected aspect category |
| `original_sentiment`| `INTEGER` | Original polarity |
| `corrected_sentiment`| `INTEGER`| Human-corrected polarity |
| `upvoted` | `INTEGER` | `1` for upvoted evidence, `0` otherwise |
| `comment` | `TEXT` | User notes / explanation |
| `timestamp` | `REAL` | Epoch timestamp |

#### Table: `exemplar_memory`
Stores verified few-shot exemplars used for dynamic in-context prompting.

---

## 4. Verification & Explainability Layer

### A. Atomic Claim Decomposition
To prevent false-negative NLI scores on multi-clause sentences, compound generated sentences are broken into atomic propositions:
- Split on contrastive conjunctions: `while`, `although`, `however`, `whereas`, `but`.
- Split on clause commas separating citations: `(?<=\])\s*,\s*`.
- Each atomic proposition inherits its respective citation anchor `[E#]`.

### B. Natural Language Inference (NLI) Cross-Examination
Premise $P$: Cited evidence text span.
Hypothesis $H$: Atomic claim generated by the LLM.
- Model: `cross-encoder/nli-deberta-v3-small`.
- Decision Rule:
  - If $\text{score}(\text{Contradiction}) > 0.30 \implies \text{CONTRADICTED}$.
  - If $\text{score}(\text{Entailment}) \ge 0.70 \implies \text{VERIFIED}$.
  - Otherwise $\implies \text{UNSUPPORTED}$.

### C. Pipeline Decision Trace
An execution logger capturing:
1. Query intent & aspect resolution
2. Global statistics calculation
3. Conflict detection flags
4. Proportional stratified sampling audit
5. Dialectical generation parameters
6. Claim decomposition and NLI entailment scores
7. Quantifier calibration checks and mean QCE
