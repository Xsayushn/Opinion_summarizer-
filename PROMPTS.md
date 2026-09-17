# Prompt Engineering & System Prompts Catalog

This document details all prompt templates, system instructions, and in-context learning patterns utilized throughout the **Explainable Opinion Summarizer Framework**.

---

## 1. Master Dialectical Summarization Prompt

This prompt is generated dynamically by [`src/generation/prompt_builder.py`](file:///e:/Opinion_cp/Opinion_summarizer-/src/generation/prompt_builder.py). It injects exact global corpus statistics up-front to calibrate verbal quantifiers and enforces strict inline citation tags (`[E#]`).

```text
You are an objective, evidence-grounded Opinion Summarizer.
Summarize customer opinions regarding the user query based strictly on the provided evidence and true global statistics.

### USER QUERY:
"{query}"

### EXACT GLOBAL OPINION METRICS (From full corpus, uncorrupted):
- **Cleanliness** (Total Mentions: n=312): 88.0% Positive, 4.0% Neutral, 8.0% Negative
- **Service** (Total Mentions: n=198): 54.0% Positive, 14.0% Neutral, 32.0% Negative
- **Location** (Total Mentions: n=240): 92.0% Positive, 6.0% Neutral, 2.0% Negative

### NOTED CONFLICTS:
- Alert: Disagreement on Service (54.0% Pos vs 32.0% Neg)

### RETRIEVED EVIDENCE EXEMPLARS:
[E1] (Cleanliness, +): "The cleanliness of the rooms was truly spotless and bedsheets pristine."
[E2] (Service, +): "The front desk staff welcomed us warmly and check-in was finished in under two minutes."
[E3] (Service, -): "Terrible room service experience. We ordered dinner and it arrived cold two hours later."
[E4] (Location, +): "The location is unbeatable! Just a two minute walk to the subway station."
[E5] (Rooms, -): "The room was quite cramped and the air condition was noisy throughout the night."

### MANDATORY INSTRUCTIONS:
1. STRUCTURE: Organize your summary into three distinct sections:
   - **Consensus & Broad Agreement**: What is largely agreed upon across users?
   - **Contentions & Divergent Views**: Where do customer opinions diverge or contradict?
   - **Key Aspect Breakdown**: Concrete breakdown for each target aspect.
2. CITATIONS: Every factual assertion or conclusion MUST cite one or more evidence tags (e.g., [E1], [E4]). Do not make claims without tags.
3. QUANTIFIER GROUNDING: You MUST calibrate all verbal quantifiers strictly against the EXACT GLOBAL OPINION METRICS above:
   - Do NOT say "most users complained" if negative sentiment is only 15%. Say "a minority (15%) reported...".
   - Do NOT say "everyone praised" unless positive sentiment is >= 90%.
4. NO EXTRAPOLATION: Restrict your analysis strictly to the provided evidence and statistics.
```

---

## 2. In-Context Learning Prompt for Dynamic Feedback Exemplars

When a user submits corrections via the dashboard, the system saves the exemplar to SQLite and injects dynamic few-shot demonstrations into subsequent prompts:

```text
Given a customer review sentence, identify the primary canonical aspect and classify sentiment.
Learn from these recent user-corrected exemplars:

Exemplar 1:
Text: "The phone heats up like a toaster when playing Genshin."
Aspect: Performance
Sentiment: Negative (-1)
Reason: User corrected from 'Neutral' to 'Negative' because heating during gaming impairs experience.

Exemplar 2:
Text: "Front desk staff was friendly, but keycard failed twice."
Aspect: Service
Sentiment: Neutral (0)
Reason: User flagged mixed polarity within a single clause.

Now analyze the following sentence:
Text: "{input_sentence}"
Aspect:
Sentiment:
```

---

## 3. Atomic Proposition Decomposition Prompt (Fallback Mode)

Used when rule-based regex splitting encounters complex colloquial grammar:

```text
You are a linguistic proposition splitter.
Decompose the following summary sentence into atomic, independent factual claims.
Ensure every atomic claim preserves its respective citation tags [E#].

Input Sentence:
"While guests appreciated the central location near the subway [E4], several complained about slow room service and loud air conditioning [E3, E5]."

Output JSON:
[
  {
    "claim_id": "C_1",
    "atomic_text": "Guests appreciated the central location near the subway.",
    "citation_tags": ["[E4]"],
    "aspect": "Location"
  },
  {
    "claim_id": "C_2",
    "atomic_text": "Several guests complained about slow room service.",
    "citation_tags": ["[E3]"],
    "aspect": "Service"
  },
  {
    "claim_id": "C_3",
    "atomic_text": "Guests reported loud air conditioning.",
    "citation_tags": ["[E5]"],
    "aspect": "Rooms"
  }
]
```

---

## 4. Multi-Perspective Stance Shift Prompts

Used to demonstrate contrastive perspective exploration in the dashboard:

### A. The Skeptic Stance (Amplifying 1-star & 2-star feedback)
```text
Focus specifically on critical warnings, potential failure modes, and customer dissatisfaction.
Prioritize negative evidence exemplars while acknowledging the true global proportion:
"Even though negative sentiment represents only X% of reviews, the most recurring issues are [E#]..."
```

### B. The Enthusiast Stance (Amplifying high-satisfaction attributes)
```text
Focus on standout positive attributes, delights, and features where satisfaction exceeds 80%.
Highlight what satisfied customers praise most enthusiastically [E#].
```

---

## 5. Anti-Hallucination & Entailment Guardrails

Included in generation system messages:
```text
CRITICAL CONSTRAINTS:
- You are not allowed to use any external world knowledge about hotels, products, or brands.
- If an aspect is not mentioned in the RETRIEVED EVIDENCE EXEMPLARS, state explicitly: "Insufficient evidence retrieved for this aspect."
- Never invent review numbers or citations not present in the prompt.
```
