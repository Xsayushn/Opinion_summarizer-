"""
Interactive Streamlit Dashboard for Opinion Summarizer.
Features the Auditable Decision Trace, Global Distribution Charts,
Dialectical Summary, Evidence Receipts Explorer, and Feedback Loop.
"""

import streamlit as st
import json
from pathlib import Path
import os
import sys

# Ensure src is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pipeline import OpinionSummarizerPipeline
from src.ingestion.loader import ReviewLoader, ReviewItem
from src.feedback.adaptive_store import UserFeedbackRecord

st.set_page_config(
    page_title="Opinion Intelligence | Explainable RAG",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for modern research dashboard aesthetics
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-bottom: 8px; }
    .trace-card { background-color: #F1F5F9; border-left: 4px solid #3B82F6; padding: 10px; margin-bottom: 6px; border-radius: 4px; font-family: monospace; font-size: 0.85rem; }
    .conflict-box { background-color: #FEF3C7; border: 1px solid #F59E0B; border-radius: 6px; padding: 12px; color: #92400E; margin-bottom: 12px; }
    .evidence-tag { display: inline-block; background-color: #DBEAFE; color: #1E40AF; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; margin-right: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    return OpinionSummarizerPipeline()


pipeline = get_pipeline()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🛠 Settings & Data")
domain = st.sidebar.selectbox("Domain Taxonomy", ["hotel", "electronics"], index=0)

sample_file = Path(__file__).resolve().parent.parent / "data" / "sample_reviews.json"
if st.sidebar.button("📥 Load Sample Reviews"):
    if sample_file.exists():
        loader = ReviewLoader()
        raw_reviews = loader.load_from_json(sample_file)
        stats = pipeline.index_reviews(raw_reviews)
        st.sidebar.success(f"Indexed {stats['total_reviews_indexed']} reviews ({stats['total_opinion_units_extracted']} opinion units)!")
    else:
        st.sidebar.error("Sample reviews file not found.")

st.sidebar.markdown("---")
st.sidebar.subheader("LLM Provider")
llm_provider = st.sidebar.selectbox("Active Provider", ["gemini", "groq", "ollama", "mock"], index=0)
pipeline.llm.provider = llm_provider

custom_key = st.sidebar.text_input("API Key (Gemini/Groq)", type="password")
if custom_key:
    if llm_provider == "gemini":
        os.environ["GEMINI_API_KEY"] = custom_key
    elif llm_provider == "groq":
        os.environ["GROQ_API_KEY"] = custom_key

# --- MAIN DASHBOARD HEADER ---
st.markdown('<div class="main-header">🧠 Explainable Opinion Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Conflict-Aware, Distributionally Grounded, and Feedback-Adaptive RAG for Opinion Summarization</div>', unsafe_allow_html=True)

query = st.text_input(
    "Enter your analysis query:",
    value="What do customers think about cleanliness, service, and location?",
    help="Type your question or target aspects to summarize opinions."
)

col_run, col_budget = st.columns([4, 1])
with col_budget:
    context_budget = st.slider("Evidence Quota", min_value=12, max_value=60, value=36, step=6)

if col_run.button("🚀 Analyze Opinions", type="primary", use_container_width=True):
    with st.spinner("Analyzing opinion corpus and deliberating..."):
        result = pipeline.summarize(query=query, context_budget=context_budget)
        st.session_state["result"] = result

# --- RESULTS SECTION ---
if "result" in st.session_state:
    res = st.session_state["result"]

    # 1. COLLAPSIBLE DECISION TRACE ("Thinking Process")
    with st.expander("⚡ Pipeline Decision Trace (Auditable Execution Log)", expanded=True):
        trace_data = res["decision_trace"]
        st.caption(f"Total Execution Time: **{trace_data['total_duration_ms']} ms** across **{trace_data['total_steps']} milestones**")
        for step in trace_data["steps"]:
            status_icon = "🟢" if step["status"] == "completed" else "🟡"
            st.markdown(
                f"<div class='trace-card'>{status_icon} <b>Step {step['step_number']}: {step['name']}</b> ({step['duration_ms']}ms)<br>"
                f"<i>{step['description']}</i></div>",
                unsafe_allow_html=True
            )

    # 2. CONFLICT ALERTS
    if res["conflicts"]:
        for c in res["conflicts"]:
            st.markdown(
                f"<div class='conflict-box'>⚠️ <b>Sharp Opinion Conflict Detected on '{c['aspect']}':</b> "
                f"Customer opinion is polarized with {c['positive_pct']}% positive vs {c['negative_pct']}% negative mentions. "
                f"Both perspectives are represented below.</div>",
                unsafe_allow_html=True
            )

    # 3. GLOBAL DISTRIBUTION CHARTS
    st.subheader("📊 Exact Corpus Sentiment Proportions")
    st.caption("Mathematically exact, uncorrupted P(S|A) computed across all indexed reviews in SQLite.")
    cols = st.columns(len(res["global_distribution"]) if res["global_distribution"] else 1)

    for i, (asp, stats) in enumerate(res["global_distribution"].items()):
        with cols[i % len(cols)]:
            st.markdown(f"**{asp}** (n={stats['total']})")
            st.progress(stats["positive_pct"] / 100.0, text=f"Pos: {stats['positive_pct']}%")
            st.progress(stats["negative_pct"] / 100.0, text=f"Neg: {stats['negative_pct']}%")

    # 4. DIALECTICAL OPINION SUMMARY
    st.markdown("---")
    st.subheader("📝 Evidence-Grounded Opinion Summary")
    st.markdown(res["summary_text"])

    # 5. ATOMIC NLI VERIFICATION & QUANTIFIERS
    st.markdown("---")
    st.subheader("🛡️ Factual Consistency & Quantifier Verification")
    col_nli, col_qce = st.columns(2)

    with col_nli:
        st.markdown("**NLI Claim Entailment Results:**")
        for claim in res["claims_verification"]:
            badge = "✅ Verified" if claim["status"] == "VERIFIED" else "⚠️ " + claim["status"]
            st.write(f"- {badge}: *\"{claim['atomic_text']}\"* (Entailment: {claim['entailment_score']})")

    with col_qce:
        st.markdown(f"**Quantifier Calibration (Mean QCE: {res['mean_qce']}):**")
        for q in res["quantifier_checks"]:
            if q.get("detected_quantifier"):
                status_icon = "✅" if q["is_calibrated"] else "⚠️ Mismatch"
                st.write(f"- {status_icon} Term *'{q['detected_quantifier']}'*: {q.get('feedback_message') or 'Calibrated with empirical ratio'}")

    # 6. EVIDENCE RECEIPTS EXPLORER
    st.markdown("---")
    st.subheader("🔍 Evidence Receipts & Source Attribution")
    st.caption("Inspect original source reviews cited in the summary.")

    for i, unit in enumerate(res["exemplars"][:12], 1):
        with st.expander(f"[E{i}] {unit['canonical_aspect']} ({unit['sentiment_label'].capitalize()}) — Review #{unit['review_id']}"):
            st.write(f"**Exact Span:** *\"{unit['text_span']}\"*")
            full_rev = pipeline.sqlite.get_review(unit["review_id"])
            if full_rev:
                st.caption(f"Full Review Text: {full_rev['text']}")
                st.caption(f"Original Rating: {full_rev['rating']}★ | Date: {full_rev['date']}")

            # Human-in-the-loop feedback button
            col_fb1, col_fb2 = st.columns([1, 4])
            if col_fb1.button("👍 Confirm", key=f"up_{unit['unit_id']}"):
                fb = UserFeedbackRecord(
                    feedback_id=f"FB_{unit['unit_id']}",
                    unit_id=unit["unit_id"],
                    upvoted=True
                )
                pipeline.feedback_store.record_feedback(fb, text_span=unit["text_span"])
                st.success("Feedback recorded in memory!")

            if col_fb2.button("🚩 Flag Mislabeled Sentiment", key=f"flag_{unit['unit_id']}"):
                corrected_sent = -1 if unit["sentiment"] == 1 else 1
                fb = UserFeedbackRecord(
                    feedback_id=f"FB_{unit['unit_id']}",
                    unit_id=unit["unit_id"],
                    original_sentiment=unit["sentiment"],
                    corrected_sentiment=corrected_sent,
                    upvoted=False,
                    comment="User flagged sentiment polarity"
                )
                pipeline.feedback_store.record_feedback(fb, text_span=unit["text_span"])
                st.warning(f"Saved correction to exemplar cache: Sentiment flipped for '{unit['canonical_aspect']}'!")
