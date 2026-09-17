"""
Dialectical Prompt Builder with Global Counts Injection.
"""

from typing import List, Dict, Any, Tuple
from src.ingestion.absa_pipeline import OpinionUnit


class PromptBuilder:
    """
    Constructs constrained, dialectical opinion summarization prompts.
    Injects unbiased global corpus statistics up-front to calibrate quantifiers.
    """

    @staticmethod
    def build_summary_prompt(
        query: str,
        target_aspects: List[str],
        global_stats: Dict[str, Dict[str, Any]],
        conflicts: List[Dict[str, Any]],
        exemplars: List[OpinionUnit]
    ) -> Tuple[str, Dict[str, str]]:
        """
        Builds the prompt and returns (prompt_text, evidence_id_to_unit_map).
        """
        # Map unit_ids to readable short citations [E1], [E2]...
        evidence_map: Dict[str, str] = {}
        exemplar_lines = []

        for idx, unit in enumerate(exemplars, 1):
            tag = f"[E{idx}]"
            evidence_map[tag] = unit.unit_id
            sentiment_sym = "+" if unit.sentiment == 1 else ("-" if unit.sentiment == -1 else "o")
            exemplar_lines.append(f"{tag} ({unit.canonical_aspect}, {sentiment_sym}): \"{unit.text_span}\"")

        # Format global statistics table
        stats_lines = []
        for asp in target_aspects:
            s = global_stats.get(asp, {"positive_pct": 0, "neutral_pct": 0, "negative_pct": 0, "total": 0})
            stats_lines.append(
                f"- **{asp}** (Total Mentions: n={s['total']}): "
                f"{s['positive_pct']}% Positive, {s['neutral_pct']}% Neutral, {s['negative_pct']}% Negative"
            )

        # Format conflict warnings
        conflict_lines = []
        if conflicts:
            for c in conflicts:
                conflict_lines.append(f"- Alert: Disagreement on {c['aspect']} ({c['positive_pct']}% Pos vs {c['negative_pct']}% Neg)")
        else:
            conflict_lines.append("- No sharp conflict detected.")

        prompt = f"""You are an objective, evidence-grounded Opinion Summarizer.
Summarize customer opinions regarding the user query based strictly on the provided evidence and true global statistics.

### USER QUERY:
"{query}"

### EXACT GLOBAL OPINION METRICS (From full corpus, uncorrupted):
{chr(10).join(stats_lines)}

### NOTED CONFLICTS:
{chr(10).join(conflict_lines)}

### RETRIEVED EVIDENCE EXEMPLARS:
{chr(10).join(exemplar_lines)}

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
"""
        return prompt, evidence_map
