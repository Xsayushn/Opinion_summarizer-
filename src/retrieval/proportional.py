"""
Proportional Stratified Exemplar Sampler.
Eliminates retrieval volume bias by sampling evidence units to reflect global sentiment proportions
while enforcing a minimum minority quota for conflict visibility.
"""

from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from src.indexing.sqlite_store import SQLiteStore
from src.ingestion.absa_pipeline import OpinionUnit
from src.config import DEFAULT_CONTEXT_EXEMPLAR_K, MINORITY_QUOTA_RATIO


class ProportionalSampler:
    """
    Samples evidence exemplars proportionally across aspects and polarities
    based on exact global population distributions from SQLite.
    """

    def __init__(self, sqlite_store: SQLiteStore, minority_quota: float = MINORITY_QUOTA_RATIO):
        self.sqlite = sqlite_store
        self.minority_quota = minority_quota

    def sample_exemplars(
        self,
        candidate_ids: List[str],
        target_aspects: List[str],
        global_stats: Dict[str, Dict[str, Any]],
        context_budget: int = DEFAULT_CONTEXT_EXEMPLAR_K
    ) -> Tuple[List[OpinionUnit], Dict[str, Any]]:
        """
        Performs proportional stratified sampling.
        Returns:
            (selected_units, sampling_audit_log)
        """
        if not candidate_ids or not target_aspects:
            return [], {"status": "empty_input", "selected_count": 0}

        # 1. Fetch unit metadata from SQLite
        all_candidates = self.sqlite.get_opinion_units_by_ids(candidate_ids)

        # Preserve the candidate ranking order from RRF
        id_to_rank = {uid: i for i, uid in enumerate(candidate_ids)}
        all_candidates.sort(key=lambda u: id_to_rank.get(u.unit_id, 999999))

        # 2. Bucket candidates by (aspect, sentiment)
        # sentiment: 1 (Pos), 0 (Neu), -1 (Neg)
        buckets: Dict[str, Dict[int, List[OpinionUnit]]] = defaultdict(lambda: defaultdict(list))
        for u in all_candidates:
            if u.canonical_aspect in target_aspects:
                buckets[u.canonical_aspect][u.sentiment].append(u)

        aspect_budget = max(4, context_budget // max(1, len(target_aspects)))
        selected_units: List[OpinionUnit] = []
        audit_log: Dict[str, Any] = {"aspect_allocations": {}, "total_selected": 0}

        # 3. For each aspect, allocate quota according to global stats
        for asp in target_aspects:
            asp_stats = global_stats.get(asp, {"positive_pct": 50.0, "neutral_pct": 20.0, "negative_pct": 30.0, "total": 0})

            pos_pct = asp_stats["positive_pct"] / 100.0
            neu_pct = asp_stats["neutral_pct"] / 100.0
            neg_pct = asp_stats["negative_pct"] / 100.0

            # Calculate raw quotas
            q_pos = round(aspect_budget * pos_pct)
            q_neu = round(aspect_budget * neu_pct)
            q_neg = round(aspect_budget * neg_pct)

            # Minority guarantee: If negative or positive is >= 15% in global stats, ensure at least 1-2 slots
            min_slots = max(1, int(aspect_budget * self.minority_quota))
            if neg_pct >= self.minority_quota and q_neg < min_slots:
                q_neg = min_slots
            if pos_pct >= self.minority_quota and q_pos < min_slots:
                q_pos = min_slots

            # Normalize to aspect_budget
            total_q = q_pos + q_neu + q_neg
            if total_q == 0:
                q_pos, q_neu, q_neg = aspect_budget // 3, aspect_budget // 3, aspect_budget // 3

            # Pick top candidates from each bucket
            asp_pos = buckets[asp][1][:q_pos]
            asp_neu = buckets[asp][0][:q_neu]
            asp_neg = buckets[asp][-1][:q_neg]

            # Fallback if a bucket has fewer candidates than quota
            chosen_for_aspect = asp_pos + asp_neu + asp_neg

            # If still under budget, fill with remaining candidates from this aspect
            if len(chosen_for_aspect) < aspect_budget:
                chosen_ids = {u.unit_id for u in chosen_for_aspect}
                remaining = [u for s in [1, -1, 0] for u in buckets[asp][s] if u.unit_id not in chosen_ids]
                needed = aspect_budget - len(chosen_for_aspect)
                chosen_for_aspect.extend(remaining[:needed])

            selected_units.extend(chosen_for_aspect)

            audit_log["aspect_allocations"][asp] = {
                "target_quota": {"pos": q_pos, "neu": q_neu, "neg": q_neg},
                "actual_selected": {
                    "pos": len(asp_pos),
                    "neu": len(asp_neu),
                    "neg": len(asp_neg),
                    "total": len(chosen_for_aspect)
                },
                "global_percentages": {
                    "pos": asp_stats["positive_pct"],
                    "neu": asp_stats["neutral_pct"],
                    "neg": asp_stats["negative_pct"]
                }
            }

        audit_log["total_selected"] = len(selected_units)
        return selected_units, audit_log
