"""Failure Clustering: groups failures by patterns and produces cluster table."""
import pandas as pd
from typing import Dict, Any, List
from collections import defaultdict


class FailureClusterer:
    """Groups failures by dominant failure code, category, and affected stages."""

    def __init__(self):
        pass

    def cluster_failures(self, failure_records: List[Dict[str, Any]]) -> pd.DataFrame:
        if not failure_records:
            return pd.DataFrame(columns=[
                "cluster_id", "dominant_failure_code", "scenario_category",
                "affected_stage", "count", "average_confidence", "average_difficulty",
                "representative_scenarios"
            ])

        groups = defaultdict(list)
        for r in failure_records:
            f_code = r.get("failure_code", "F20")
            st = r.get("stage", "unknown")
            groups[(f_code, st)].append(r)

        cluster_rows = []
        for idx, ((f_code, st), recs) in enumerate(groups.items(), 1):
            confs = [r.get("confidence", 0.0) for r in recs if r.get("confidence")]
            diffs = [r.get("difficulty_score", 50.0) for r in recs]
            sids = list(set(r.get("scenario_id", "") for r in recs))[:4]
            
            cluster_rows.append({
                "cluster_id": f"CLUST-{idx:03d}",
                "dominant_failure_code": f_code,
                "scenario_category": "Resistance & Bypass" if f_code in ["F01", "F04", "F06"] else "High Risk Progression",
                "affected_stage": st,
                "count": len(recs),
                "average_confidence": round(sum(confs) / max(1, len(confs)), 3),
                "average_difficulty": round(sum(diffs) / max(1, len(diffs)), 1),
                "representative_scenarios": ", ".join(sids)
            })

        return pd.DataFrame(cluster_rows)
