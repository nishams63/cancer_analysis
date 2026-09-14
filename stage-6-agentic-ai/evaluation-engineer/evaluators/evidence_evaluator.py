"""Evidence grounding and hallucination detection evaluator."""
from typing import List, Tuple, Optional
from schemas.result import AgentResult
from schemas.ground_truth import GroundTruth
from schemas.failure import FailureRecord, FailureSeverity, FailureCategory


class EvidenceEvaluator:
    """Verifies that findings are grounded in quantitative evidence and free of forbidden claims."""

    def evaluate(
        self,
        agent_result: AgentResult,
        ground_truth: Optional[GroundTruth] = None,
    ) -> Tuple[float, List[FailureRecord]]:
        failures: List[FailureRecord] = []
        if not ground_truth:
            return 1.0, []

        findings_text = " ".join(
            [f.get("cause", "") + " " + str(f.get("details", "")) for f in agent_result.findings]
        ).lower()

        # 1. Check for Forbidden Claims (Hallucinations / Unsupported assertions)
        forbidden_violations = 0
        for claim in ground_truth.forbidden_claims:
            claim_lower = claim.lower()
            # If significant substring or words match
            claim_words = [w for w in claim_lower.split() if len(w) > 4]
            if all(w in findings_text for w in claim_words) and len(claim_words) >= 3:
                forbidden_violations += 1
                failures.append(
                    FailureRecord(
                        failure_id=f"FAIL-EVID-FORBIDDEN-{claim[:20].replace(' ', '_')}",
                        scenario_id=ground_truth.scenario_id,
                        run_id=agent_result.run_id,
                        category=FailureCategory.F12_HALLUCINATED_FACT,
                        severity=FailureSeverity.CRITICAL,
                        expected=f"Must NOT claim: '{claim}'",
                        actual=f"Findings contain forbidden claim: '{claim}'",
                        evidence=f"Findings excerpt: {findings_text[:200]}",
                        explanation="Agent generated an empirical claim explicitly prohibited by ground truth.",
                        recommended_fix="Constrain LLM synthesis prompts to strictly observed metrics.",
                    )
                )

        # 2. Check evidence grounding score
        # Proportion of findings that have empirical scores or observations
        grounded_count = 0
        total_findings = len(agent_result.findings)
        if total_findings == 0:
            return 1.0 if not ground_truth.expected_facts else 0.5, failures

        for f in agent_result.findings:
            if "score" in f or "effect_size" in f or "details" in f:
                grounded_count += 1

        grounding_ratio = grounded_count / total_findings
        if forbidden_violations > 0:
            grounding_score = max(0.0, grounding_ratio - 0.40)
        else:
            grounding_score = round(grounding_ratio, 4)

        return grounding_score, failures
