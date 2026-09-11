#!/usr/bin/env python3
"""
Root entrypoint for Stage 5 — GenAI Synthetic Oncology Stress-Test Engine.
Executes the master integration orchestrator across:
- Data Foundation
- Scenario Library
- Structured Generation & Validation
- RAG Evidence Retrieval
- Clinical Narrative Generation & Validation
- Realism, Plausibility, Fidelity, and Faithfulness Evaluation
- Stage 1-4 Downstream Stress Execution & Failure Isolation
- System Stress Difficulty & Failure Impact Scoring
- Counterfactual Stability & Invariance Testing
- Wildcard Candidate Ranking & Evidence Dossiers
- SQLite + JSONL/CSV Persistence & Dashboard
"""
import sys
import argparse
from pathlib import Path

# Resolve Stage 5 Integration Engineer directory
base_dir = Path(__file__).resolve().parent / "stage 5 Gen-AI" / "Integration Engineer"
sys.path.insert(0, str(base_dir))

from src.integration.orchestrator import MasterOrchestrator
from src.utils.logging import get_logger

logger = get_logger("Stage5MasterCLI")

def main():
    parser = argparse.ArgumentParser(
        description="Master Stage 5 GenAI Synthetic Oncology Stress-Test Engine Runner"
    )
    parser.add_argument("--n", type=int, default=20, help="Number of synthetic oncology scenarios to evaluate (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for reproducibility (default: 42)")
    parser.add_argument("--scenario", type=str, default=None, help="Specific prompt scenario ID from catalog (e.g. PROMPT-R01)")
    parser.add_argument("--batch-id", type=str, default=None, help="Custom unique batch ID identifier")
    parser.add_argument("--rag-enabled", action="store_true", default=True, help="Enable RAG guideline retrieval (default: True)")
    parser.add_argument("--no-rag", dest="rag_enabled", action="store_false", help="Disable RAG guideline retrieval")
    parser.add_argument("--counterfactual-enabled", action="store_true", default=True, help="Enable counterfactual generation & sensitivity check")
    parser.add_argument("--no-counterfactual", dest="counterfactual_enabled", action="store_false", help="Disable counterfactual checks")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted batch execution from step checkpoints")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom export output directory")

    args = parser.parse_args()

    orchestrator = MasterOrchestrator()
    result = orchestrator.run_batch(
        n=args.n,
        seed=args.seed,
        scenario_id_filter=args.scenario,
        batch_id=args.batch_id,
        rag_enabled=args.rag_enabled,
        counterfactual_enabled=args.counterfactual_enabled,
        resume=args.resume
    )

    print("\n" + "="*70)
    print("STAGE 5 GENAI SYNTHETIC ONCOLOGY STRESS-TEST ENGINE COMPLETED")
    print("="*70)
    print(f"Batch ID                  : {result['batch_id']}")
    print(f"Execution Status          : {result['status']}")
    print(f"Total Scenarios Evaluated : {result['scenario_count']}")
    print(f"Ranked Wildcard Candidates: {result['ranked_candidates_count']}")
    print("\nExported Artifacts:")
    for k, v in result["exports"].items():
        print(f"  [{k.upper():<10}] -> {v}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
