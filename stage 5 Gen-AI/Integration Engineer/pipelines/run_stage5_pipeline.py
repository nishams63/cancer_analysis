import sys
import argparse
from pathlib import Path

# Add Integration Engineer to path
integ_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(integ_root))

from src.integration.orchestrator import MasterOrchestrator
from src.utils.logging import get_logger

logger = get_logger("RunStage5Pipeline")

def main():
    parser = argparse.ArgumentParser(description="Stage 5 GenAI Synthetic Oncology Stress-Test Master Pipeline")
    parser.add_argument("--n", type=int, default=20, help="Number of scenarios to generate and evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--scenario", type=str, default=None, help="Optional specific prompt scenario ID (e.g. PROMPT-R01)")
    parser.add_argument("--batch-id", type=str, default=None, help="Custom Batch ID")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG retrieval")
    parser.add_argument("--no-counterfactual", action="store_true", help="Disable counterfactual generation")
    parser.add_argument("--resume", action="store_true", help="Resume from previous checkpoints")
    parser.add_argument("--db-path", type=str, default=None, help="Custom SQLite database path")

    args = parser.parse_args()

    orchestrator = MasterOrchestrator(db_path=args.db_path)
    result = orchestrator.run_batch(
        n=args.n,
        seed=args.seed,
        scenario_id_filter=args.scenario,
        batch_id=args.batch_id,
        rag_enabled=not args.no_rag,
        counterfactual_enabled=not args.no_counterfactual,
        resume=args.resume
    )

    print("\n========================================================")
    print("STAGE 5 GENAI STRESS-TEST ENGINE BATCH COMPLETED")
    print("========================================================")
    print(f"Batch ID                 : {result['batch_id']}")
    print(f"Status                   : {result['status']}")
    print(f"Scenarios Evaluated      : {result['scenario_count']}")
    print(f"Ranked Wildcard Candidates: {result['ranked_candidates_count']}")
    print("Exports:")
    for k, v in result["exports"].items():
        print(f"  - {k.capitalize()}: {v}")
    print("========================================================\n")

if __name__ == "__main__":
    main()
