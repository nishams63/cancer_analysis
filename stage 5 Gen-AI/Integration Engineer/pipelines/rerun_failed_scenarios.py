import sys
import argparse
from pathlib import Path

integ_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(integ_root))

from src.storage.result_store import ResultStore
from src.storage.failure_store import FailureStore
from src.storage.batch_store import BatchStore
from src.integration.orchestrator import MasterOrchestrator
from src.utils.ids import generate_run_id
from src.utils.logging import get_logger

logger = get_logger("RerunFailedScenarios")

def main():
    parser = argparse.ArgumentParser(description="Targeted Rerun for Failed Stage 5 Scenarios")
    parser.add_argument("--batch-id", type=str, required=True, help="Parent batch ID to inspect")
    parser.add_argument("--failure-code", type=str, default=None, help="Filter by failure code (e.g. F01, F03)")
    parser.add_argument("--scenario-id", type=str, default=None, help="Filter by scenario ID")
    parser.add_argument("--seed", type=int, default=2026, help="Seed for rerun")

    args = parser.parse_args()

    rs = ResultStore()
    fs = FailureStore(rs)
    bs = BatchStore(rs)

    parent_batch = bs.get_batch(args.batch_id)
    if not parent_batch:
        logger.error(f"Parent batch {args.batch_id} not found in database!")
        sys.exit(1)

    failures = fs.list_failures(batch_id=args.batch_id, failure_code=args.failure_code)
    if not failures:
        logger.info(f"No failures found for batch {args.batch_id} matching filter.")
        return

    rerun_scenarios = list(set(f["scenario_id"] for f in failures))
    if args.scenario_id:
        rerun_scenarios = [s for s in rerun_scenarios if s == args.scenario_id]

    rerun_id = generate_run_id("RERUN")
    logger.info(f"Initiating rerun {rerun_id} for {len(rerun_scenarios)} scenarios from parent {args.batch_id}")

    orchestrator = MasterOrchestrator()
    result = orchestrator.run_batch(
        n=len(rerun_scenarios),
        seed=args.seed,
        batch_id=rerun_id
    )

    print("\n========================================================")
    print("STAGE 5 TARGETED RERUN COMPLETED")
    print("========================================================")
    print(f"Parent Batch ID : {args.batch_id}")
    print(f"Rerun ID        : {rerun_id}")
    print(f"Rerun Scenarios : {len(rerun_scenarios)}")
    print(f"Status          : {result['status']}")
    print("========================================================\n")

if __name__ == "__main__":
    main()
