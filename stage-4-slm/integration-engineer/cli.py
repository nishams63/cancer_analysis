"""
Offline Command-Line Interface for Clinical Decision Support.
Enables terminal-based clinical inference without browser or network access:
Usage:
    python cli.py --file clinical_note.txt
    python cli.py --note "Patient with EGFR-mutated NSCLC on osimertinib 80mg daily..."
"""

import sys
import argparse
from pathlib import Path

# Add src directory to sys.path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_config
from model_manager import ModelManager
from safety_gateway import SafetyGateway
from audit_logger import AuditLogger
from inference_service import InferenceService


def main():
    parser = argparse.ArgumentParser(
        description="Clinical SLM Decision Support — 100% Offline CLI",
        epilog="Notice: Clinical Decision Support prototype — human review required when safety checks fail."
    )
    parser.add_argument("--file", "-f", type=str, help="Path to text file containing unstructured clinical progress note.")
    parser.add_argument("--note", "-n", type=str, help="Direct inline clinical note text string.")

    args = parser.parse_args()

    if args.file:
        note_path = Path(args.file)
        if not note_path.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        clinical_note = note_path.read_text(encoding="utf-8")
    elif args.note:
        clinical_note = args.note
    else:
        print("Error: Either --file or --note must be provided.")
        parser.print_help()
        sys.exit(1)

    # Initialize components
    config = load_config()
    model_manager = ModelManager(
        model_path=config.model.path,
        context_length=config.model.context_length,
        threads=config.model.threads
    )
    safety_gateway = SafetyGateway(
        strict_mode=config.safety.strict_mode,
        confidence_threshold=config.safety.confidence_threshold
    )
    audit_logger = AuditLogger(log_path=config.logging.audit_log_path)
    service = InferenceService(config, model_manager, safety_gateway, audit_logger)

    # Run inference
    result = service.process_note(clinical_note)

    # Print conservative clinical decision support output
    print("\n=======================================================")
    print("      CLINICAL DECISION SUPPORT BRIEFING (OFFLINE)      ")
    print("=======================================================")
    print(f"Risk: {result['risk']}")
    print(f"\nKey Finding:\n{result['key_finding']}")
    print(f"\nAction:\n{result['action']}")
    print(f"\nConfidence: {result['confidence'] * 100:.1f}%")
    print(f"Safety: {result['safety_status']}")
    print(f"Review Required: {result['review_required']}")
    print(f"Latency: {result['latency_ms']:.1f} ms")
    print(f"Inference ID: {result['inference_id']}")
    print("-------------------------------------------------------")
    print("Spoken Summary (Voice-Ready):")
    print(result['spoken_summary'])
    print("=======================================================\n")

    if result['review_required']:
        print("ATTENTION: One or more safety checks failed. Routed to clinician review.")
        print(f"Reason: {result.get('failure_reason')}\n")


if __name__ == "__main__":
    main()
