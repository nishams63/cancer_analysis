"""
Offline Command-Line Interface for Clinical Decision Support.
Enables terminal-based clinical inference and status inspection without browser or network access.

Usage:
    python cli.py --note "Patient on osimertinib 80mg daily denies adverse toxicities."
    python cli.py --file clinical_note.txt
    python cli.py status
    python cli.py logs --limit 5
"""

import sys
import json
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
from health import HealthChecker


def print_summary(result: dict):
    print("\n=======================================================")
    print("      CLINICAL DECISION SUPPORT BRIEFING (OFFLINE)      ")
    print("=======================================================")
    print(f"Risk:            {result['risk']}")
    print(f"Confidence:      {result['confidence'] * 100:.1f}%")
    print(f"Safety Status:   {result['safety_status']}")
    print(f"Review Required: {result['review_required']}")
    print(f"Latency:         {result['latency_ms']:.1f} ms")
    print(f"Inference ID:    {result['inference_id']}")
    print("-------------------------------------------------------")
    print(f"Key Finding:\n{result['key_finding']}")
    print(f"\nAction:\n{result['action']}")
    print("-------------------------------------------------------")
    print("Spoken Summary (Voice-Ready):")
    print(result['spoken_summary'])
    print("=======================================================\n")

    if result.get('review_required'):
        print("ATTENTION: One or more safety checks failed. Routed to clinician review.")
        print(f"Reason: {result.get('failure_reason')}\n")


def run_status():
    config = load_config()
    model_mgr = ModelManager(model_path=config.model.path, context_length=config.model.context_length, threads=config.model.threads)
    safety_gw = SafetyGateway()
    checker = HealthChecker(model_mgr, safety_gw, config)
    health = checker.check_health()
    print("\n=== SYSTEM HEALTH & OFFLINE STATUS ===")
    print(f"Status:               {health['status']}")
    print(f"Offline Mode:         {health['offline_mode']}")
    print(f"Model Name:           {health['model_name']}")
    print(f"Model Loaded:         {health['model_loaded']}")
    print(f"Quantization:         {health['quantization']}")
    print(f"Context / Threads:    {health['context_length']} / {health['threads']}")
    print(f"Safety Gate Active:   {health['safety_gateway']} (Threshold: {health['confidence_threshold']})")
    print("======================================\n")


def run_logs(limit: int = 5):
    log_path = Path(__file__).resolve().parent / "artifacts" / "audit_log.jsonl"
    if not log_path.exists():
        print(f"No audit log found at {log_path}")
        return

    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"\n=== RECENT AUDIT LOGS (Last {min(limit, len(lines))} of {len(lines)}) ===")
    for line in lines[-limit:]:
        try:
            d = json.loads(line)
            print(f"[{d.get('timestamp')}] ID: {d.get('inference_id')} | Risk: {d.get('firewall_verdict', {}).get('parsed_fields', {}).get('Risk', 'N/A')} | Route: {d.get('firewall_verdict', {}).get('route')} | Latency: {d.get('latency_ms', 0):.1f}ms")
        except Exception:
            print(line[:100])
    print("========================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Clinical SLM Decision Support — 100% Offline CLI",
        epilog="Notice: Clinical Decision Support prototype — human review required when safety checks fail."
    )
    parser.add_argument("command", nargs="?", default="summarize", choices=["summarize", "status", "logs", "health"],
                        help="Command to run: summarize (default), status, health, or logs.")
    parser.add_argument("--file", "-f", type=str, help="Path to text file containing clinical note.")
    parser.add_argument("--note", "-n", type=str, help="Direct inline clinical note text string.")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Number of recent logs to display.")

    args = parser.parse_args()

    if args.command in ("status", "health"):
        run_status()
        return
    elif args.command == "logs":
        run_logs(args.limit)
        return

    # Summarize mode
    clinical_note = ""
    if args.file:
        note_path = Path(args.file)
        if not note_path.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        clinical_note = note_path.read_text(encoding="utf-8")
    elif args.note:
        clinical_note = args.note
    else:
        print("Error: Either --file or --note must be provided for summarize.")
        parser.print_help()
        sys.exit(1)

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

    result = service.process_note(clinical_note)
    print_summary(result)


if __name__ == "__main__":
    main()
