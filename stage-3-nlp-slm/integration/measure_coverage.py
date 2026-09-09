"""
Measures line coverage for integration modules using Python's standard library trace module.
"""
import sys
import trace
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[2]
    integration_dir = root / "stage-3-nlp-slm" / "integration"
    tests_dir = integration_dir / "tests"

    test_files = [
        str(tests_dir / "test_contract_validation.py"),
        str(tests_dir / "test_confidence_gate.py"),
        str(tests_dir / "test_failure_mode_handlers.py"),
        str(tests_dir / "test_integration_audit_logger.py"),
        "--basetemp=.pytest_temp",
        "-q"
    ]

    tracer = trace.Trace(count=1, trace=0)
    tracer.runfunc(__import__("pytest").main, test_files)
    results = tracer.results()

    modules = [
        "contract_validation.py",
        "confidence_gate.py",
        "failure_mode_handlers.py",
        "integration_audit_logger.py"
    ]

    print("\n" + "="*50)
    print("INTEGRATION MODULES CODE COVERAGE AUDIT")
    print("="*50)

    total_exec = 0
    total_cov = 0

    for mod in modules:
        mod_counts = {k[1] for k in results.counts.keys() if mod in k[0]}
        filepath = integration_dir / mod
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        # Use Python's compiler to get the true executable bytecode line numbers
        code = compile(source, str(filepath), "exec")
        executable_lines = set()
        
        def collect_lines(co):
            for start, end, line in co.co_lines():
                if line is not None:
                    executable_lines.add(line)
            for const in co.co_consts:
                if hasattr(const, "co_lines"):
                    collect_lines(const)

        collect_lines(code)


        covered = set(executable_lines).intersection(mod_counts)
        pct = (len(covered) / len(executable_lines) * 100) if executable_lines else 100.0
        total_exec += len(executable_lines)
        total_cov += len(covered)

        status = "PASS (>=90%)" if pct >= 90.0 else "FAIL (<90%)"
        print(f"{mod:30s}: {len(covered):3d}/{len(executable_lines):3d} lines ({pct:5.1f}%) | {status}")
        unhit = sorted(set(executable_lines) - covered)
        if unhit and pct < 90.0:
            print(f"   Unhit lines ({len(unhit)}): {unhit[:20]}")


    overall_pct = (total_cov / total_exec * 100) if total_exec else 100.0
    print("-"*50)
    print(f"{'OVERALL AVERAGE':30s}: {total_cov:3d}/{total_exec:3d} lines ({overall_pct:5.1f}%) | {'PASS' if overall_pct >= 90.0 else 'FAIL'}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
