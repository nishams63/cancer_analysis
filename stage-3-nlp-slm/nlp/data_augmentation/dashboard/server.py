"""
Lightweight Localhost Dashboard Server for Stage 3 Clinical NLP Data Augmentation.
Serves the web dashboard on http://localhost:8505 and provides REST API for metrics.
"""

import http.server
import socketserver
import json
from pathlib import Path

PORT = 8505
DASHBOARD_DIR = Path(__file__).resolve().parent
RESULTS_DIR = DASHBOARD_DIR.parent / "results"


class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            payload = {}
            # Load diversity metrics
            div_file = RESULTS_DIR / "diversity_metrics.json"
            if div_file.exists():
                payload["diversity"] = json.loads(div_file.read_text(encoding="utf-8"))
            # Load QC summary
            qc_file = RESULTS_DIR / "qc_summary.json"
            if qc_file.exists():
                payload["qc"] = json.loads(qc_file.read_text(encoding="utf-8"))
            # Load leakage audit
            leak_file = RESULTS_DIR / "leakage_audit.json"
            if leak_file.exists():
                payload["leakage"] = json.loads(leak_file.read_text(encoding="utf-8"))
            # Load experiment results if ready
            exp_file = RESULTS_DIR / "experiment_results.json"
            if exp_file.exists():
                payload["experiments"] = json.loads(exp_file.read_text(encoding="utf-8"))

            self.wfile.write(json.dumps(payload).encode("utf-8"))
        else:
            super().do_GET()


def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), DashboardRequestHandler) as httpd:
        print(f"Stage 3 Clinical NLP Dashboard active at: http://localhost:{PORT}")
        print("Press Ctrl+C to terminate.")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
