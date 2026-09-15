"use client";

import { useEffect } from "react";
import Link from "next/link";
import { AlertTriangle, RefreshCw, Home, Activity } from "lucide-react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[ONCO.AI Telemetry] Runtime boundary error:", error);
  }, [error]);

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "grid",
        placeItems: "center",
        padding: "30px",
        background: "radial-gradient(circle at 50% 30%, #f0f5fc 0%, #e8f0fa 100%)",
        color: "#10243f",
        fontFamily: "Inter, system-ui, -apple-system, sans-serif",
      }}
    >
      <div
        style={{
          maxWidth: "540px",
          width: "100%",
          background: "#ffffff",
          border: "1px solid #dbe5f1",
          borderRadius: "18px",
          padding: "36px",
          boxShadow: "0 18px 48px rgba(18, 48, 80, 0.08)",
          textAlign: "center",
        }}
      >
        <div
          style={{
            width: "56px",
            height: "56px",
            borderRadius: "14px",
            background: "#fff0f2",
            color: "#dc3f50",
            display: "grid",
            placeItems: "center",
            margin: "0 auto 20px",
          }}
        >
          <AlertTriangle size={28} />
        </div>

        <p
          style={{
            fontSize: "11px",
            fontWeight: 800,
            letterSpacing: "0.12em",
            color: "#dc3f50",
            textTransform: "uppercase",
            margin: "0 0 8px",
          }}
        >
          Clinical Stream Interrupted
        </p>

        <h1
          style={{
            fontSize: "22px",
            fontWeight: 750,
            margin: "0 0 12px",
            letterSpacing: "-0.02em",
          }}
        >
          Module Execution Error
        </h1>

        <p
          style={{
            fontSize: "13px",
            lineHeight: 1.6,
            color: "#63738a",
            margin: "0 0 24px",
          }}
        >
          An unexpected interruption occurred within the oncology intelligence pipeline. Patient data integrity and audit logging remain safeguarded.
        </p>

        {error.digest && (
          <div
            style={{
              padding: "10px 14px",
              background: "#f7f9fc",
              border: "1px solid #e1e9f2",
              borderRadius: "8px",
              fontSize: "11px",
              color: "#4a627d",
              fontFamily: "monospace",
              marginBottom: "24px",
              textAlign: "left",
            }}
          >
            Incident Digest: <b>{error.digest}</b>
          </div>
        )}

        <div
          style={{
            display: "flex",
            gap: "12px",
            justifyContent: "center",
            flexWrap: "wrap",
          }}
        >
          <button
            onClick={() => reset()}
            style={{
              minHeight: "42px",
              padding: "0 18px",
              borderRadius: "10px",
              background: "linear-gradient(135deg, #1875ed, #0c55bc)",
              color: "#fff",
              fontWeight: 700,
              fontSize: "13px",
              border: "none",
              cursor: "pointer",
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
            }}
          >
            <RefreshCw size={15} /> Retry Pipeline
          </button>

          <Link
            href="/dashboard"
            style={{
              minHeight: "42px",
              padding: "0 18px",
              borderRadius: "10px",
              background: "#fff",
              border: "1px solid #ccd8e7",
              color: "#2a435f",
              fontWeight: 700,
              fontSize: "13px",
              textDecoration: "none",
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
            }}
          >
            <Home size={15} /> Overview
          </Link>

          <Link
            href="/analytics"
            style={{
              minHeight: "42px",
              padding: "0 18px",
              borderRadius: "10px",
              background: "#edf5ff",
              border: "1px solid #d2e5ff",
              color: "#1769e0",
              fontWeight: 700,
              fontSize: "13px",
              textDecoration: "none",
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
            }}
          >
            <Activity size={15} /> Health Status
          </Link>
        </div>
      </div>
    </div>
  );
}
