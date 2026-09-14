import Link from "next/link";
import { Dna, ArrowLeft, Users, Home } from "lucide-react";

export default function NotFound() {
  return (
    <div
      style={{
        minHeight: "100vh",
        display: "grid",
        placeItems: "center",
        padding: "30px",
        background: "radial-gradient(circle at 50% 30%, #f4f8fc 0%, #e8f0fa 100%)",
        color: "#10243f",
        fontFamily: "Inter, system-ui, -apple-system, sans-serif",
      }}
    >
      <div
        style={{
          maxWidth: "520px",
          width: "100%",
          background: "#ffffff",
          border: "1px solid #dbe5f1",
          borderRadius: "18px",
          padding: "38px",
          boxShadow: "0 18px 48px rgba(18, 48, 80, 0.08)",
          textAlign: "center",
        }}
      >
        <div
          style={{
            width: "58px",
            height: "58px",
            borderRadius: "14px",
            background: "#eaf3ff",
            color: "#1769e0",
            display: "grid",
            placeItems: "center",
            margin: "0 auto 18px",
          }}
        >
          <Dna size={30} />
        </div>

        <p
          style={{
            fontSize: "11px",
            fontWeight: 800,
            letterSpacing: "0.12em",
            color: "#1769e0",
            textTransform: "uppercase",
            margin: "0 0 8px",
          }}
        >
          404 · Clinical Record Not Found
        </p>

        <h1
          style={{
            fontSize: "24px",
            fontWeight: 750,
            margin: "0 0 12px",
            letterSpacing: "-0.02em",
          }}
        >
          Requested Page or Record Unavailable
        </h1>

        <p
          style={{
            fontSize: "13px",
            lineHeight: 1.6,
            color: "#63738a",
            margin: "0 0 26px",
          }}
        >
          The patient ID, diagnostic report, or platform path you requested could not be located in the current clinical cohort.
        </p>

        <div
          style={{
            display: "flex",
            gap: "12px",
            justifyContent: "center",
            flexWrap: "wrap",
          }}
        >
          <Link
            href="/overview"
            style={{
              minHeight: "42px",
              padding: "0 18px",
              borderRadius: "10px",
              background: "linear-gradient(135deg, #1875ed, #0c55bc)",
              color: "#fff",
              fontWeight: 700,
              fontSize: "13px",
              textDecoration: "none",
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
            }}
          >
            <Home size={15} /> Return to Command Center
          </Link>

          <Link
            href="/patients"
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
            <Users size={15} /> Patient Directory
          </Link>
        </div>
      </div>
    </div>
  );
}
