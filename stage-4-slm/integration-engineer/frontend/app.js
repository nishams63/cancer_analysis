// Clinical Decision Support Frontend Logic (100% Offline)

const SAMPLES = {
  low: "62-year-old male with metastatic EGFR exon 19 deletion NSCLC receiving osimertinib 80mg daily. Tolerating therapy well with no rash, diarrhea, or shortness of breath. Confirms complete absence of acute treatment-limiting toxicities.",
  high: "58-year-old female with advanced NSCLC on osimertinib 80mg daily presenting with progressive dyspnea and persistent dry cough. CT chest demonstrates bilateral ground-glass opacities consistent with Grade 3 drug-induced pneumonitis.",
  neg: "70-year-old male with KRAS G12C colorectal carcinoma on targeted therapy. Denies nausea, denies vomiting, and denies treatment-related diarrhea. Free of any systemic adverse events."
};

function loadSample(key) {
  const input = document.getElementById("clinicalNoteInput");
  if (SAMPLES[key]) {
    input.value = SAMPLES[key];
  }
}

async function generateBriefing() {
  const note = document.getElementById("clinicalNoteInput").value.trim();
  const generateBtn = document.getElementById("generateBtn");
  const safetyBadge = document.getElementById("safetyBadge");
  const resultCard = document.getElementById("resultCard");
  const emptyState = document.getElementById("emptyState");
  const reviewAlert = document.getElementById("reviewAlert");

  if (!note) {
    alert("Please enter or paste a clinical note.");
    return;
  }

  generateBtn.disabled = true;
  generateBtn.textContent = "Processing Local Inference...";
  safetyBadge.className = "badge badge-waiting";
  safetyBadge.textContent = "ANALYZING...";

  try {
    const res = await fetch("/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clinical_note: note })
    });

    if (!res.ok) {
      throw new Error(`Service returned error: ${res.status}`);
    }

    const data = await res.json();

    // Render output
    emptyState.style.display = "none";
    resultCard.style.display = "block";

    // Risk tier
    const riskElem = document.getElementById("riskTier");
    riskElem.textContent = data.risk.toUpperCase();
    riskElem.className = "risk-value";
    if (data.risk.toLowerCase() === "high") {
      riskElem.classList.add("risk-high");
    } else if (data.risk.toLowerCase() === "moderate") {
      riskElem.classList.add("risk-moderate");
    } else {
      riskElem.classList.add("risk-low");
    }

    // Structured fields
    document.getElementById("keyFindingText").textContent = data.key_finding || "None specified.";
    document.getElementById("actionText").textContent = data.action || "None specified.";
    document.getElementById("spokenSummaryText").textContent = data.spoken_summary || "";

    // Metadata
    document.getElementById("confidenceVal").textContent = `${(data.confidence * 100).toFixed(1)}%`;
    document.getElementById("latencyVal").textContent = `${data.latency_ms.toFixed(0)} ms`;
    document.getElementById("modelVal").textContent = data.model_version;
    document.getElementById("inferenceIdVal").textContent = data.inference_id;

    // Safety badge & alert
    if (data.safety_status === "PASS") {
      safetyBadge.className = "badge badge-pass";
      safetyBadge.textContent = "✓ PASS (SAFE)";
      reviewAlert.style.display = "none";
    } else {
      safetyBadge.className = "badge badge-review";
      safetyBadge.textContent = "⚠️ REVIEW REQUIRED";
      reviewAlert.style.display = "block";
      document.getElementById("reviewReasonText").textContent =
        data.failure_reason || "Safety check threshold triggered. Routed to clinical review.";
    }

  } catch (err) {
    alert(`Inference failed: ${err.message}`);
    safetyBadge.className = "badge badge-review";
    safetyBadge.textContent = "ERROR";
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = "Generate Clinical Briefing";
  }
}

function speakBriefing() {
  const summary = document.getElementById("spokenSummaryText").textContent;
  if (!summary || summary === "--") return;

  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(summary);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  } else {
    alert("Local browser SpeechSynthesis API is not supported on this device.");
  }
}
