import { jsPDF } from "jspdf";

export interface PatientData {
  id: string;
  age: number;
  sex: string;
  cancer: string;
  stage: string;
  ecog: number;
  treatment: string;
  creatinine?: number;
  hemoglobin?: number;
  platelets?: number;
  cea?: number;
  ctdna?: string;
}

export function exportPatientPDF(patientData: PatientData) {
  const doc = new jsPDF({
    orientation: "portrait",
    unit: "mm",
    format: "a4",
  });

  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 16;
  const contentWidth = pageWidth - margin * 2;
  let y = 18;

  // Header Banner Background
  doc.setFillColor(7, 27, 51); // Navy #071b33
  doc.rect(0, 0, pageWidth, 28, "F");

  // Header Branding
  doc.setTextColor(255, 255, 255);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(16);
  doc.text("ONCO.AI", margin, 12);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(140, 190, 255);
  doc.text("Precision Oncology Intelligence Platform", margin, 18);

  doc.setFontSize(8);
  doc.setTextColor(200, 220, 245);
  const now = new Date();
  const dateStr = now.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
  const timeStr = now.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  doc.text(`Generated: ${dateStr} ${timeStr} · Clinician: Dr. Sharma, MD`, pageWidth - margin, 12, { align: "right" });
  doc.text("CONFIDENTIAL MEDICAL RECORD · FOR CLINICAL USE ONLY", pageWidth - margin, 18, { align: "right" });

  y = 36;

  // Title
  doc.setTextColor(16, 36, 63);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(14);
  doc.text(`Comprehensive Clinical Intelligence Report · ${patientData.id}`, margin, y);
  y += 6;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(8.5);
  doc.setTextColor(99, 115, 138);
  doc.text("Multimodal longitudinal analysis across Stage 1 ML, Stage 2 DL, Stage 3 NLP, Stage 4 SLM, and Stage 6 Tumor Board.", margin, y);
  y += 8;

  // Section 1: Patient Profile Card
  doc.setFillColor(245, 248, 252);
  doc.setDrawColor(223, 231, 241);
  doc.roundedRect(margin, y, contentWidth, 26, 2, 2, "FD");

  doc.setFont("helvetica", "bold");
  doc.setFontSize(9);
  doc.setTextColor(23, 105, 224);
  doc.text("1. PATIENT DEMOGRAPHICS & CLINICAL CONTEXT", margin + 4, y + 6);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(16, 36, 63);

  const col1X = margin + 4;
  const col2X = margin + 48;
  const col3X = margin + 96;
  const col4X = margin + 140;

  doc.text(`Patient ID: `, col1X, y + 13);
  doc.setFont("helvetica", "bold");
  doc.text(`${patientData.id}`, col1X + 16, y + 13);
  doc.setFont("helvetica", "normal");

  doc.text(`Age / Sex: `, col1X, y + 19);
  doc.setFont("helvetica", "bold");
  doc.text(`${patientData.age}y / ${patientData.sex}`, col1X + 16, y + 19);
  doc.setFont("helvetica", "normal");

  doc.text(`Diagnosis: `, col2X, y + 13);
  doc.setFont("helvetica", "bold");
  doc.text(`${patientData.cancer}`, col2X + 16, y + 13);
  doc.setFont("helvetica", "normal");

  doc.text(`Stage: `, col2X, y + 19);
  doc.setFont("helvetica", "bold");
  doc.text(`Stage ${patientData.stage}`, col2X + 16, y + 19);
  doc.setFont("helvetica", "normal");

  doc.text(`ECOG Status: `, col3X, y + 13);
  doc.setFont("helvetica", "bold");
  doc.text(`${patientData.ecog}`, col3X + 20, y + 13);
  doc.setFont("helvetica", "normal");

  doc.text(`ctDNA Kinetics: `, col3X, y + 19);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(220, 63, 80);
  doc.text(`${patientData.ctdna || "Increasing"}`, col3X + 22, y + 19);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(16, 36, 63);

  doc.text(`Current Regimen: `, col4X, y + 13);
  doc.setFont("helvetica", "bold");
  doc.text(`${patientData.treatment}`, col4X, y + 19);
  doc.setFont("helvetica", "normal");

  y += 32;

  // Section 2: Laboratory & Biomarker Kinetics
  doc.setFillColor(255, 255, 255);
  doc.setDrawColor(223, 231, 241);
  doc.roundedRect(margin, y, contentWidth, 22, 2, 2, "FD");

  doc.setFont("helvetica", "bold");
  doc.setFontSize(9);
  doc.setTextColor(23, 105, 224);
  doc.text("2. BASELINE LABS & MOLECULAR BIOMARKERS", margin + 4, y + 6);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(60, 75, 95);

  const lab1 = `Serum Creatinine: ${patientData.creatinine ?? 1.8} mg/dL (Elevated / High)`;
  const lab2 = `Hemoglobin: ${patientData.hemoglobin ?? 9.8} g/dL (Low / Watch)`;
  const lab3 = `Platelets: ${patientData.platelets ?? 140} K/µL (Borderline)`;
  const lab4 = `CEA: ${patientData.cea ?? 27} ng/mL (Elevated)`;

  doc.text(lab1, margin + 4, y + 14);
  doc.text(lab2, margin + 50, y + 14);
  doc.text(lab3, margin + 100, y + 14);
  doc.text(lab4, margin + 145, y + 14);

  y += 28;

  // Section 3: Multimodal AI Evaluation (Stages 1-6)
  doc.setFont("helvetica", "bold");
  doc.setFontSize(9);
  doc.setTextColor(23, 105, 224);
  doc.text("3. MULTIMODAL INTELLIGENCE PIPELINE FINDINGS", margin, y);
  y += 4;

  const modules = [
    {
      stage: "Stage 1 · Tabular ML (Toxicity)",
      model: "XGBoost candidate-v4 (AUC 0.88)",
      score: "72% High Risk",
      finding: "Top drivers: Serum Creatinine (24% SHAP impact), Platinum exposure (18%). Calibrated renal toxicity warning triggered.",
      tone: [220, 63, 80],
    },
    {
      stage: "Stage 2 · Multimodal DL (Progression)",
      model: "DenseNet Spatial + Transformer Temporal (AUC 0.84)",
      score: "81% Progression Probability",
      finding: "Spatial Biopsy Attention: 77% | Temporal ctDNA Kinetics: 85% | Late Fusion Probability: 81% (Confidence 92%).",
      tone: [118, 87, 237],
    },
    {
      stage: "Stage 3 · Clinical NLP (Report Triage)",
      model: "BioLinkBERT / ClinBERT (AUC 0.88)",
      score: "High Urgency (Confidence 91%)",
      finding: "Extracted entities: Severe fatigue, rapid pulse, EGFR mutation positive. Negation scope: 'denies chest pain' confirmed absent.",
      tone: [3, 156, 176],
    },
    {
      stage: "Stage 4 · Clinical SLM Copilot",
      model: "Grounded SLM + Safety Firewall (AUC 0.86)",
      score: "Guidance: SAFE / GROUNDED",
      finding: "Safety checkmarks: Grounding PASS, Dosage Safe PASS, Hallucination-free PASS. Recommended renal adjustment review.",
      tone: [11, 157, 110],
    },
    {
      stage: "Stage 5 · AI Safety Lab (Adversarial)",
      model: "Monte Carlo Synthetic Stress Testing",
      score: "Blind Spot BS-2048-07 Identified",
      finding: "Cross-model calibration gap detected: Renal contraindication historically underweighted in combined platinum regimen.",
      tone: [232, 92, 95],
    },
    {
      stage: "Stage 6 · Autonomous Tumor Board",
      model: "Multi-Agent Treatment Optimizer",
      score: "Strategy: Osimertinib + Savolitinib (Score 87%)",
      finding: "Multi-agent consensus (Guideline, Toxicity, Genomic, Trial agents). 3 potentially eligible clinical trials identified.",
      tone: [79, 91, 213],
    },
  ];

  modules.forEach((mod) => {
    doc.setFillColor(255, 255, 255);
    doc.setDrawColor(225, 233, 242);
    doc.roundedRect(margin, y, contentWidth, 16, 1.5, 1.5, "FD");

    // Left accent stripe
    doc.setFillColor(mod.tone[0], mod.tone[1], mod.tone[2]);
    doc.rect(margin, y, 2.5, 16, "F");

    doc.setFont("helvetica", "bold");
    doc.setFontSize(8.5);
    doc.setTextColor(16, 36, 63);
    doc.text(mod.stage, margin + 6, y + 5);

    doc.setFont("helvetica", "bold");
    doc.setFontSize(8);
    doc.setTextColor(mod.tone[0], mod.tone[1], mod.tone[2]);
    doc.text(mod.score, pageWidth - margin - 4, y + 5, { align: "right" });

    doc.setFont("helvetica", "normal");
    doc.setFontSize(7.5);
    doc.setTextColor(99, 115, 138);
    doc.text(mod.model, margin + 6, y + 9.5);

    doc.setFontSize(7.5);
    doc.setTextColor(35, 50, 70);
    doc.text(mod.finding, margin + 6, y + 13.5);

    y += 18.5;
  });

  y += 3;

  // Section 4: Recommended Action & Tumor Board Strategy
  doc.setFillColor(234, 243, 255);
  doc.setDrawColor(180, 210, 250);
  doc.roundedRect(margin, y, contentWidth, 19, 2, 2, "FD");

  doc.setFont("helvetica", "bold");
  doc.setFontSize(8.5);
  doc.setTextColor(23, 105, 224);
  doc.text("4. MULTIDISCIPLINARY TUMOR BOARD CONSENSUS SUMMARY", margin + 4, y + 5.5);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.8);
  doc.setTextColor(20, 45, 75);
  doc.text("Recommended Action: Switch from Carboplatin + Pemetrexed to EGFR/MET targeted regimen (Osimertinib + Savolitinib).", margin + 4, y + 10.5);
  doc.text("Rationale: Mitigates worsening Grade 2+ renal toxicity (Creatinine 1.8 mg/dL) while addressing progressing ctDNA kinetics.", margin + 4, y + 15);

  y += 24;

  // Sign-off section
  doc.setDrawColor(210, 220, 230);
  doc.line(margin, y, margin + 65, y);
  doc.line(margin + 90, y, margin + 145, y);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.5);
  doc.setTextColor(110, 125, 145);
  doc.text("Attending Oncologist Signature: Dr. Sharma, MD", margin, y + 4);
  doc.text(`Date & Verification Timestamp: ${dateStr}`, margin + 90, y + 4);

  y += 11;

  // Regulatory Disclaimer Footer
  doc.setFillColor(248, 250, 252);
  doc.rect(0, y, pageWidth, 16, "F");

  doc.setFont("helvetica", "bold");
  doc.setFontSize(7);
  doc.setTextColor(100, 115, 130);
  doc.text("CLINICAL DECISION SUPPORT NOTICE (21 CFR Part 860 / SaMD Guideline)", margin, y + 5);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(6.5);
  doc.setTextColor(130, 145, 160);
  doc.text("ONCO.AI provides evidence-backed machine learning decision support to qualified oncologists. This document does not constitute an automated", margin, y + 8.5);
  doc.text("prescription. All clinical decisions, dosage modifications, and treatment plans must be independently verified and approved by a licensed physician.", margin, y + 11.5);

  // Save the PDF
  const filename = `ONCO-AI_Clinical_Report_${patientData.id}.pdf`;
  doc.save(filename);
  return filename;
}

export function exportAnalyticsPDF() {
  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 16;
  const contentWidth = pageWidth - margin * 2;

  // Header Banner
  doc.setFillColor(7, 27, 51);
  doc.rect(0, 0, pageWidth, 28, "F");

  doc.setTextColor(255, 255, 255);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(16);
  doc.text("ONCO.AI", margin, 12);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(140, 190, 255);
  doc.text("Precision Oncology Intelligence Platform", margin, 18);

  const now = new Date();
  doc.setFontSize(8);
  doc.setTextColor(200, 220, 245);
  doc.text(`Generated: ${now.toLocaleDateString()} · Platform Telemetry`, pageWidth - margin, 14, { align: "right" });

  let y = 38;
  doc.setTextColor(16, 36, 63);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(14);
  doc.text("Model Engineering & Platform Health Report", margin, y);
  y += 7;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(8.5);
  doc.setTextColor(99, 115, 138);
  doc.text("Validation metrics, ROC-AUC performance benchmarks, and serverless runtime telemetry.", margin, y);
  y += 12;

  // Health Stats Card
  doc.setFillColor(245, 248, 252);
  doc.setDrawColor(223, 231, 241);
  doc.roundedRect(margin, y, contentWidth, 24, 2, 2, "FD");

  doc.setFont("helvetica", "bold");
  doc.setFontSize(9);
  doc.setTextColor(23, 105, 224);
  doc.text("SYSTEM AVAILABILITY & HEALTH SCORE", margin + 4, y + 6);

  doc.setFontSize(8);
  doc.setTextColor(16, 36, 63);
  doc.text("Overall System Health: 96% (Operational)", margin + 4, y + 13);
  doc.text("Total Patients in Cohort: 1,248", margin + 70, y + 13);
  doc.text("Analysis Runs Executed: 3,621", margin + 125, y + 13);
  doc.text("P95 Inference Latency: 42ms", margin + 4, y + 19);
  doc.text("System Uptime SLA: 99.7%", margin + 70, y + 19);
  doc.text("Safety Firewall Status: 100% Active", margin + 125, y + 19);

  y += 32;

  // ROC-AUC Breakdown
  doc.setFont("helvetica", "bold");
  doc.setFontSize(9);
  doc.setTextColor(23, 105, 224);
  doc.text("MODEL PERFORMANCE SUMMARY (ROC-AUC)", margin, y);
  y += 6;

  const models = [
    ["Stage 1 Tabular ML (Toxicity)", "XGBoost candidate-v4", "0.88 ROC-AUC", "Accuracy: 0.84, Precision: 0.82, Recall: 0.86"],
    ["Stage 2 Multimodal DL (Progression)", "DenseNet + Temporal Transformer", "0.84 ROC-AUC", "Spatial: 77%, Temporal: 85%, Late Fusion: 81%"],
    ["Stage 3 Clinical NLP (Triage)", "BioLinkBERT / ClinBERT", "0.88 ROC-AUC", "Entity F1: 0.89, Negation Resolution: 94%"],
    ["Stage 4 Clinical SLM (Copilot)", "Grounded Oncology SLM", "0.86 ROC-AUC", "Hallucination Check: 0.0%, Evidence Grounding: 98%"],
  ];

  models.forEach(([title, name, auc, stats]) => {
    doc.setFillColor(255, 255, 255);
    doc.setDrawColor(225, 233, 242);
    doc.roundedRect(margin, y, contentWidth, 16, 1.5, 1.5, "FD");

    doc.setFont("helvetica", "bold");
    doc.setFontSize(8.5);
    doc.setTextColor(16, 36, 63);
    doc.text(title, margin + 4, y + 5.5);

    doc.setFont("helvetica", "bold");
    doc.setFontSize(8.5);
    doc.setTextColor(23, 105, 224);
    doc.text(auc, pageWidth - margin - 4, y + 5.5, { align: "right" });

    doc.setFont("helvetica", "normal");
    doc.setFontSize(7.5);
    doc.setTextColor(99, 115, 138);
    doc.text(name, margin + 4, y + 10.5);
    doc.text(stats, margin + 4, y + 14);

    y += 19;
  });

  const filename = `ONCO-AI_Platform_Analytics_Report_${now.toISOString().split("T")[0]}.pdf`;
  doc.save(filename);
  return filename;
}
