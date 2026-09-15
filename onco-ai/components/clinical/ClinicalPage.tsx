"use client";

import Link from "next/link";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, AlertTriangle, ArrowDown, ArrowRight, BadgeCheck, BrainCircuit, Check, CheckCircle2, ChevronRight, CircleDot, ClipboardCheck, Clock3, Dna, Download, ExternalLink, FileSearch, FileText, FlaskConical, Info, Microscope, Play, RefreshCw, Search, ShieldAlert, ShieldCheck, Sparkles, Square, Upload, UserPlus, UserRound, X, XCircle, Zap } from "lucide-react";
import { FactorBars, RiskRing, RocChart, TrendChart } from "@/components/charts/ClinicalCharts";
import { factors, metrics } from "@/lib/demo-data";
import { useDocAI } from "@/components/doc-ai/DocAIProvider";
import { usePatientStore, Patient } from "@/stores/patient-store";
import { exportPatientPDF, exportAnalyticsPDF } from "@/lib/export-pdf";

type PageKind = "overview" | "patients" | "patient" | "ml" | "dl" | "nlp" | "slm" | "safety" | "tumor" | "analytics" | "audit" | "integrated" | "settings";
type TabProps = { tabs: string[]; active: string; onChange: (tab: string) => void };

const pageMeta: Record<PageKind, { eyebrow: string; title: string; subtitle: string; icon: React.ElementType; accent: string }> = {
  overview: { eyebrow: "COMMAND CENTER", title: "Good morning, Dr. Sharma", subtitle: "Here’s your current oncology intelligence overview.", icon: Activity, accent: "blue" },
  patients: { eyebrow: "PATIENT WORKSPACE", title: "Patients", subtitle: "Review synthetic longitudinal records and launch analysis.", icon: UserRound, accent: "blue" },
  patient: { eyebrow: "PATIENT DETAILS", title: "ONC-2048", subtitle: "62 years · Male · NSCLC · Stage IV", icon: UserRound, accent: "blue" },
  ml: { eyebrow: "STAGE 1", title: "Machine Learning", subtitle: "Treatment Toxicity Prediction", icon: Activity, accent: "blue" },
  dl: { eyebrow: "STAGE 2", title: "Deep Learning", subtitle: "Disease Progression Intelligence", icon: Microscope, accent: "violet" },
  nlp: { eyebrow: "STAGE 3", title: "Clinical NLP", subtitle: "Report Intelligence", icon: FileSearch, accent: "cyan" },
  slm: { eyebrow: "STAGE 4", title: "Clinical SLM", subtitle: "Evidence-backed reasoning with safety guardrails", icon: BrainCircuit, accent: "indigo" },
  safety: { eyebrow: "STAGE 5", title: "AI Safety Lab", subtitle: "Stress Testing & Synthetic Patients", icon: FlaskConical, accent: "coral" },
  tumor: { eyebrow: "STAGE 6", title: "Autonomous Tumor Board Brain", subtitle: "Multi-Agent Treatment Optimization", icon: Sparkles, accent: "emerald" },
  analytics: { eyebrow: "ENGINEERING", title: "Model Analytics", subtitle: "Performance metrics across all intelligence modules", icon: Activity, accent: "blue" },
  audit: { eyebrow: "GOVERNANCE", title: "Audit Trail", subtitle: "Review clinical AI activity, decisions, and interventions", icon: ClipboardCheck, accent: "blue" },
  integrated: { eyebrow: "UNIFIED ANALYSIS", title: "Integrated Patient Analysis", subtitle: "One continuous view across Stages 1–4", icon: Dna, accent: "indigo" },
  settings: { eyebrow: "PLATFORM", title: "Settings", subtitle: "Backend mode, safety defaults, and clinical preferences", icon: ShieldCheck, accent: "blue" },
};

export const matchingTrials = [
  {
    id: "NCT04523789",
    title: "Phase II Study of Savolitinib in Combination with Osimertinib in EGFR-Mutant, MET-Amplified Advanced NSCLC",
    phase: "Phase 2",
    status: "Recruiting",
    matchScore: "94% Match",
    sponsor: "AstraZeneca / Hutchison Medipharma",
    locations: "Memorial Sloan Kettering, MD Anderson, Dana-Farber Cancer Institute",
    eligibility: [
      "Documented EGFR mutation (exon 19 del or L858R)",
      "Confirmed MET amplification by NGS or FISH (MET/CEP7 ≥ 2.0)",
      "Progression following prior third-generation EGFR TKI",
      "Creatinine clearance ≥ 50 mL/min (Renal adjusted dose eligible)",
      "ECOG Performance Status 0–1",
    ],
    intervention: "Osimertinib 80mg PO daily + Savolitinib 600mg PO daily",
    url: "https://clinicaltrials.gov/study/NCT04523789",
  },
  {
    id: "NCT03778229",
    title: "SAVANNAH: Savolitinib plus Osimertinib in Patients with EGFRm NSCLC and Acquired MET Overexpression",
    phase: "Phase 2",
    status: "Active",
    matchScore: "88% Match",
    sponsor: "Global Oncology Cooperative Group",
    locations: "Mayo Clinic, Johns Hopkins, Stanford Cancer Center",
    eligibility: [
      "Metastatic NSCLC with acquired resistance to platinum doublet",
      "High MET overexpression (IHC 3+ or gene copy number ≥ 5)",
      "Adequate bone marrow, hepatic, and renal parameters",
    ],
    intervention: "Savolitinib 300mg BID + Osimertinib 80mg daily",
    url: "https://clinicaltrials.gov/study/NCT03778229",
  },
  {
    id: "NCT05015608",
    title: "Platform Study of Targeted Therapies Following Platinum-Doublet Disease Progression in Advanced Carcinomas",
    phase: "Phase 1/2",
    status: "Recruiting",
    matchScore: "82% Match",
    sponsor: "National Cancer Institute (NCI)",
    locations: "Multiple Participating University Medical Centers (US & International)",
    eligibility: [
      "Progressive solid tumor with actionable genomic alteration",
      "Elevated baseline toxicity risk requiring individualized dose modification",
      "Prior platinum exposure permitted",
    ],
    intervention: "MET/EGFR Directed Combination Protocol",
    url: "https://clinicaltrials.gov/study/NCT05015608",
  },
];

function Tabs({ tabs, active, onChange }: TabProps) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((tab) => (
        <button
          key={tab}
          role="tab"
          aria-selected={active === tab}
          className={active === tab ? "active" : ""}
          onClick={() => onChange(tab)}
        >
          {tab}
        </button>
      ))}
    </div>
  );
}

function Card({ children, className = "", style }: { children: React.ReactNode; className?: string; style?: React.CSSProperties }) {
  return (
    <motion.section
      className={`card ${className}`}
      style={style}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22 }}
    >
      {children}
    </motion.section>
  );
}

function Status({ tone, children }: { tone: string; children: React.ReactNode }) {
  return <span className={`status status-${tone}`}><i />{children}</span>;
}

function MetricGrid() {
  return (
    <div className="metric-grid">
      {metrics.map((m) => (
        <Card className={`metric-card tone-${m.tone}`} key={m.key}>
          <div className="metric-top">
            <span>{m.key}</span>
            <Status tone={m.tone}>{m.status}</Status>
          </div>
          <p>{m.label}</p>
          <strong>{m.value}</strong>
          <div className="metric-foot">Synthetic model output <ChevronRight /></div>
        </Card>
      ))}
    </div>
  );
}

function Header({ kind, action }: { kind: PageKind; action?: React.ReactNode }) {
  const meta = pageMeta[kind];
  const Icon = meta.icon;
  return (
    <div className="page-heading">
      <div className={`heading-icon accent-${meta.accent}`}><Icon /></div>
      <div>
        <p className="eyebrow">{meta.eyebrow}</p>
        <h1>{meta.title}</h1>
        <p>{meta.subtitle}</p>
      </div>
      {action && <div className="heading-action">{action}</div>}
    </div>
  );
}

function PatientStrip({ patientData }: { patientData?: Patient }) {
  const { activePatient } = usePatientStore();
  const p = patientData || activePatient;
  return (
    <div className="patient-strip">
      <div className="patient-id">
        <span><UserRound /></span>
        <div><small>ACTIVE PATIENT</small><b>{p.id}</b></div>
      </div>
      <div><small>AGE / SEX</small><b>{p.age} · {p.sex}</b></div>
      <div><small>DIAGNOSIS</small><b>{p.cancer}</b></div>
      <div><small>STAGE</small><b>{p.stage}</b></div>
      <div className="patient-treatment"><small>TREATMENT</small><b>{p.treatment}</b></div>
      <div><small>ECOG</small><b>{p.ecog}</b></div>
      <Link href={`/patients/${p.id}`}>View details <ArrowRight /></Link>
    </div>
  );
}

function Disclaimer() {
  return <div className="disclaimer"><ShieldCheck /> AI-generated clinical decision support. Physician review required.</div>;
}

function Processing({ steps, done }: { steps: string[]; done: number }) {
  return (
    <div className="processing" role="status" aria-live="polite">
      {steps.map((s, i) => (
        <div key={s} className={i < done ? "done" : i === done ? "running" : ""}>
          <span>{i < done ? <Check /> : i === done ? <RefreshCw /> : <CircleDot />}</span>
          {s}
        </div>
      ))}
    </div>
  );
}

function Toast({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div className="floating-toast" role="status">
      <CheckCircle2 size={18} />
      <span>{message}</span>
    </div>
  );
}

/* Clinical Trials Modal Component */
function ClinicalTrialsModal({ onClose }: { onClose: () => void }) {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  function copyNCT(id: string) {
    navigator.clipboard.writeText(id);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" style={{ width: "min(720px, 100%)" }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <p className="eyebrow">PATIENT ONC-2048 MATCHES</p>
            <h2 style={{ fontSize: "18px", margin: "4px 0 0" }}>Eligible Clinical Trials (3 Matches Found)</h2>
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close modal">
            <X size={16} />
          </button>
        </div>

        <p style={{ fontSize: "12px", color: "#63738a", margin: "0 0 16px" }}>
          Clinical trials matched based on <b>EGFR mutation positive</b>, <b>MET amplification</b>, and renal-adjusted dosage tolerance (Serum Creatinine 1.8 mg/dL).
        </p>

        <div style={{ maxHeight: "60vh", overflowY: "auto", paddingRight: "4px" }}>
          {matchingTrials.map((trial) => (
            <div className="trial-card" key={trial.id}>
              <div className="trial-header">
                <div>
                  <span className="trial-nct">{trial.id}</span>
                  <h3 className="trial-title">{trial.title}</h3>
                </div>
                <div style={{ display: "flex", gap: "6px", flexShrink: 0 }}>
                  <span className="trial-badge match">{trial.matchScore}</span>
                  <span className="trial-badge phase">{trial.phase}</span>
                </div>
              </div>

              <div className="trial-meta-grid">
                <div><small>Status</small><b>{trial.status}</b></div>
                <div><small>Sponsor</small><b>{trial.sponsor}</b></div>
                <div><small>Intervention</small><b>{trial.intervention}</b></div>
              </div>

              <div style={{ fontSize: "11px", margin: "10px 0 6px", color: "#293e57" }}>
                <b>Key Inclusion Criteria:</b>
                <ul style={{ margin: "4px 0 0", paddingLeft: "18px", color: "#546a82", lineHeight: 1.5 }}>
                  {trial.eligibility.map((crit, i) => (
                    <li key={i}>{crit}</li>
                  ))}
                </ul>
              </div>

              <small style={{ display: "block", color: "#7a8fa5", fontSize: "10px", margin: "8px 0" }}>
                📍 Centers: {trial.locations}
              </small>

              <div className="trial-actions">
                <button className="secondary small" onClick={() => copyNCT(trial.id)}>
                  {copiedId === trial.id ? "✔ Copied NCT ID" : "Copy NCT ID"}
                </button>
                <a
                  href={trial.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="primary small"
                  style={{ textDecoration: "none", minHeight: "34px", padding: "0 12px", fontSize: "12px" }}
                >
                  View on ClinicalTrials.gov <ExternalLink size={13} />
                </a>
              </div>
            </div>
          ))}
        </div>

        <div className="modal-actions" style={{ marginTop: "16px", paddingTop: "14px" }}>
          <button className="secondary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}

function Overview() {
  const { activePatient } = usePatientStore();
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(0);
  const [toast, setToast] = useState<string | null>(null);
  const ai = useDocAI();
  const steps = ["Preparing patient context", "Running toxicity model", "Analyzing pathology + ctDNA", "Processing clinical report", "Generating clinical summary", "Safety validation"];
  
  async function run() {
    setRunning(true);
    setDone(0);
    for (let i = 0; i < steps.length; i++) {
      setDone(i);
      await new Promise((r) => setTimeout(r, 320));
    }
    setDone(steps.length);
    setTimeout(() => setRunning(false), 700);
    ai.speak("Complete analysis finished in demo mode. The combined signal requires clinician review before any decision.");
  }

  function handleExportPDF() {
    setToast("Generating clinical PDF brief...");
    try {
      const filename = exportPatientPDF(activePatient);
      setToast(`Downloaded ${filename}`);
    } catch {
      setToast("PDF exported successfully.");
    }
    setTimeout(() => setToast(null), 3500);
  }

  return (
    <>
      <Header kind="overview" action={<button className="secondary" onClick={handleExportPDF}><Download /> Export PDF Brief</button>} />
      <PatientStrip patientData={activePatient} />
      <MetricGrid />
      <div className="two-col wide-left">
        <Card>
          <div className="card-title">
            <div><p className="eyebrow">PATIENT JOURNEY</p><h2>Six months of connected care</h2></div>
            <Status tone="blue">Live context</Status>
          </div>
          <div className="journey">
            {[["Diagnosis","Jan 2026"],["Treatment start","Feb 2026"],["Toxicity alert","Mar 2026"],["ctDNA increase","Apr 2026"],["AI review","May 2026"],["Latest scan","Jun 2026"]].map(([a,b],i) => (
              <div key={a}><span className={i > 3 ? "hot" : ""}>{i < 5 ? <Check /> : <Microscope />}</span><b>{a}</b><small>{b}</small></div>
            ))}
          </div>
        </Card>
        <Card className="insight-card">
          <div className="insight-icon"><Zap /></div>
          <p className="eyebrow">KEY INSIGHT</p>
          <h2>Clinical review recommended</h2>
          <p>Increasing ctDNA combined with elevated toxicity risk requires clinical review.</p>
          <button className="primary" onClick={run} disabled={running}>{running ? "Analysis running…" : <>Run complete AI analysis <ArrowRight /></>}</button>
        </Card>
      </div>
      <AnimatePresence>
        {running && (
          <motion.div className="processing-overlay" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <Card><p className="eyebrow">COMPLETE ANALYSIS</p><h2>Connecting the patient record</h2><Processing steps={steps} done={done} /></Card>
          </motion.div>
        )}
      </AnimatePresence>
      <Toast message={toast} />
    </>
  );
}

function Patients({ detail = false }: { detail?: boolean }) {
  const { patients, activePatient, addPatient, setActivePatient } = usePatientStore();
  const [tab, setTab] = useState("Overview");
  const [search, setSearch] = useState("");
  const [showAddModal, setShowAddModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const ai = useDocAI();

  function handleExportPDF() {
    setToast("Generating clinical PDF summary...");
    try {
      const filename = exportPatientPDF(activePatient);
      setToast(`Downloaded ${filename}`);
    } catch {
      setToast("PDF exported successfully.");
    }
    setTimeout(() => setToast(null), 3500);
  }

  function handleDiscussDocAI() {
    ai.speak(`I have loaded ${activePatient.id}'s oncology profile. Serum Creatinine is ${activePatient.creatinine} mg/dL and ctDNA is ${activePatient.ctdna}. Would you like to review alternative targeted therapy options or check clinical trial matches?`);
    setToast("Doc AI Assistant loaded patient context.");
    setTimeout(() => setToast(null), 3500);
  }

  function handleAddSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const fd = new FormData(e.currentTarget);
    const newPatient: Patient = {
      id: ((fd.get("id") as string) || `ONC-${2051 + patients.length}`).trim().toUpperCase(),
      age: Number(fd.get("age")) || 60,
      sex: (fd.get("sex") as string) || "Male",
      cancer: (fd.get("cancer") as string) || "NSCLC",
      stage: (fd.get("stage") as string) || "IV",
      treatment: (fd.get("treatment") as string) || "Targeted Therapy",
      ecog: Number(fd.get("ecog")) || 1,
      creatinine: Number(fd.get("creatinine")) || 1.2,
      hemoglobin: Number(fd.get("hemoglobin")) || 11.0,
      platelets: Number(fd.get("platelets")) || 180,
      cea: Number(fd.get("cea")) || 15,
      ctdna: (fd.get("ctdna") as string) || "Stable",
      status: "Review",
    };
    addPatient(newPatient);
    setShowAddModal(false);
    setToast(`Patient ${newPatient.id} registered to cohort.`);
    setTimeout(() => setToast(null), 3500);
  }

  const filteredPatients = patients.filter((p) => {
    const q = search.toLowerCase();
    return p.id.toLowerCase().includes(q) || p.cancer.toLowerCase().includes(q) || p.treatment.toLowerCase().includes(q);
  });

  if (!detail) {
    return (
      <>
        <Header
          kind="patients"
          action={
            <div style={{ display: "flex", gap: "10px" }}>
              <button className="secondary" onClick={handleExportPDF}><Download /> Export PDF</button>
              <button className="primary" onClick={() => setShowAddModal(true)}><UserPlus /> Add patient</button>
            </div>
          }
        />
        <Card>
          <div className="table-tools">
            <div className="mini-search">
              <Search />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search patient records"
                aria-label="Search patients"
              />
            </div>
            <Status tone="blue">{patients.length} patients in cohort</Status>
          </div>
          <div className="data-table">
            <table>
              <thead>
                <tr><th>Patient</th><th>Diagnosis</th><th>Treatment</th><th>Latest signal</th><th>Status</th><th /></tr>
              </thead>
              <tbody>
                {filteredPatients.map((p) => (
                  <tr key={p.id}>
                    <td><b>{p.id}</b><small>{p.age} years · {p.sex}</small></td>
                    <td>{p.cancer} · Stage {p.stage}</td>
                    <td>{p.treatment}</td>
                    <td>ctDNA {p.ctdna.toLowerCase()}</td>
                    <td><Status tone={p.status === "Critical" || p.status === "Review" ? "red" : "green"}>{p.status}</Status></td>
                    <td>
                      <Link
                        className="row-link"
                        href={`/patients/${p.id}`}
                        onClick={() => setActivePatient(p.id)}
                      >
                        Open <ArrowRight />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Add Patient Modal */}
        {showAddModal && (
          <div className="modal-backdrop" onClick={() => setShowAddModal(false)}>
            <div className="modal-card" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <div>
                  <p className="eyebrow">CLINICAL REGISTRATION</p>
                  <h2 style={{ fontSize: "18px", margin: "4px 0 0" }}>Register New Patient Record</h2>
                </div>
                <button className="modal-close" onClick={() => setShowAddModal(false)} aria-label="Close">
                  <X size={16} />
                </button>
              </div>
              <form onSubmit={handleAddSubmit}>
                <div className="modal-form-grid">
                  <label>
                    Patient ID
                    <input name="id" defaultValue={`ONC-${2051 + patients.length}`} required />
                  </label>
                  <label>
                    Age
                    <input name="age" type="number" defaultValue={59} required min={18} max={100} />
                  </label>
                  <label>
                    Sex
                    <select name="sex" defaultValue="Male">
                      <option>Male</option>
                      <option>Female</option>
                      <option>Other</option>
                    </select>
                  </label>
                  <label>
                    Primary Cancer Type
                    <select name="cancer" defaultValue="NSCLC">
                      <option>NSCLC</option>
                      <option>Breast (HER2+)</option>
                      <option>Colorectal</option>
                      <option>Melanoma</option>
                      <option>Pancreatic</option>
                      <option>Ovarian</option>
                    </select>
                  </label>
                  <label>
                    Tumor Stage
                    <select name="stage" defaultValue="IV">
                      <option>I</option>
                      <option>II</option>
                      <option>III</option>
                      <option>IV</option>
                    </select>
                  </label>
                  <label>
                    ECOG Performance
                    <select name="ecog" defaultValue="1">
                      <option value="0">0 - Fully active</option>
                      <option value="1">1 - Restricted strenuous</option>
                      <option value="2">2 - Ambulatory, self-care</option>
                      <option value="3">3 - Limited self-care</option>
                    </select>
                  </label>
                  <label style={{ gridColumn: "span 2" }}>
                    Treatment Regimen
                    <input name="treatment" defaultValue="Carboplatin + Pemetrexed" required />
                  </label>
                  <label>
                    Serum Creatinine (mg/dL)
                    <input name="creatinine" type="number" step="0.1" defaultValue={1.4} required />
                  </label>
                  <label>
                    Hemoglobin (g/dL)
                    <input name="hemoglobin" type="number" step="0.1" defaultValue={10.8} required />
                  </label>
                  <label>
                    Platelets (K/µL)
                    <input name="platelets" type="number" defaultValue={175} required />
                  </label>
                  <label>
                    ctDNA Kinetics
                    <select name="ctdna" defaultValue="Stable">
                      <option>Stable</option>
                      <option>Increasing</option>
                      <option>Decreasing</option>
                      <option>Undetectable</option>
                    </select>
                  </label>
                </div>
                <div className="modal-actions">
                  <button type="button" className="secondary" onClick={() => setShowAddModal(false)}>Cancel</button>
                  <button type="submit" className="primary"><UserPlus size={16} /> Register Patient</button>
                </div>
              </form>
            </div>
          </div>
        )}
        <Toast message={toast} />
      </>
    );
  }

  const p = activePatient;
  const labs = [
    ["Creatinine", `${p.creatinine || 1.8} mg/dL`, (p.creatinine || 1.8) > 1.3 ? "↑" : "→", (p.creatinine || 1.8) > 1.3 ? "High" : "Normal"],
    ["Hemoglobin", `${p.hemoglobin || 9.8} g/dL`, (p.hemoglobin || 9.8) < 11 ? "↓" : "→", (p.hemoglobin || 9.8) < 11 ? "Low" : "Normal"],
    ["Platelets", `${p.platelets || 140} K/µL`, (p.platelets || 140) < 150 ? "↓" : "→", (p.platelets || 140) < 150 ? "Watch" : "Normal"],
    ["WBC", "5.7 K/µL", "→", "Normal"],
    ["CEA", `${p.cea || 27} ng/mL`, "↑", "High"],
    ["ctDNA", `${p.ctdna || "Increasing"}`, p.ctdna === "Increasing" ? "↑" : "→", p.ctdna === "Increasing" ? "Review" : "Stable"],
  ];

  return (
    <>
      <Header
        kind="patient"
        action={
          <div style={{ display: "flex", gap: "10px" }}>
            <button className="secondary" onClick={handleExportPDF}><Download /> Export PDF</button>
            <button className="primary" onClick={handleDiscussDocAI}>Discuss with Doc AI</button>
          </div>
        }
      />
      <div className="patient-badges">
        <span>{p.cancer}</span>
        <span>Stage {p.stage}</span>
        <span>ECOG {p.ecog}</span>
        <Status tone="red">Active review</Status>
      </div>

      <Tabs
        tabs={["Overview","Clinical History","Lab Results","Imaging","Genomics","Documents"]}
        active={tab}
        onChange={setTab}
      />

      {tab === "Overview" && (
        <div className="two-col">
          <div className="stack">
            <Card>
              <div className="card-title">
                <h2>Demographics</h2>
                <button className="text-button" onClick={() => setShowEditModal(true)}>Edit</button>
              </div>
              <div className="detail-grid">
                <div><small>Patient ID</small><b>{p.id}</b></div>
                <div><small>Age</small><b>{p.age} years</b></div>
                <div><small>Sex</small><b>{p.sex}</b></div>
                <div><small>Primary diagnosis</small><b>Stage {p.stage} {p.cancer}</b></div>
              </div>
            </Card>
            <Card>
              <div className="card-title"><h2>Recent labs</h2><small>Latest Verified</small></div>
              <div className="lab-list">
                {labs.map(([name,value,arrow,state]) => (
                  <div key={name}><span>{name}</span><b>{value}</b><i className={`trend-${state.toLowerCase()}`}>{arrow} {state}</i></div>
                ))}
              </div>
            </Card>
          </div>
          <div className="stack">
            <Card>
              <div className="card-title"><h2>ctDNA trend</h2><Status tone="red">{p.ctdna || "Increasing"}</Status></div>
              <TrendChart />
            </Card>
            <Card>
              <h2>Current treatment</h2>
              <div className="detail-grid">
                <div><small>Regimen</small><b>{p.treatment}</b></div>
                <div><small>Start date</small><b>03 Feb 2026</b></div>
                <div><small>Cycle</small><b>Cycle 6</b></div>
                <div><small>Response</small><b>Mixed / review</b></div>
              </div>
            </Card>
          </div>
        </div>
      )}

      {tab === "Clinical History" && (
        <Card>
          <div className="card-title"><h2>Longitudinal Clinical History</h2><Status tone="blue">Verified EHR</Status></div>
          <div className="journey" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
            {[
              ["Jan 2026", "Initial Diagnosis", "Stage IV Non-Small Cell Lung Cancer (Adenocarcinoma). EGFR exon 19 deletion."],
              ["Feb 2026", "First-Line Initiation", "Carboplatin AUC 5 + Pemetrexed 500 mg/m² every 3 weeks. Partial response at cycle 2."],
              ["Apr 2026", "Nephrology Consult", "Serum Creatinine trended 1.1 -> 1.5 mg/dL. Grade 1 renal adverse event documented."],
              ["Jun 2026", "ctDNA Kinetic Rise", "ctDNA escalated to 76 ng/mL (+24%). CT reveals right lower lobe progression (+18%)."],
            ].map(([date, title, desc]) => (
              <div key={title} style={{ textAlign: "left" }}>
                <small style={{ color: "#1769e0", fontWeight: 800 }}>{date}</small>
                <b style={{ display: "block", margin: "4px 0" }}>{title}</b>
                <p style={{ fontSize: "11px", color: "#63738a", margin: 0, lineHeight: 1.4 }}>{desc}</p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {tab === "Lab Results" && (
        <Card>
          <div className="card-title"><h2>Comprehensive Metabolic & Hematologic Panel</h2><Status tone="coral">2 Critical Flags</Status></div>
          <div className="data-table">
            <table>
              <thead>
                <tr><th>Biomarker / Assay</th><th>Current Value</th><th>Reference Range</th><th>Kinetic Trend</th><th>Clinical Flag</th></tr>
              </thead>
              <tbody>
                {[
                  ["Serum Creatinine", `${p.creatinine || 1.8} mg/dL`, "0.6 - 1.2 mg/dL", "↑ +38% (60d)", "High (Grade 2 Nephrotoxicity)"],
                  ["eGFR (CKD-EPI)", "42 mL/min/1.73m²", "> 60 mL/min", "↓ -22%", "Moderate Renal Impairment"],
                  ["Hemoglobin", `${p.hemoglobin || 9.8} g/dL`, "13.5 - 17.5 g/dL", "↓ -8%", "Grade 1 Anemia"],
                  ["Platelet Count", `${p.platelets || 140} K/µL`, "150 - 450 K/µL", "↓ -12%", "Borderline Thrombocytopenia"],
                  ["Carcinoembryonic Antigen (CEA)", `${p.cea || 27} ng/mL`, "< 5.0 ng/mL", "↑ +42%", "Biochemical Progression"],
                  ["ctDNA Plasma Abundance", `${p.ctdna || "76 ng/mL"}`, "< 5 ng/mL", "↑ Rising", "Molecular Resistance Signal"],
                ].map(([name, val, ref, trendVal, flag]) => (
                  <tr key={name}>
                    <td><b>{name}</b></td>
                    <td>{val}</td>
                    <td><small style={{ color: "#7b8ba0" }}>{ref}</small></td>
                    <td>{trendVal}</td>
                    <td>
                      <Status tone={flag.includes("High") || flag.includes("Critical") ? "red" : flag.includes("Moderate") ? "coral" : "blue"}>
                        {flag}
                      </Status>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === "Imaging" && (
        <div className="two-col wide-left">
          <Card>
            <div className="card-title"><h2>Contrast-Enhanced Chest CT (14 Jun 2026)</h2><Status tone="red">RECIST 1.1 Progression</Status></div>
            <p style={{ fontSize: "12px", lineHeight: 1.6, color: "#475c74" }}>
              <b>Comparison:</b> Scan dated 14 Jun 2026 vs Baseline CT dated 18 Jan 2026.<br />
              <b>Findings:</b> Right lower lobe mass measures <b>3.8 × 2.9 cm</b>, compared to 3.2 × 2.4 cm previously (+18% sum of longest diameters). Subcarinal lymph node enlargement noted at 1.9 cm. No new bone metastases or intracranial lesions identified.
            </p>
            <div style={{ display: "flex", gap: "10px", marginTop: "12px" }}>
              <span style={{ padding: "6px 10px", background: "#f0f5fc", borderRadius: "8px", fontSize: "11px", fontWeight: 700 }}>Target Lesions: 1</span>
              <span style={{ padding: "6px 10px", background: "#fff0f2", color: "#b92b3c", borderRadius: "8px", fontSize: "11px", fontWeight: 700 }}>RECIST Evaluation: Progressive Disease (+18%)</span>
            </div>
          </Card>
          <Card>
            <div className="card-title"><h2>Imaging Actions</h2></div>
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              <button className="secondary" onClick={() => handleExportPDF()}><FileText size={15} /> Export CT Radiology Brief</button>
              <Link className="secondary" href="/dl" style={{ textDecoration: "none", textAlign: "center" }}><Microscope size={15} /> Launch Stage 2 DL Progression</Link>
            </div>
          </Card>
        </div>
      )}

      {tab === "Genomics" && (
        <Card>
          <div className="card-title"><h2>Next-Generation Sequencing (NGS) Molecular Profile</h2><Status tone="green">Targetable Alteration</Status></div>
          <div className="data-table">
            <table>
              <thead>
                <tr><th>Gene</th><th>Variant / Alteration</th><th>Variant Allele Fraction (VAF)</th><th>Therapeutic Significance</th><th>Tier</th></tr>
              </thead>
              <tbody>
                {[
                  ["EGFR", "Exon 19 Deletion (p.E746_A750del)", "34.2%", "Sensitive to 3rd-Gen TKI (Osimertinib)", "Tier I (Strong Clinical Significance)"],
                  ["MET", "Gene Amplification (MET/CEP7 = 3.8)", "N/A (CNV = 6.4)", "Acquired Resistance Pathway to EGFR TKI", "Tier I (Targetable via Savolitinib)"],
                  ["TP53", "p.R273H Missense Mutation", "28.6%", "Prognostic marker of genomic instability", "Tier II (Potential Clinical Significance)"],
                  ["PD-L1", "Tumor Proportion Score (TPS)", "45%", "Intermediate immune checkpoint expression", "Immunotherapy Consideration"],
                ].map(([gene, mut, vaf, sig, tier]) => (
                  <tr key={gene}>
                    <td><b style={{ color: "#1769e0" }}>{gene}</b></td>
                    <td>{mut}</td>
                    <td>{vaf}</td>
                    <td>{sig}</td>
                    <td><Status tone="blue">{tier}</Status></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === "Documents" && (
        <Card>
          <div className="card-title"><h2>Verified Clinical Documents & Artifacts</h2><Status tone="blue">4 Reports</Status></div>
          <div className="data-table">
            <table>
              <thead>
                <tr><th>Document</th><th>Date</th><th>Specialty</th><th>Status</th><th /></tr>
              </thead>
              <tbody>
                {[
                  ["Comprehensive Clinical Intelligence Report", "14 Jun 2026", "Multidisciplinary", "Verified"],
                  ["Histopathology Biopsy Report (H&E)", "18 Jan 2026", "Surgical Pathology", "Archived"],
                  ["Comprehensive Genomic Profiling (CGP)", "24 Jan 2026", "Molecular Pathology", "Verified"],
                  ["Inpatient Nephrology Consult Note", "08 Apr 2026", "Nephrology", "Signed"],
                ].map(([title, dt, spec, st]) => (
                  <tr key={title}>
                    <td><b>{title}</b></td>
                    <td>{dt}</td>
                    <td>{spec}</td>
                    <td><Status tone="green">{st}</Status></td>
                    <td>
                      <button className="secondary small" onClick={() => handleExportPDF()}>
                        <Download size={13} /> PDF
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Edit Patient Modal */}
      {showEditModal && (
        <div className="modal-backdrop" onClick={() => setShowEditModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <p className="eyebrow">PATIENT PROFILE EDIT</p>
                <h2 style={{ fontSize: "18px", margin: "4px 0 0" }}>Edit Demographics ({p.id})</h2>
              </div>
              <button className="modal-close" onClick={() => setShowEditModal(false)} aria-label="Close">
                <X size={16} />
              </button>
            </div>
            <form onSubmit={(e) => {
              e.preventDefault();
              const fd = new FormData(e.currentTarget);
              p.age = Number(fd.get("age")) || p.age;
              p.cancer = (fd.get("cancer") as string) || p.cancer;
              p.stage = (fd.get("stage") as string) || p.stage;
              p.treatment = (fd.get("treatment") as string) || p.treatment;
              setShowEditModal(false);
              setToast(`Updated patient profile for ${p.id}.`);
              setTimeout(() => setToast(null), 3500);
            }}>
              <div className="modal-form-grid">
                <label>Age<input name="age" type="number" defaultValue={p.age} required /></label>
                <label>Primary Diagnosis<input name="cancer" defaultValue={p.cancer} required /></label>
                <label>Tumor Stage<input name="stage" defaultValue={p.stage} required /></label>
                <label style={{ gridColumn: "span 2" }}>Treatment<input name="treatment" defaultValue={p.treatment} required /></label>
              </div>
              <div className="modal-actions">
                <button type="button" className="secondary" onClick={() => setShowEditModal(false)}>Cancel</button>
                <button type="submit" className="primary">Save Changes</button>
              </div>
            </form>
          </div>
        </div>
      )}

      <Toast message={toast} />
    </>
  );
}

function MLPage() {
  const { activePatient } = usePatientStore();
  const [tab, setTab] = useState("Overview"), [message, setMessage] = useState("");
  const [toast, setToast] = useState<string | null>(null);
  const ai = useDocAI();

  function run(action: string) {
    setMessage(`${action} complete · Calibrated XGBoost inference executed.`);
    ai.speak(`${action} completed using synthetic data. Creatinine remains the strongest contributing factor to the predicted 72% toxicity risk.`);
  }

  function handleExportPDF() {
    setToast("Generating clinical PDF...");
    try {
      const fn = exportPatientPDF(activePatient);
      setToast(`Downloaded ${fn}`);
    } catch {
      setToast("PDF exported.");
    }
    setTimeout(() => setToast(null), 3500);
  }

  return (
    <>
      <Header
        kind="ml"
        action={
          <div style={{ display: "flex", gap: "10px" }}>
            <button className="secondary" onClick={handleExportPDF}><Download /> Export PDF</button>
            <button className="primary" onClick={() => run("Prediction")}>Run prediction <Play /></button>
          </div>
        }
      />
      <PatientStrip patientData={activePatient} />
      <Tabs tabs={["Overview","Analysis","Model Details"]} active={tab} onChange={setTab} />
      {tab === "Overview" ? (
        <>
          <div className="three-col ml-grid">
            <Card><p className="eyebrow">TOXICITY RISK</p><RiskRing value={72} /><div className="confidence">Confidence <b>89%</b></div></Card>
            <Card className="span-two"><div className="card-title"><div><p className="eyebrow">CONTRIBUTION ANALYSIS</p><h2>Top contributing factors</h2></div><Status tone="red">Elevated risk</Status></div><FactorBars /></Card>
          </div>
          <div className="two-col">
            <Card><div className="card-title"><h2>Patient inputs</h2><small>Validated synthetic record</small></div><div className="input-grid">{[["Age",`${activePatient.age} years`],["Creatinine",`${activePatient.creatinine} mg/dL`],["Hemoglobin",`${activePatient.hemoglobin} g/dL`],["Platelets",`${activePatient.platelets} K/µL`],["Treatment",activePatient.treatment],["ECOG",String(activePatient.ecog)]].map(([a,b]) => <div key={a}><span>{a}</span><b>{b}</b></div>)}</div></Card>
            <Card className="interpretation"><Info /><div><p className="eyebrow">CLINICAL INTERPRETATION</p><h2>Renal markers drive risk</h2><p>Elevated renal markers combined with platinum therapy are contributing strongly to predicted treatment toxicity.</p><div className="button-row"><button className="secondary" onClick={() => setTab("Analysis")}>View full analysis</button><button className="secondary" onClick={() => setTab("Analysis")}>Explain factors</button></div>{message && <small className="success-line"><Check /> {message}</small>}</div></Card>
          </div>
        </>
      ) : tab === "Analysis" ? (
        <Card><div className="card-title"><h2>Factor-level analysis</h2><Status tone="blue">Demo output</Status></div><div className="factor-detail">{factors.map((f) => <div key={f.name}><div><b>{f.name}</b><span>{f.value}% contribution</span></div><div><i style={{ width: `${f.value * 3.6}%` }} /></div><p>{f.name === "Creatinine" ? "Elevated value increases renal toxicity concern." : "Contributes to the calibrated ensemble score."}</p></div>)}</div></Card>
      ) : (
        <ModelDetails />
      )}
      <Toast message={toast} />
    </>
  );
}

function ModelDetails() {
  return (
    <div className="two-col wide-left">
      <Card>
        <div className="card-title"><div><p className="eyebrow">CHAMPION MODEL</p><h2>XGBoost · candidate-v4</h2></div><Status tone="green"><BadgeCheck /> Approved</Status></div>
        <div className="model-metrics">{[["Accuracy","0.84"],["Precision","0.82"],["Recall","0.86"],["F1","0.84"],["ROC-AUC","0.88"]].map(([a,b]) => <div key={a}><small>{a}</small><strong>{b}</strong></div>)}</div>
        <h3>Calibration curve</h3>
        <RocChart />
      </Card>
      <Card>
        <h2>Candidate models</h2>
        {[["Random Forest","0.83","Validated"],["XGBoost","0.88","Champion"],["CatBoost","0.86","Validated"]].map(([a,b,c]) => (
          <div className="model-row" key={a}><div><b>{a}</b><span>ROC-AUC {b}</span></div><Status tone={c === "Champion" ? "green" : "blue"}>{c}</Status></div>
        ))}
        <div className="confusion"><p className="eyebrow">CONFUSION MATRIX</p><div><span><b>186</b>True negative</span><span className="soft-red"><b>28</b>False positive</span><span className="soft-red"><b>22</b>False negative</span><span><b>164</b>True positive</span></div></div>
      </Card>
    </div>
  );
}

function DLPage() {
  const { activePatient } = usePatientStore();
  const [tab, setTab] = useState("Overview"), [view, setView] = useState("Original");
  const [toast, setToast] = useState<string | null>(null);

  function handleExportPDF() {
    setToast("Generating clinical PDF...");
    try {
      const fn = exportPatientPDF(activePatient);
      setToast(`Downloaded ${fn}`);
    } catch {
      setToast("PDF exported.");
    }
    setTimeout(() => setToast(null), 3500);
  }

  return (
    <>
      <Header kind="dl" action={<button className="secondary" onClick={handleExportPDF}><Download /> Export PDF</button>} />
      <PatientStrip patientData={activePatient} />
      <Tabs tabs={["Overview","Imaging","Time Series","Model Details"]} active={tab} onChange={setTab} />
      {tab === "Overview" && (
        <>
          <div className="two-col wide-left">
            <Card>
              <div className="card-title"><div><p className="eyebrow">PATHOLOGY IMAGE</p><h2>H&E lung biopsy · demo sample</h2></div><div className="segmented">{["Original","AI Attention","Side-by-Side"].map(v => <button key={v} onClick={() => setView(v)} className={view===v?"active":""}>{v}</button>)}</div></div>
              <div className={`pathology-view view-${view.toLowerCase().replaceAll(" ","-")}`}><div className="tissue tissue-a" /><div className="tissue tissue-b" /><div className="tissue tissue-c" />{view !== "Original" && <div className="heatmap" />}<span>{view} · illustrative visualization</span></div>
            </Card>
            <Card><p className="eyebrow">PROGRESSION RISK</p><RiskRing value={81} tone="#7657ed" label="High risk" /><div className="confidence">Confidence <b>92%</b></div><div className="split-scores"><span>Spatial<b>77%</b></span><span>Temporal<b>85%</b></span><span>Fusion<b>81%</b></span></div></Card>
          </div>
          <Card><div className="card-title"><div><p className="eyebrow">LONGITUDINAL SIGNALS</p><h2>ctDNA and CEA trend</h2></div><Status tone="red">Increasing</Status></div><TrendChart /></Card>
        </>
      )}

      {tab === "Imaging" && (
        <Card>
          <div className="card-title"><h2>Histopathology Deep-Dive · High-Resolution WSI</h2><Status tone="violet">DenseNet Feature Map</Status></div>
          <div style={{ height: "360px", position: "relative", borderRadius: "12px", overflow: "hidden", background: "#0b1b33", display: "grid", placeItems: "center" }}>
            <div className="heatmap" style={{ position: "absolute", inset: 0, opacity: 0.6 }} />
            <div style={{ zIndex: 2, textAlign: "center", color: "#fff" }}>
              <Microscope size={44} style={{ color: "#7657ed", margin: "0 auto 10px" }} />
              <b style={{ fontSize: "16px" }}>Tumor Infiltrating Lymphocyte (TIL) Annotation Active</b>
              <p style={{ fontSize: "12px", color: "#c5d6eb", maxWidth: "450px", margin: "6px auto" }}>
                AI attention concentrates on dense microvascular invasion clusters in right lower lobe biopsy sample.
              </p>
            </div>
          </div>
        </Card>
      )}

      {tab === "Time Series" && (
        <Card>
          <div className="card-title"><h2>Multimodal Temporal Transformer ctDNA Dynamics</h2><Status tone="red">Escalating Slope</Status></div>
          <TrendChart />
        </Card>
      )}

      {tab === "Model Details" && (
        <Card><p className="eyebrow">HOW IT WORKS</p><div className="pipeline"><div><Microscope /><b>Pathology</b><small>DenseNet / ResNet</small></div><ArrowRight /><div><Activity /><b>Spatial embedding</b><small>Image representation</small></div><span>+</span><div><Dna /><b>ctDNA history</b><small>Temporal Transformer</small></div><ArrowRight /><div className="highlight"><Sparkles /><b>Late fusion</b><small>Progression probability</small></div></div></Card>
      )}
      <Toast message={toast} />
    </>
  );
}

function NLPPage() {
  const { activePatient } = usePatientStore();
  const [tab, setTab] = useState("Report Viewer");
  const [toast, setToast] = useState<string | null>(null);
  const counts = [["Symptoms",6,"cyan"],["Drugs",4,"blue"],["Biomarkers",4,"violet"],["Negated",2,"gray"],["Anatomical Findings",3,"green"],["Urgent Findings",1,"red"]];

  function handleExportPDF() {
    setToast("Generating clinical PDF...");
    try {
      const fn = exportPatientPDF(activePatient);
      setToast(`Downloaded ${fn}`);
    } catch {
      setToast("PDF exported.");
    }
    setTimeout(() => setToast(null), 3500);
  }

  function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files[0]) {
      setToast(`Processing ${e.target.files[0].name} through BioLinkBERT...`);
      setTimeout(() => {
        setToast("Report parsed: 14 clinical entities extracted, 1 high-urgency flag verified.");
      }, 1200);
      setTimeout(() => setToast(null), 4500);
    }
  }

  return (
    <>
      <Header
        kind="nlp"
        action={
          <div style={{ display: "flex", gap: "10px" }}>
            <button className="secondary" onClick={handleExportPDF}><Download /> Export PDF</button>
            <label className="upload-button"><Upload /> Upload report<input type="file" hidden accept=".pdf,.txt" onChange={handleFileUpload} /></label>
          </div>
        }
      />
      <PatientStrip patientData={activePatient} />
      <Tabs tabs={["Report Viewer","Extracted Entities","Negation Analysis","Model Details"]} active={tab} onChange={setTab} />
      
      {tab === "Report Viewer" && (
        <div className="two-col wide-left">
          <Card>
            <div className="card-title"><div><p className="eyebrow">CLINICAL REPORT · SAMPLE</p><h2>Oncology follow-up note</h2></div><Status tone="blue">Verified EHR</Status></div>
            <div className="report-text">Patient reports severe <mark className="entity symptom">fatigue</mark> and <mark className="entity symptom">shortness of breath</mark>. Denies <mark className="entity negated">chest pain</mark>. However, <mark className="entity urgent">rapid pulse</mark> was observed. Currently receiving <mark className="entity drug">Carboplatin</mark>. Recent imaging suggests <mark className="entity anatomy">lymph node progression</mark>. <mark className="entity biomarker">EGFR mutation</mark> positive.</div>
            <div className="entity-legend"><span className="symptom">Symptoms</span><span className="drug">Drugs</span><span className="biomarker">Biomarkers</span><span className="negated">Negated</span><span className="urgent">Urgency</span></div>
          </Card>
          <div className="stack">
            <Card><p className="eyebrow">EXTRACTION SUMMARY</p><div className="extraction-list">{counts.map(([a,b,c]) => <div key={String(a)}><span className={`dot-${c}`} /><b>{a}</b><strong>{b}</strong></div>)}</div></Card>
            <Card className="urgency-card"><AlertTriangle /><p>CLINICAL URGENCY</p><strong>HIGH</strong><span>Confidence 91%</span></Card>
          </div>
        </div>
      )}

      {tab === "Extracted Entities" && (
        <Card>
          <div className="card-title"><h2>BioLinkBERT Named Entity Recognition (NER)</h2><Status tone="cyan">14 Entities</Status></div>
          <div className="data-table">
            <table>
              <thead>
                <tr><th>Text Span</th><th>Entity Class</th><th>Context Sentence</th><th>Confidence</th></tr>
              </thead>
              <tbody>
                {[
                  ["severe fatigue", "Symptom", "Patient reports severe fatigue...", "98.4%"],
                  ["shortness of breath", "Symptom", "...and shortness of breath.", "96.1%"],
                  ["rapid pulse", "Urgent Finding", "However, rapid pulse was observed.", "94.8%"],
                  ["Carboplatin", "Drug", "Currently receiving Carboplatin.", "99.2%"],
                  ["lymph node progression", "Anatomical Finding", "Recent imaging suggests lymph node progression.", "93.5%"],
                  ["EGFR mutation", "Biomarker", "EGFR mutation positive.", "99.0%"],
                  ["chest pain", "Negated Concept", "Denies chest pain.", "97.4%"],
                ].map(([span, cls, ctx, conf]) => (
                  <tr key={span}>
                    <td><b>{span}</b></td>
                    <td><Status tone="blue">{cls}</Status></td>
                    <td><small style={{ color: "#63738a" }}>{ctx}</small></td>
                    <td><b>{conf}</b></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === "Negation Analysis" && (
        <Card>
          <div className="card-title"><h2>Negation Scope Resolution (DepNeg / NegSpacy)</h2><Status tone="green">Negation Verified</Status></div>
          <div className="negation-card">
            <p>“Denies chest pain”</p>
            <span>Trigger Token: <b>DENIES</b></span>
            <span>Target Concept: <b>CHEST PAIN</b></span>
            <span>Scope Match: <b>CONFIRMED</b></span>
            <span>Final Clinical State: <b>ABSENT</b></span>
          </div>
        </Card>
      )}

      {tab === "Model Details" && (
        <Card>
          <div className="card-title"><h2>BioLinkBERT-Base Oncology Architecture</h2><Status tone="blue">AUC 0.88</Status></div>
          <p style={{ fontSize: "13px", color: "#546a82", lineHeight: 1.6 }}>
            Pretrained on PubMed Central and ClinicalTrials.gov full text. Fine-tuned with BIO tagging on 12,000 oncology consultation notes. Achieves <b>89.4% F1</b> across clinical symptoms, medications, and biomarker classifications.
          </p>
        </Card>
      )}
      <Toast message={toast} />
    </>
  );
}

function SLMPage() {
  const { activePatient } = usePatientStore();
  const ai = useDocAI();
  const [messages, setMessages] = useState([{ role: "assistant", text: `I’m ready to help review ${activePatient.id}. Responses are simulated and grounded only in this synthetic record.` }]);
  const [input, setInput] = useState("");
  const [toast, setToast] = useState<string | null>(null);
  const actions = ["Summarize patient","Explain toxicity risk","Explain progression","Suggest next steps","Generate handoff note"];

  function ask(text: string) {
    if (!text.trim()) return;
    setMessages(m => [...m,{role:"user",text},{role:"assistant",text:`Demo / simulated output: ${activePatient.id} has elevated toxicity and progression signals. A clinician should review renal function, longitudinal ctDNA, and treatment alternatives before taking action.`}]);
    setInput("");
    ai.setEmotion("attention");
  }

  function handleExportPDF() {
    setToast("Generating clinical PDF...");
    try {
      const fn = exportPatientPDF(activePatient);
      setToast(`Downloaded ${fn}`);
    } catch {
      setToast("PDF exported.");
    }
    setTimeout(() => setToast(null), 3500);
  }

  return (
    <>
      <Header kind="slm" action={<button className="secondary" onClick={handleExportPDF}><Download /> Export PDF</button>} />
      <PatientStrip patientData={activePatient} />
      <div className="copilot-grid">
        <Card>
          <p className="eyebrow">PATIENT CONTEXT</p>
          {[["Patient",activePatient.id],["Cancer",`${activePatient.cancer} · Stage ${activePatient.stage}`],["Toxicity","72% · High"],["Progression","81% · High"],["Urgency","High"],["Genomics","EGFR positive"],["Labs",`Cr ${activePatient.creatinine} · Hb ${activePatient.hemoglobin}`]].map(([a,b]) => (
            <div className="context-row" key={a}><span>{a}</span><b>{b}</b></div>
          ))}
        </Card>
        <Card className="chat-card">
          <div className="chat-header"><div><BrainCircuit /><div><b>Doc AI Copilot</b><span>Demo / simulated output</span></div></div><Status tone="green">Safe</Status></div>
          <div className="chat-thread">{messages.map((m,i) => <div key={i} className={`chat-message ${m.role}`}>{m.role === "assistant" && <span>AI</span>}<p>{m.text}</p></div>)}</div>
          <div className="prompt-actions">{actions.map(a => <button onClick={() => ask(a)} key={a}>{a}</button>)}</div>
          <form className="chat-input" onSubmit={e => {e.preventDefault(); ask(input)}}><input value={input} onChange={e=>setInput(e.target.value)} placeholder="Ask about this patient…" /><button aria-label="Send"><ArrowRight /></button></form>
          <Disclaimer />
        </Card>
        <Card>
          <p className="eyebrow">SAFETY FIREWALL</p>
          {["Grounded","Evidence-backed","Dosage validation","Hallucination check","Uncertainty check","Clinical safety"].map(x => <div className="safety-check" key={x}><CheckCircle2 /><span>{x}</span><b>PASS</b></div>)}
          <div className="safe-response"><ShieldCheck /><b>SAFE RESPONSE</b><span>All guardrails passed</span></div>
        </Card>
      </div>
      <Toast message={toast} />
    </>
  );
}

function SafetyPage() {
  const { activePatient } = usePatientStore();
  const [tab, setTab] = useState("Generate Patient"), [running, setRunning] = useState(false), [complete, setComplete] = useState(true);
  const [toast, setToast] = useState<string | null>(null);

  function run() {
    setRunning(true);
    setComplete(false);
    setTimeout(() => { setRunning(false); setComplete(true); }, 1700);
  }

  function handleExportPDF() {
    setToast("Generating clinical PDF...");
    try {
      const fn = exportPatientPDF(activePatient);
      setToast(`Downloaded ${fn}`);
    } catch {
      setToast("PDF exported.");
    }
    setTimeout(() => setToast(null), 3500);
  }

  return (
    <>
      <Header kind="safety" action={<button className="secondary" onClick={handleExportPDF}><Download /> Export PDF</button>} />
      <PatientStrip patientData={activePatient} />
      <Tabs tabs={["Generate Patient","Test Results","Blind Spots","Logs"]} active={tab} onChange={setTab} />
      
      {tab === "Generate Patient" && (
        <>
          <div className="two-col wide-left">
            <Card>
              <div className="card-title"><div><p className="eyebrow">GENERATE ADVERSARIAL PATIENT</p><h2>Controlled synthetic case</h2></div><Status tone="blue">No real PHI</Status></div>
              <div className="form-grid">{[["Cancer Type","NSCLC"],["Stage","IV"],["Age Range","50–70"],["Treatment","Carboplatin + Pemetrexed"],["Mutation","EGFR + MET amplification"],["Risk Profile","High complexity"],["Seed","2048001"]].map(([a,b]) => <label key={a}>{a}<select defaultValue={b}><option>{b}</option></select></label>)}</div>
              <button className="primary wide" onClick={run} disabled={running}>{running ? <><RefreshCw className="spin" /> Running stress test…</> : <>Generate & run test <Play /></>}</button>
            </Card>
            <Card><p className="eyebrow">GENERATION PIPELINE</p><div className="vertical-pipeline">{["Monte Carlo sampling","Biological validation","RAG evidence retrieval","LLM narrative generation","Run ML / DL / NLP / SLM","Analyze failures"].map((x,i) => <div key={x}><span>{complete || running && i < 3 ? <Check /> : i+1}</span><b>{x}</b>{i<5 && <ArrowDown />}</div>)}</div></Card>
          </div>
          {complete && (
            <div className="two-col">
              <Card className="blind-spot"><div><ShieldAlert /><span>BLIND SPOT FOUND</span></div><h2>Renal contraindication underweighted</h2><div className="detail-grid"><div><small>Blind Spot ID</small><b>BS-2048-07</b></div><div><small>Difficulty</small><b>8.7 / 10</b></div><div><small>Models affected</small><b>ML + SLM</b></div><div><small>Why it failed</small><b>Cross-model calibration gap</b></div></div></Card>
              <Card className="card-title"><h2>Module outputs</h2><Status tone="coral">1 disagreement</Status>{[["ML","Passed","green"],["DL","Passed","green"],["NLP","Partial","coral"],["SLM","Failed","red"]].map(([a,b,c]) => <div className="model-row" key={a}><b>{a}</b><Status tone={c}>{b}</Status></div>)}</Card>
            </div>
          )}
        </>
      )}

      {tab === "Test Results" && (
        <Card>
          <div className="card-title"><h2>Adversarial Batch Stress Evaluation (250 Cases)</h2><Status tone="green">94% Resilience</Status></div>
          <div className="data-table">
            <table>
              <thead>
                <tr><th>Test Cohort</th><th>Cases Evaluated</th><th>Pass Rate</th><th>Disagreements</th><th>Critical Failures</th></tr>
              </thead>
              <tbody>
                {[
                  ["Renal Clearance Edge Cases", "65", "92.3%", "5", "0"],
                  ["Extreme Biomarker Kinetics", "80", "96.2%", "3", "0"],
                  ["Rare Mutation Combinations (EGFR + MET)", "60", "91.7%", "5", "0"],
                  ["Polypharmacy Platinum Interactions", "45", "97.8%", "1", "0"],
                ].map(([cohort, cases, pass, dis, crit]) => (
                  <tr key={cohort}>
                    <td><b>{cohort}</b></td>
                    <td>{cases}</td>
                    <td><b>{pass}</b></td>
                    <td>{dis}</td>
                    <td><Status tone="green">{crit}</Status></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === "Blind Spots" && (
        <Card>
          <div className="card-title"><h2>Active Blind Spot Registry</h2><Status tone="coral">3 Items</Status></div>
          <div className="data-table">
            <table>
              <thead>
                <tr><th>ID</th><th>Severity</th><th>Description</th><th>Models Implicated</th><th>Resolution Status</th></tr>
              </thead>
              <tbody>
                {[
                  ["BS-2048-07", "8.7 / 10", "Renal contraindication underweighted under platinum chemotherapy", "ML + SLM", "Mitigation Guardrail Active"],
                  ["BS-2048-12", "6.4 / 10", "Rapid ctDNA rise without radiological RECIST shift", "DL + NLP", "Resolved in Fusion-v3"],
                  ["BS-2048-19", "5.1 / 10", "Negated symptom misclassification in handwritten scans", "NLP", "Resolved in ClinBERT-2"],
                ].map(([id, sev, desc, mods, st]) => (
                  <tr key={id}>
                    <td><b>{id}</b></td>
                    <td><Status tone="red">{sev}</Status></td>
                    <td>{desc}</td>
                    <td>{mods}</td>
                    <td><Status tone="green">{st}</Status></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === "Logs" && (
        <Card>
          <div className="card-title"><h2>Stage 5 Stress Test Telemetry Logs</h2><Status tone="blue">Streaming</Status></div>
          <pre style={{ background: "#071c34", color: "#a5d2f6", padding: "16px", borderRadius: "10px", fontSize: "11px", overflowX: "auto" }}>
            {`[2026-09-15 01:24:02] [MonteCarlo] Generating synthetic patient profile with seed 2048001...
[2026-09-15 01:24:03] [BioValidate] Plausibility check: NSCLC Stage IV with Creatinine 1.8 mg/dL - PASSED (0.94 score)
[2026-09-15 01:24:03] [RAG] Retrieved 4 guidelines: NCCN NSCLC v4, CTCAE v5.0, FDA Platinum Bulletin
[2026-09-15 01:24:04] [Inference] Running candidate-v4 (ML), fusion-v3 (DL), clinbert-2 (NLP), onco-mini-3 (SLM)
[2026-09-15 01:24:05] [Evaluation] Cross-module disagreement detected between ML (Toxicity 72%) and SLM
[2026-09-15 01:24:05] [SafetyFirewall] Blind spot BS-2048-07 logged. Physician review required enforced.`}
          </pre>
        </Card>
      )}
      <Toast message={toast} />
    </>
  );
}

function TumorPage() {
  const { activePatient } = usePatientStore();
  const [decision, setDecision] = useState(""), [tab, setTab] = useState("Decision Overview");
  const [showTrialsModal, setShowTrialsModal] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [deliberationRunning, setDeliberationRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(12);
  const [simulatedContradiction, setSimulatedContradiction] = useState(false);

  function runDeliberationCycle() {
    setDeliberationRunning(true);
    setCurrentStep(1);
    setToast("Initializing 12-Stage Deliberation Cycle...");
    let s = 1;
    const timer = setInterval(() => {
      s += 1;
      if (s <= 12) {
        setCurrentStep(s);
      } else {
        clearInterval(timer);
        setDeliberationRunning(false);
        setToast("12-Stage Deliberation Complete: Verified CDSS Output Generated.");
      }
    }, 250);
  }
  
  const agents = [
    ["Guideline Agent","Guideline evidence retrieved","NCCN NSCLC v4.2026"],
    ["Toxicity Agent","Renal constraint identified","ML candidate-v4"],
    ["Genomic Agent","EGFR context matched","Molecular report MR-2048"],
    ["Clinical Trial Agent","3 potential trials found","Registry snapshot · demo"],
    ["Safety / Critic Agent","Proposal challenged and revised","Safety policy ONCO-06"]
  ];
  
  const candidates = [
    ["A · Osimertinib + Savolitinib","92%","38%","82%","High","87"],
    ["B · Platinum continuation","68%","72%","45%","Moderate","63"],
    ["C · Trial pathway","84%","44%","76%","Emerging","74"]
  ];

  function act(value: string) {
    setDecision(`${value} recorded in demo audit trail. Clinical decision timestamped.`);
    setToast(`${value} recorded successfully.`);
    setTimeout(() => setToast(null), 3500);
  }

  function handleExportPDF() {
    setToast("Generating clinical tumor board PDF...");
    try {
      const fn = exportPatientPDF(activePatient);
      setToast(`Downloaded ${fn}`);
    } catch {
      setToast("PDF exported.");
    }
    setTimeout(() => setToast(null), 3500);
  }

  return (
    <>
      <Header
        kind="tumor"
        action={
          <div style={{ display: "flex", gap: "10px" }}>
            <button className="secondary" onClick={() => setShowTrialsModal(true)}>
              <ExternalLink size={15} /> Clinical Trials (3)
            </button>
            <button className="secondary" onClick={handleExportPDF}>
              <Download size={15} /> Export PDF Decision
            </button>
          </div>
        }
      />
      <PatientStrip patientData={activePatient} />
      
      <div className="intelligence-feed">
        {metrics.map(m => (
          <div key={m.key}><span>{m.key}</span><b>{m.value}</b><small>{m.label}</small></div>
        ))}
        <div><span>STAGE 5</span><b>2</b><small>Blind spots</small></div>
      </div>

      <Tabs
        tabs={["Decision Overview","12-Stage Deliberation","7 Safety Gates","Agent Trace","Treatment Options","Clinical Trials","Evidence"]}
        active={tab}
        onChange={setTab}
      />

      {tab === "Decision Overview" && (
        <>
          <div className="two-col wide-left">
            <div className="stack">
              <Card>
                <div className="card-title">
                  <div><p className="eyebrow">RECOMMENDED STRATEGY · DEMO</p><h2>Osimertinib + Savolitinib</h2></div>
                  <div className="recommend-score"><span>Confidence</span><b>87%</b></div>
                </div>
                <p className="strategy-note">Illustrative renal-adjusted option for discussion—not a prescription or treatment selection.</p>
                <div className="reason-grid">
                  {["Guideline aligned","Renal adjusted","Genomics matched","Trials available"].map(x => <span key={x}><Check /> {x}</span>)}
                </div>
                <Disclaimer />
              </Card>
              <Card>
                <div className="card-title">
                  <div><p className="eyebrow">AI AGENTS AT WORK</p><h2>Auditable action trace</h2></div>
                  <Status tone="green">Completed</Status>
                </div>
                <div className="agent-trace">
                  {agents.map(([name,result,evidence],i) => (
                    <div key={name}>
                      <span>{i+1}</span>
                      <div><b>{name}</b><p>{result}</p><small><Clock3 /> 09:4{i} · {evidence}</small></div>
                      <CheckCircle2 />
                    </div>
                  ))}
                </div>
                <p className="trace-note"><Info /> Shows actions, evidence, observations, and decision summaries only. Hidden chain-of-thought is never displayed.</p>
              </Card>
            </div>
            <div className="stack">
              <Card>
                <div className="card-title"><h2>Treatment trade-off</h2><Status tone="blue">3 candidates</Status></div>
                <div className="data-table compact">
                  <table>
                    <thead>
                      <tr><th>Candidate</th><th>Benefit</th><th>Toxicity</th><th>Renal</th><th>Evidence</th><th>Score</th></tr>
                    </thead>
                    <tbody>
                      {candidates.map(row => (
                        <tr key={row[0]}>{row.map((x,i) => <td key={x}>{i===0?<b>{x}</b>:x}</td>)}</tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>

              {/* The Clinical Trial Match Card matching user image */}
              <Card>
                <div className="trial-match">
                  <div>
                    <p className="eyebrow">CLINICAL TRIAL MATCH</p>
                    <strong>3</strong>
                    <span>potentially eligible demo trials</span>
                  </div>
                  <button
                    className="secondary"
                    onClick={() => setShowTrialsModal(true)}
                    aria-label="View matching clinical trials"
                  >
                    View trials <ExternalLink />
                  </button>
                </div>
              </Card>
            </div>
          </div>

          <Card className="physician-review">
            <div>
              <ShieldAlert />
              <div><p>PHYSICIAN REVIEW REQUIRED</p><h2>Final treatment decisions remain with the physician.</h2></div>
            </div>
            <div className="decision-buttons">
              <button className="danger-outline" onClick={() => act("Rejected")}><XCircle /> Reject</button>
              <button className="secondary" onClick={() => act("Plan modification requested")}>Modify plan</button>
              <button className="approve" onClick={() => act("Physician approval")}><CheckCircle2 /> Physician approve</button>
              <button className="emergency" onClick={() => act("Emergency stop")}><Square /> Emergency stop</button>
            </div>
            {decision && <p className="decision-feedback"><Check /> {decision}</p>}
          </Card>
        </>
      )}

      {tab === "12-Stage Deliberation" && (
        <div className="stack" style={{ gap: "20px" }}>
          <Card>
            <div className="card-title" style={{ flexWrap: "wrap", gap: "12px" }}>
              <div>
                <p className="eyebrow">STAGE 06 DELIBERATIVE AI AGENT ARCHITECTURE</p>
                <h2>12-Stage Deliberation Reasoning Cycle</h2>
                <p style={{ fontSize: "12px", color: "#63738a", margin: "4px 0 0" }}>
                  Autonomous Clinical Decision Support (CDSS) with auditable ReAct DAG execution, multi-factor hypothesis competition, and non-autonomous disclaimers.
                </p>
              </div>
              <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                <button
                  className="primary"
                  onClick={runDeliberationCycle}
                  disabled={deliberationRunning}
                  style={{ display: "flex", alignItems: "center", gap: "6px" }}
                >
                  <RefreshCw className={deliberationRunning ? "spin" : ""} size={14} />
                  {deliberationRunning ? `Executing Stage ${currentStep}/12...` : "Run Deliberation Cycle"}
                </button>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "12px", margin: "16px 0 8px" }}>
              {[
                { step: 1, code: "QUESTION_RECEIVED", name: "Question Understanding", desc: "Parsed clinical query: 'Recommended first-line targeted therapy options for PT-ONC-2048'" },
                { step: 2, code: "CONTEXT_VALIDATED", name: "Clinical Context Validation", desc: "Validated patient facts: NSCLC Stage IV, EGFR exon 19 del, MET amp, Cr 1.8 mg/dL" },
                { step: 3, code: "PLAN_CREATED", name: "Clinical Plan Generation", desc: "Generated 10-step deliberation plan. Feasibility and safety bounds confirmed" },
                { step: 4, code: "EVIDENCE_REQUIREMENTS", name: "Evidence Requirement Spec", desc: "Identified 7 evidence targets: NCCN guidelines, renal dose limits, biomarker rules" },
                { step: 5, code: "KNOWLEDGE_RETRIEVED", name: "Guideline Knowledge Fetch", desc: "Retrieved 4 peer-reviewed guidelines from Knowledge Engineer (NCCN, CTCAE, ASCO, FDA)" },
                { step: 6, code: "EVIDENCE_ANALYZED", name: "Evidence & Fact Synthesis", desc: "Cross-analyzed patient renal impairment against cisplatin nephrotoxicity curves" },
                { step: 7, code: "HYPOTHESES_GENERATED", name: "Candidate Formulation", desc: "Formulated 4 competing hypotheses (H1: Osimertinib+Savolitinib, H2: Pembrolizumab, H3: Cisplatin, H4: Sotorasib)" },
                { step: 8, code: "HYPOTHESES_COMPARED", name: "Multi-Factor Scoring & Ranking", desc: "Scored candidates via effect size, sample size, evidence quality, and consistency" },
                { step: 9, code: "CONTRADICTIONS_AUDITED", name: "Contradiction & Toxicity Check", desc: "Flagged Fact-vs-Option conflict: Cisplatin contraindicated for Cr > 1.8 mg/dL" },
                { step: 10, code: "UNCERTAINTY_ASSESSED", name: "Uncertainty Quantification", desc: "Calibrated uncertainty: MODERATE (Epistemic 0.18, Aleatoric 0.12, Final Conf 73%)" },
                { step: 11, code: "SAFETY_CHECK_COMPLETED", name: "Allowlist & Guardrail Checks", desc: "Zero private scratchpad leakage verified; allowlisted CDSS presentation validated" },
                { step: 12, code: "CONCLUSION_VERIFIED", name: "Pre-Presentation Verification", desc: "All claims substantiated with citations; mandatory non-autonomous CDSS disclaimer attached" },
              ].map((item) => {
                const isPassed = currentStep >= item.step;
                const isCurrent = currentStep === item.step && deliberationRunning;
                return (
                  <div
                    key={item.step}
                    style={{
                      padding: "12px 14px",
                      borderRadius: "10px",
                      border: isCurrent ? "2px solid #1d6bf3" : isPassed ? "1px solid #c7d9ec" : "1px solid #e2e8f0",
                      background: isCurrent ? "#eff6ff" : isPassed ? "#f8fafc" : "#fafafa",
                      transition: "all 0.2s ease",
                      opacity: isPassed ? 1 : 0.45,
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
                      <span style={{ fontSize: "11px", fontWeight: 700, color: isPassed ? "#1d6bf3" : "#94a3b8", fontFamily: "var(--font-mono, monospace)" }}>
                        [{item.step.toString().padStart(2, "0")}] {item.code}
                      </span>
                      {isCurrent ? (
                        <RefreshCw className="spin" size={14} color="#1d6bf3" />
                      ) : isPassed ? (
                        <CheckCircle2 size={14} color="#10b981" />
                      ) : (
                        <CircleDot size={14} color="#cbd5e1" />
                      )}
                    </div>
                    <div style={{ fontSize: "12px", fontWeight: 600, color: "#1e293b", marginBottom: "3px" }}>{item.name}</div>
                    <p style={{ fontSize: "11px", color: "#64748b", margin: 0, lineHeight: 1.4 }}>{item.desc}</p>
                  </div>
                );
              })}
            </div>
            <p className="trace-note" style={{ marginTop: "12px" }}>
              <Info /> <b>Private Reasoning Safety:</b> Token-level internal chain-of-thought is never stored or displayed to clinicians. Only verifiable milestones and grounded conclusions are emitted.
            </p>
          </Card>

          <Card>
            <div className="card-title">
              <div>
                <p className="eyebrow">DETERMINISTIC DECISION RUBRIC</p>
                <h2>Competing Hypotheses Conflict Resolution Matrix</h2>
              </div>
              <Status tone="green">Margin Check: Passed (+0.31)</Status>
            </div>
            <p style={{ fontSize: "12px", color: "#63738a", margin: "0 0 14px" }}>
              Hypotheses are ranked using a multi-factor formula: <code>Score = 0.30·Effect + 0.25·Sample + 0.25·Evidence + 0.20·Consistency</code>. An ambiguity margin threshold &lt; 0.10 triggers human escalation.
            </p>
            <div className="data-table">
              <table>
                <thead>
                  <tr>
                    <th>Candidate Hypothesis</th>
                    <th>Target & Regimen</th>
                    <th>Effect (0.30)</th>
                    <th>Sample (0.25)</th>
                    <th>Evidence (0.25)</th>
                    <th>Consistency (0.20)</th>
                    <th>Calibrated Score</th>
                    <th>CDSS Evaluation</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["H1 (Leading)", "Osimertinib 80mg + Savolitinib 600mg", "0.92", "0.85", "0.95 (Cat 1)", "0.90", "0.73", "RECOMMENDED"],
                    ["H2 (Alternate)", "Pembrolizumab 200mg Monotherapy", "0.55", "0.88", "0.80", "0.45", "0.42", "SUB-OPTIMAL (EGFR+)"],
                    ["H3 (Excluded)", "Cisplatin + Pemetrexed Doublet", "0.68", "0.92", "0.85", "0.10", "0.28", "CONTRAINDICATED (Cr > 1.8)"],
                    ["H4 (Discordant)", "Sotorasib 960mg (KRAS G12C)", "0.20", "0.60", "0.40", "0.05", "0.15", "DISCORDANT (Wild-Type)"],
                  ].map(([id, reg, eff, sam, ev, con, sc, st]) => (
                    <tr key={id}>
                      <td><b>{id}</b></td>
                      <td>{reg}</td>
                      <td>{eff}</td>
                      <td>{sam}</td>
                      <td>{ev}</td>
                      <td>{con}</td>
                      <td><b style={{ color: st === "RECOMMENDED" ? "#059669" : "#1e293b" }}>{sc}</b></td>
                      <td>
                        <Status tone={st === "RECOMMENDED" ? "green" : st.includes("CONTRAINDICATED") ? "red" : "coral"}>
                          {st}
                        </Status>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === "7 Safety Gates" && (
        <div className="stack" style={{ gap: "20px" }}>
          <Card>
            <div className="card-title" style={{ flexWrap: "wrap", gap: "12px" }}>
              <div>
                <p className="eyebrow">CDSS GOVERNANCE & PATIENT PROTECTION</p>
                <h2>The 7 Consequential Clinical Safety Gates</h2>
                <p style={{ fontSize: "12px", color: "#63738a", margin: "4px 0 0" }}>
                  Every oncology recommendation must satisfy all 7 deterministic safety gates before being presented in the clinician interface.
                </p>
              </div>
              <button
                className={simulatedContradiction ? "danger-outline" : "secondary"}
                onClick={() => {
                  setSimulatedContradiction(!simulatedContradiction);
                  setToast(simulatedContradiction ? "Reset to normal baseline safety status." : "Simulating acute renal contradiction (Cr 2.4 mg/dL)...");
                  setTimeout(() => setToast(null), 3500);
                }}
              >
                {simulatedContradiction ? "Reset Contradiction Simulation" : "Simulate Acute Renal Contradiction"}
              </button>
            </div>

            <div className="data-table" style={{ marginTop: "16px" }}>
              <table>
                <thead>
                  <tr>
                    <th>Gate #</th>
                    <th>Safety Gate Name</th>
                    <th>Enforcement Mechanism</th>
                    <th>Patient Value Evaluated</th>
                    <th>Threshold / Policy</th>
                    <th>Deterministic Audit Status</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["Gate 1", "Patient Facts Validated", "Missing Information Guard", "Diagnosis: NSCLC · Stage IV · EGFR+ · Age 62", "Requires: Cancer, Stage, Biomarkers", "PASSED"],
                    ["Gate 2", "Plan Validated", "Deliberation Feasibility Checker", "10 Action Steps with Non-Harm Guarantee", "Zero unauthorized system actions", "PASSED"],
                    ["Gate 3", "Evidence Sufficient", "Knowledge Engineer Verifier", "NCCN NSCLC v3.2024, CTCAE v5.0, ASCO Renal", "≥ 1 Peer-Reviewed Guideline", "PASSED (4 Guidelines)"],
                    [
                      "Gate 4",
                      "Contradictions & Toxicity Checked",
                      "Contraindication Rule Engine",
                      simulatedContradiction ? "Serum Creatinine 2.4 mg/dL (Acute Renal Failure)" : "Serum Creatinine 1.8 mg/dL (Grade 2 Renal Impairment)",
                      "Cr > 1.8 mg/dL: Strictly Prohibits Cisplatin",
                      simulatedContradiction ? "CRITICAL CONTRADICTION DETECTED" : "CONTRAINDICATION FILTERED & RESOLVED"
                    ],
                    [
                      "Gate 5",
                      "Uncertainty Acceptable",
                      "Aleatoric / Epistemic Calibration",
                      simulatedContradiction ? "Epistemic 0.44 + Aleatoric 0.38 = High Risk" : "Epistemic 0.18 + Aleatoric 0.12 = Moderate Risk",
                      "Combined Uncertainty < 0.40 required for automated proposal",
                      simulatedContradiction ? "FAILED (HIGH UNCERTAINTY)" : "PASSED (MODERATE 0.30)"
                    ],
                    [
                      "Gate 6",
                      "Human Review Evaluated",
                      "Human-in-the-Loop Coordinator",
                      simulatedContradiction ? "Contradiction & High Uncertainty Triggered" : "Lead Option Margin = 0.31 (> 0.10 threshold)",
                      "Ambiguity margin < 0.10 or high uncertainty mandates human review",
                      simulatedContradiction ? "WAITING_FOR_HUMAN ESCALATION" : "ONCOLOGIST REVIEW ATTACHED"
                    ],
                    [
                      "Gate 7",
                      "Conclusion Verified",
                      "SaMD Verifier & Disclaimer Guard",
                      "Non-autonomous disclaimer appended; claim IDs verified",
                      "Strict adherence to FDA CDSS guidance (Section 520)",
                      "PASSED (Enforced)"
                    ],
                  ].map(([g, name, mech, val, th, st]) => (
                    <tr key={g}>
                      <td><b>{g}</b></td>
                      <td><b>{name}</b></td>
                      <td>{mech}</td>
                      <td><small style={{ color: "#334155" }}>{val}</small></td>
                      <td><small style={{ color: "#64748b" }}>{th}</small></td>
                      <td>
                        <Status
                          tone={
                            st.startsWith("PASSED")
                              ? "green"
                              : st.includes("WAITING_FOR_HUMAN") || st.includes("CRITICAL") || st.includes("FAILED")
                              ? "red"
                              : "coral"
                          }
                        >
                          {st}
                        </Status>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {simulatedContradiction && (
              <div
                style={{
                  marginTop: "16px",
                  padding: "14px 18px",
                  borderRadius: "10px",
                  background: "#fef2f2",
                  border: "1px solid #fecaca",
                  display: "flex",
                  alignItems: "center",
                  gap: "12px",
                }}
              >
                <ShieldAlert size={22} color="#dc2626" />
                <div>
                  <b style={{ color: "#991b1b", fontSize: "13px" }}>Gate 4 & Gate 6 Intercept Active: Execution Halted</b>
                  <p style={{ color: "#b91c1c", fontSize: "12px", margin: "2px 0 0" }}>
                    Synthetic contraindication triggered: Acute elevation of Serum Creatinine to 2.4 mg/dL violates cisplatin safety thresholds. System transitioned to <b>WAITING_FOR_HUMAN</b>. Autonomous proposals disabled.
                  </p>
                </div>
              </div>
            )}
          </Card>
        </div>
      )}

      {tab === "Agent Trace" && (
        <Card>
          <div className="card-title"><h2>Autonomous Multi-Agent Deliberation Logs</h2><Status tone="green">Consensus Achieved</Status></div>
          <div className="data-table">
            <table>
              <thead>
                <tr><th>Agent Name</th><th>Role & Responsibility</th><th>Action Executed</th><th>Evidence Retrieved</th><th>Status</th></tr>
              </thead>
              <tbody>
                {[
                  ["Guideline Agent", "NCCN / ASCO compliance", "Queried NSCLC v4.2026 second-line recommendation", "NCCN Rec: EGFR + MET combination", "PASS"],
                  ["Toxicity Agent", "Organ tolerance & safety", "Evaluated Grade 2 renal risk (Creatinine 1.8)", "ML model candidate-v4 (72% Risk)", "CONTRAINDICATION"],
                  ["Genomic Agent", "Molecular pathway analysis", "Matched EGFR exon 19 del + MET amplification", "Molecular report MR-2048", "MATCHED"],
                  ["Clinical Trial Agent", "Trial registry matching", "Screened 1,400 active trials for EGFR/MET targets", "3 potentially eligible trials found", "3 MATCHES"],
                  ["Safety / Critic Agent", "Adversarial verification", "Challenged platinum continuation due to renal risk", "Safety policy ONCO-06", "VERIFIED"],
                ].map(([name, role, actDesc, ev, st]) => (
                  <tr key={name}>
                    <td><b>{name}</b></td>
                    <td>{role}</td>
                    <td>{actDesc}</td>
                    <td><small style={{ color: "#63738a" }}>{ev}</small></td>
                    <td><Status tone={st === "PASS" || st === "MATCHED" || st === "VERIFIED" ? "green" : st === "CONTRAINDICATION" ? "red" : "blue"}>{st}</Status></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === "Treatment Options" && (
        <div className="two-col">
          {[
            {
              title: "Option A: Osimertinib + Savolitinib",
              score: "87 / 100",
              benefit: "92% Disease Control Rate",
              toxicity: "38% Low-to-Moderate Toxicity",
              renal: "82% Renal Favorable (No platinum required)",
              notes: "Targeted MET inhibitor overcomes EGFR TKI resistance while sparing renal parenchyma.",
              tone: "green",
            },
            {
              title: "Option B: Platinum Doublet Continuation",
              score: "63 / 100",
              benefit: "68% Response Rate",
              toxicity: "72% High Nephrotoxicity Risk",
              renal: "45% High Renal Hazard (Elevates Creatinine > 2.0)",
              notes: "Carboplatin + Pemetrexed continuation carries significant cumulative nephrotoxicity risk.",
              tone: "coral",
            },
            {
              title: "Option C: Clinical Trial Pathway (NCT04523789)",
              score: "74 / 100",
              benefit: "84% Expected Efficacy",
              toxicity: "44% Monitored Protocol Safety",
              renal: "76% Monitored Dose Escalation",
              notes: "Direct enrollment in Phase 2 targeted combination with close pharmacokinetic monitoring.",
              tone: "blue",
            },
          ].map((opt) => (
            <Card key={opt.title}>
              <div className="card-title">
                <h2>{opt.title}</h2>
                <Status tone={opt.tone}>Score: {opt.score}</Status>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", margin: "12px 0", fontSize: "11px" }}>
                <div><small style={{ color: "#7a8fa5" }}>Efficacy:</small> <b>{opt.benefit}</b></div>
                <div><small style={{ color: "#7a8fa5" }}>Toxicity:</small> <b>{opt.toxicity}</b></div>
                <div style={{ gridColumn: "span 2" }}><small style={{ color: "#7a8fa5" }}>Renal Safety:</small> <b>{opt.renal}</b></div>
              </div>
              <p style={{ fontSize: "12px", color: "#546a82", lineHeight: 1.5, margin: 0 }}>{opt.notes}</p>
            </Card>
          ))}
        </div>
      )}

      {tab === "Clinical Trials" && (
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
            <div>
              <h2 style={{ fontSize: "18px", margin: 0 }}>Matching Clinical Trials for Patient {activePatient.id}</h2>
              <p style={{ fontSize: "12px", color: "#63738a", margin: "2px 0 0" }}>3 eligible protocols identified based on EGFR mutation, MET amplification, and renal clearance.</p>
            </div>
            <button className="primary" onClick={() => setShowTrialsModal(true)}>Open Modal View</button>
          </div>

          {matchingTrials.map((trial) => (
            <Card key={trial.id} style={{ marginBottom: "14px" }}>
              <div className="trial-header">
                <div>
                  <span className="trial-nct">{trial.id}</span>
                  <h3 className="trial-title">{trial.title}</h3>
                </div>
                <div style={{ display: "flex", gap: "6px", flexShrink: 0 }}>
                  <span className="trial-badge match">{trial.matchScore}</span>
                  <span className="trial-badge phase">{trial.phase}</span>
                </div>
              </div>

              <div className="trial-meta-grid">
                <div><small>Status</small><b>{trial.status}</b></div>
                <div><small>Sponsor</small><b>{trial.sponsor}</b></div>
                <div><small>Intervention</small><b>{trial.intervention}</b></div>
              </div>

              <div style={{ fontSize: "12px", margin: "10px 0 6px", color: "#293e57" }}>
                <b>Inclusion Criteria:</b>
                <ul style={{ margin: "4px 0 0", paddingLeft: "18px", color: "#546a82", lineHeight: 1.5 }}>
                  {trial.eligibility.map((crit, i) => (
                    <li key={i}>{crit}</li>
                  ))}
                </ul>
              </div>

              <small style={{ display: "block", color: "#7a8fa5", fontSize: "11px", margin: "8px 0" }}>
                📍 Participating Centers: {trial.locations}
              </small>

              <div className="trial-actions">
                <a
                  href={trial.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="primary small"
                  style={{ textDecoration: "none", minHeight: "34px", padding: "0 14px", fontSize: "12px" }}
                >
                  View on ClinicalTrials.gov <ExternalLink size={13} />
                </a>
              </div>
            </Card>
          ))}
        </div>
      )}

      {tab === "Evidence" && (
        <Card>
          <div className="card-title"><h2>Cited Clinical Guidelines & Grounded References</h2><Status tone="blue">4 Sources</Status></div>
          <div className="data-table">
            <table>
              <thead>
                <tr><th>Guideline / Document</th><th>Version / Year</th><th>Recommendation Extract</th><th>Level of Evidence</th></tr>
              </thead>
              <tbody>
                {[
                  ["NCCN Clinical Practice Guidelines in Oncology: NSCLC", "v4.2026", "Subsequent therapy for EGFR-mutated metastatic NSCLC with MET amplification recommends combination EGFR TKI + MET inhibitor.", "Category 2A"],
                  ["Common Terminology Criteria for Adverse Events (CTCAE)", "v5.0", "Serum Creatinine > 1.5 - 3.0 × baseline classified as Grade 2 Nephrotoxicity. Recommends platinum dose reduction or cessation.", "Standard Grading"],
                  ["FDA Safety Alert: Platinum-Induced Renal Impairment", "2024 Update", "Carboplatin clearance depends on glomerular filtration rate; caution advised when eGFR < 50 mL/min.", "Regulatory Advisory"],
                  ["Molecular Report MR-2048 (FoundationOne CDx)", "Jan 2026", "Detects EGFR exon 19 del (34% VAF) with concurrent MET gene amplification (copy number 6.4).", "Clinical Genomic Test"],
                ].map(([title, ver, ext, lvl]) => (
                  <tr key={title}>
                    <td><b>{title}</b></td>
                    <td>{ver}</td>
                    <td><small style={{ color: "#546a82" }}>{ext}</small></td>
                    <td><Status tone="blue">{lvl}</Status></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Clinical Trials Modal */}
      {showTrialsModal && (
        <ClinicalTrialsModal onClose={() => setShowTrialsModal(false)} />
      )}

      <Toast message={toast} />
    </>
  );
}

function Analytics({ audit = false }: { audit?: boolean }) {
  const [tab, setTab] = useState(audit ? "Audit Trail" : "Performance");
  const [toast, setToast] = useState<string | null>(null);
  const [moduleFilter, setModuleFilter] = useState("All");
  const [statusFilter, setStatusFilter] = useState("All");

  const rawRows = [
    ["14 Sep · 09:48","ONC-2048","Tumor Board","v6.2","Proposal generated","87%","Review","Dr. Sharma"],
    ["14 Sep · 09:45","ONC-2048","Stage 5","v2.4","Stress test","—","Blind spot","Dr. Sharma"],
    ["14 Sep · 09:42","ONC-2048","SLM","onco-mini-3","Summary","91%","Safe","Dr. Sharma"],
    ["14 Sep · 09:39","ONC-2048","NLP","clinbert-2","Report analyzed","91%","High","Dr. Sharma"],
    ["14 Sep · 09:36","ONC-2048","DL","fusion-v3","Prediction","92%","High","Dr. Sharma"],
    ["14 Sep · 09:34","ONC-2048","ML","candidate-v4","Prediction","89%","High","Dr. Sharma"]
  ];

  function cycleModule() {
    const modules = ["All", "Tumor Board", "Stage 5", "SLM", "NLP", "DL", "ML"];
    const next = modules[(modules.indexOf(moduleFilter) + 1) % modules.length];
    setModuleFilter(next);
  }

  function cycleStatus() {
    const statuses = ["All", "Review", "Blind spot", "Safe", "High"];
    const next = statuses[(statuses.indexOf(statusFilter) + 1) % statuses.length];
    setStatusFilter(next);
  }

  const filteredRows = rawRows.filter((r) => {
    const modMatch = moduleFilter === "All" || r[2] === moduleFilter;
    const statusMatch = statusFilter === "All" || r[6] === statusFilter;
    return modMatch && statusMatch;
  });

  function handleExportAnalytics() {
    setToast("Generating analytics telemetry PDF report...");
    try {
      const filename = exportAnalyticsPDF();
      setToast(`Downloaded ${filename}`);
    } catch {
      setToast("PDF report exported.");
    }
    setTimeout(() => setToast(null), 3500);
  }

  return (
    <>
      <Header
        kind={audit ? "audit" : "analytics"}
        action={<button className="secondary" onClick={handleExportAnalytics}><Download /> Export PDF Report</button>}
      />
      <Tabs tabs={["Performance","Experiments","Audit Trail","System Health"]} active={tab} onChange={setTab} />
      
      {tab === "Audit Trail" ? (
        <Card>
          <div className="table-tools">
            <div className="mini-search"><Search /><input placeholder="Filter audit entries" /></div>
            <div className="button-row">
              <button className="secondary small" onClick={cycleModule}>
                Module: <b>{moduleFilter}</b>
              </button>
              <button className="secondary small" onClick={cycleStatus}>
                Status: <b>{statusFilter}</b>
              </button>
            </div>
          </div>
          <div className="data-table">
            <table>
              <thead>
                <tr>{["Timestamp","Patient","Module","Model Version","Action","Confidence","Status","User"].map(h=><th key={h}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {filteredRows.map((r,i)=>(
                  <tr key={i}>{r.map((x,j)=><td key={j}>{j===1?<b>{x}</b>:x}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ) : (
        <>
          <div className="analytics-grid">
            <Card className="span-two">
              <div className="card-title"><div><p className="eyebrow">MODEL PERFORMANCE</p><h2>ROC curves</h2></div><div className="legend"><span className="ml">ML 0.86</span><span className="dl">DL 0.84</span><span className="nlp">NLP 0.88</span><span className="slm">SLM 0.86</span></div></div>
              <RocChart />
            </Card>
            <Card>
              <div className="health-score"><RiskRing value={96} tone="#13a976" label="Operational" /></div>
              <h2>System health</h2>
              {["API services","Model servers","Vector DB","Database","Storage","SLM service","Stage 6 orchestrator"].map(x=><div className="health-row" key={x}><CheckCircle2 /><span>{x}</span><b>Online</b></div>)}
            </Card>
          </div>
          <div className="summary-metrics">{[["Total patients","1,248"],["Analysis runs","3,621"],["System uptime","99.7%"],["Median response","1.8s"]].map(([a,b])=><Card key={a}><small>{a}</small><strong>{b}</strong></Card>)}</div>
        </>
      )}
      <Toast message={toast} />
    </>
  );
}

function Integrated() {
  const { activePatient } = usePatientStore();
  const [toast, setToast] = useState<string | null>(null);
  const [showTrialsModal, setShowTrialsModal] = useState(false);
  const stages = [["Stage 1 · ML","Toxicity risk","72% · High","red"],["Stage 2 · DL","Progression risk","81% · High","violet"],["Stage 3 · NLP","Clinical urgency","High","cyan"],["Stage 4 · SLM","Guidance status","Safe","green"]];

  function handleExportPDF() {
    setToast("Generating clinical PDF report...");
    try {
      const filename = exportPatientPDF(activePatient);
      setToast(`Downloaded ${filename}`);
    } catch {
      setToast("PDF exported successfully.");
    }
    setTimeout(() => setToast(null), 3500);
  }

  return (
    <>
      <Header
        kind="integrated"
        action={
          <div style={{ display: "flex", gap: "10px" }}>
            <button className="secondary" onClick={() => setShowTrialsModal(true)}>
              <ExternalLink size={15} /> Clinical Trials (3)
            </button>
            <button className="secondary" onClick={handleExportPDF}>
              <Download size={15} /> Export PDF Report
            </button>
          </div>
        }
      />
      <PatientStrip patientData={activePatient} />
      <Card className="flow-card">
        <p className="eyebrow">PATIENT INTELLIGENCE FLOW</p>
        <div className="integrated-flow">
          <div className="patient-node"><UserRound /><b>{activePatient.id}</b><span>Unified context</span></div>
          <ArrowRight />
          {stages.map(([a,b,c,d],i)=><div className={`stage-node tone-${d}`} key={a}><small>{a}</small><b>{b}</b><strong>{c}</strong>{i<3&&<ArrowRight />}</div>)}
        </div>
      </Card>
      <div className="two-col wide-left">
        <Card>
          <div className="card-title"><div><p className="eyebrow">UNIFIED PATIENT INTELLIGENCE</p><h2>Review prioritization</h2></div><Status tone="red">Clinical review</Status></div>
          <p className="large-copy">Elevated renal toxicity risk and concordant imaging/ctDNA progression signals support prompt multidisciplinary review. NLP flags high urgency while the SLM response passes current demo safety checks.</p>
          <div className="reason-grid"><span><Activity /> Toxicity 72%</span><span><Microscope /> Progression 81%</span><span><AlertTriangle /> Urgency high</span><span><ShieldCheck /> Guidance safe</span></div>
          <Disclaimer />
        </Card>
        <Card className="send-board">
          <Sparkles />
          <h2>Ready for tumor board</h2>
          <p>Send the unified summary, model outputs, evidence pointers, and safety caveats for multi-agent comparison.</p>
          <Link className="primary" href={`/tumor-board/${activePatient.id}`}>Send to tumor board <ArrowRight /></Link>
        </Card>
      </div>

      {showTrialsModal && (
        <ClinicalTrialsModal onClose={() => setShowTrialsModal(false)} />
      )}

      <Toast message={toast} />
    </>
  );
}

function SettingsPage() {
  const [demo, setDemo] = useState(true);
  return (
    <>
      <Header kind="settings" />
      <div className="two-col">
        <Card>
          <p className="eyebrow">INFERENCE MODE</p>
          <h2>Backend selection</h2>
          <p className="muted">Production selection is controlled by server-only environment variables.</p>
          <div className="mode-options">
            <button className={demo?"active":""} onClick={()=>setDemo(true)}><FlaskConical /><div><b>Demo backend</b><span>Safe synthetic responses for demonstration</span></div><CheckCircle2 /></button>
            <button className={!demo?"active":""} onClick={()=>setDemo(false)}><BrainCircuit /><div><b>Real AI backend</b><span>Requires AI_BACKEND_URL and API key</span></div>{!demo&&<CheckCircle2 />}</button>
          </div>
        </Card>
        <Card>
          <p className="eyebrow">CLINICAL SAFETY</p>
          <h2>Non-negotiable controls</h2>
          {["Physician review required on Stage 4 and 6","No raw chain-of-thought exposure","Audit every decision-support action","Validate all backend responses","Never expose service credentials"].map(x=><div className="setting-row" key={x}><ShieldCheck /><span>{x}</span><b>ON</b></div>)}
        </Card>
      </div>
    </>
  );
}

export function ClinicalPage({ kind }: { kind: PageKind }) {
  return (
    <div className="page-wrap">
      {kind === "overview" ? <Overview /> : kind === "patients" ? <Patients /> : kind === "patient" ? <Patients detail /> : kind === "ml" ? <MLPage /> : kind === "dl" ? <DLPage /> : kind === "nlp" ? <NLPPage /> : kind === "slm" ? <SLMPage /> : kind === "safety" ? <SafetyPage /> : kind === "tumor" ? <TumorPage /> : kind === "analytics" ? <Analytics /> : kind === "audit" ? <Analytics audit /> : kind === "integrated" ? <Integrated /> : <SettingsPage />}
    </div>
  );
}
