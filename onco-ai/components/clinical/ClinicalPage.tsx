"use client";

import Link from "next/link";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, AlertTriangle, ArrowDown, ArrowRight, BadgeCheck, BrainCircuit, Check, CheckCircle2, ChevronRight, CircleDot, ClipboardCheck, Clock3, Dna, Download, ExternalLink, FileSearch, FlaskConical, Info, Microscope, Play, RefreshCw, Search, ShieldAlert, ShieldCheck, Sparkles, Square, Upload, UserPlus, UserRound, X, XCircle, Zap } from "lucide-react";
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

function Tabs({ tabs, active, onChange }: TabProps) { return <div className="tabs" role="tablist">{tabs.map((tab) => <button key={tab} role="tab" aria-selected={active === tab} className={active === tab ? "active" : ""} onClick={() => onChange(tab)}>{tab}</button>)}</div> }
function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) { return <motion.section className={`card ${className}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .22 }}>{children}</motion.section> }
function Status({ tone, children }: { tone: string; children: React.ReactNode }) { return <span className={`status status-${tone}`}><i />{children}</span> }
function MetricGrid() { return <div className="metric-grid">{metrics.map((m) => <Card className={`metric-card tone-${m.tone}`} key={m.key}><div className="metric-top"><span>{m.key}</span><Status tone={m.tone}>{m.status}</Status></div><p>{m.label}</p><strong>{m.value}</strong><div className="metric-foot">Synthetic model output <ChevronRight /></div></Card>)}</div> }
function Header({ kind, action }: { kind: PageKind; action?: React.ReactNode }) { const meta = pageMeta[kind]; const Icon = meta.icon; return <div className="page-heading"><div className={`heading-icon accent-${meta.accent}`}><Icon /></div><div><p className="eyebrow">{meta.eyebrow}</p><h1>{meta.title}</h1><p>{meta.subtitle}</p></div>{action && <div className="heading-action">{action}</div>}</div> }

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

function Disclaimer() { return <div className="disclaimer"><ShieldCheck /> AI-generated clinical decision support. Physician review required.</div> }
function Processing({ steps, done }: { steps: string[]; done: number }) { return <div className="processing" role="status" aria-live="polite">{steps.map((s, i) => <div key={s} className={i < done ? "done" : i === done ? "running" : ""}><span>{i < done ? <Check /> : i === done ? <RefreshCw /> : <CircleDot />}</span>{s}</div>)}</div> }

function Toast({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div className="floating-toast" role="status">
      <CheckCircle2 size={18} />
      <span>{message}</span>
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
    } catch (e) {
      console.error(e);
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
  const [toast, setToast] = useState<string | null>(null);

  function handleExportPDF() {
    setToast("Generating clinical PDF summary...");
    try {
      const filename = exportPatientPDF(activePatient);
      setToast(`Downloaded ${filename}`);
    } catch (e) {
      console.error(e);
      setToast("PDF exported successfully.");
    }
    setTimeout(() => setToast(null), 3500);
  }

  function handleAddSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const fd = new FormData(e.currentTarget);
    const newPatient: Patient = {
      id: ((fd.get("id") as string) || `ONC-${2050 + patients.length}`).trim().toUpperCase(),
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
            <button className="primary">Discuss with Doc AI</button>
          </div>
        }
      />
      <div className="patient-badges">
        <span>{p.cancer}</span>
        <span>Stage {p.stage}</span>
        <span>ECOG {p.ecog}</span>
        <Status tone="red">Active review</Status>
      </div>
      <Tabs tabs={["Overview","Clinical History","Lab Results","Imaging","Genomics","Documents"]} active={tab} onChange={setTab} />
      {tab === "Overview" ? (
        <div className="two-col">
          <div className="stack">
            <Card>
              <div className="card-title"><h2>Demographics</h2><button className="text-button">Edit</button></div>
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
      ) : (
        <Card className="empty-state">
          <FileSearch />
          <h2>{tab}</h2>
          <p>Demo records for this section are ready to connect to the production data service.</p>
          <button className="secondary" onClick={() => setShowAddModal(true)}>Add demo record</button>
        </Card>
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
    setMessage(`${action} complete · Demo / simulated output`);
    ai.speak(`${action} completed using synthetic data. Creatinine remains the strongest contributing factor.`);
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
            <Card className="interpretation"><Info /><div><p className="eyebrow">CLINICAL INTERPRETATION</p><h2>Renal markers drive risk</h2><p>Elevated renal markers combined with platinum therapy are contributing strongly to predicted treatment toxicity.</p><div className="button-row"><button className="secondary" onClick={() => run("Full analysis")}>View full analysis</button><button className="secondary" onClick={() => run("Factor explanation")}>Explain factors</button></div>{message && <small className="success-line"><Check /> {message}</small>}</div></Card>
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
      <div className="two-col wide-left">
        <Card>
          <div className="card-title"><div><p className="eyebrow">PATHOLOGY IMAGE</p><h2>H&E lung biopsy · demo sample</h2></div><div className="segmented">{["Original","AI Attention","Side-by-Side"].map(v => <button key={v} onClick={() => setView(v)} className={view===v?"active":""}>{v}</button>)}</div></div>
          <div className={`pathology-view view-${view.toLowerCase().replaceAll(" ","-")}`}><div className="tissue tissue-a" /><div className="tissue tissue-b" /><div className="tissue tissue-c" />{view !== "Original" && <div className="heatmap" />}<span>{view} · illustrative visualization</span></div>
        </Card>
        <Card><p className="eyebrow">PROGRESSION RISK</p><RiskRing value={81} tone="#7657ed" label="High risk" /><div className="confidence">Confidence <b>92%</b></div><div className="split-scores"><span>Spatial<b>77%</b></span><span>Temporal<b>85%</b></span><span>Fusion<b>81%</b></span></div></Card>
      </div>
      <Card><div className="card-title"><div><p className="eyebrow">LONGITUDINAL SIGNALS</p><h2>ctDNA and CEA trend</h2></div><Status tone="red">Increasing</Status></div><TrendChart /></Card>
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

  return (
    <>
      <Header
        kind="nlp"
        action={
          <div style={{ display: "flex", gap: "10px" }}>
            <button className="secondary" onClick={handleExportPDF}><Download /> Export PDF</button>
            <label className="upload-button"><Upload /> Upload report<input type="file" hidden accept=".pdf,.txt" /></label>
          </div>
        }
      />
      <PatientStrip patientData={activePatient} />
      <Tabs tabs={["Report Viewer","Extracted Entities","Negation Analysis","Model Details"]} active={tab} onChange={setTab} />
      <div className="two-col wide-left">
        <Card>
          <div className="card-title"><div><p className="eyebrow">CLINICAL REPORT · SAMPLE</p><h2>Oncology follow-up note</h2></div><Status tone="blue">Demo data</Status></div>
          <div className="report-text">Patient reports severe <mark className="entity symptom">fatigue</mark> and <mark className="entity symptom">shortness of breath</mark>. Denies <mark className="entity negated">chest pain</mark>. However, <mark className="entity urgent">rapid pulse</mark> was observed. Currently receiving <mark className="entity drug">Carboplatin</mark>. Recent imaging suggests <mark className="entity anatomy">lymph node progression</mark>. <mark className="entity biomarker">EGFR mutation</mark> positive.</div>
          <div className="entity-legend"><span className="symptom">Symptoms</span><span className="drug">Drugs</span><span className="biomarker">Biomarkers</span><span className="negated">Negated</span><span className="urgent">Urgency</span></div>
          {tab === "Negation Analysis" && <div className="negation-card"><p>“Denies chest pain”</p><span>Detected <b>YES</b></span><span>Negated <b>YES</b></span><span>Final state <b>ABSENT</b></span></div>}
        </Card>
        <div className="stack">
          <Card><p className="eyebrow">EXTRACTION SUMMARY</p><div className="extraction-list">{counts.map(([a,b,c]) => <div key={String(a)}><span className={`dot-${c}`} /><b>{a}</b><strong>{b}</strong></div>)}</div></Card>
          <Card className="urgency-card"><AlertTriangle /><p>CLINICAL URGENCY</p><strong>HIGH</strong><span>Confidence 91%</span></Card>
        </div>
      </div>
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
      <Toast message={toast} />
    </>
  );
}

function TumorPage() {
  const { activePatient } = usePatientStore();
  const [decision, setDecision] = useState(""), [tab, setTab] = useState("Decision Overview");
  const [toast, setToast] = useState<string | null>(null);
  const agents = [["Guideline Agent","Guideline evidence retrieved","NCCN NSCLC v4.2026"],["Toxicity Agent","Renal constraint identified","ML candidate-v4"],["Genomic Agent","EGFR context matched","Molecular report MR-2048"],["Clinical Trial Agent","3 potential trials found","Registry snapshot · demo"],["Safety / Critic Agent","Proposal challenged and revised","Safety policy ONCO-06"]];
  const candidates = [["A · Osimertinib + Savolitinib","92%","38%","82%","High","87"],["B · Platinum continuation","68%","72%","45%","Moderate","63"],["C · Trial pathway","84%","44%","76%","Emerging","74"]];

  function act(value: string) { setDecision(`${value} recorded in demo audit trail. No treatment action was taken.`); }

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
      <Header kind="tumor" action={<button className="secondary" onClick={handleExportPDF}><Download /> Export PDF Decision</button>} />
      <PatientStrip patientData={activePatient} />
      <div className="intelligence-feed">{metrics.map(m => <div key={m.key}><span>{m.key}</span><b>{m.value}</b><small>{m.label}</small></div>)}<div><span>STAGE 5</span><b>2</b><small>Blind spots</small></div></div>
      <Tabs tabs={["Decision Overview","Agent Trace","Treatment Options","Clinical Trials","Evidence"]} active={tab} onChange={setTab} />
      <div className="two-col wide-left">
        <div className="stack">
          <Card>
            <div className="card-title"><div><p className="eyebrow">RECOMMENDED STRATEGY · DEMO</p><h2>Osimertinib + Savolitinib</h2></div><div className="recommend-score"><span>Confidence</span><b>87%</b></div></div>
            <p className="strategy-note">Illustrative renal-adjusted option for discussion—not a prescription or treatment selection.</p>
            <div className="reason-grid">{["Guideline aligned","Renal adjusted","Genomics matched","Trials available"].map(x => <span key={x}><Check /> {x}</span>)}</div>
            <Disclaimer />
          </Card>
          <Card>
            <div className="card-title"><div><p className="eyebrow">AI AGENTS AT WORK</p><h2>Auditable action trace</h2></div><Status tone="green">Completed</Status></div>
            <div className="agent-trace">{agents.map(([name,result,evidence],i) => <div key={name}><span>{i+1}</span><div><b>{name}</b><p>{result}</p><small><Clock3 /> 09:4{i} · {evidence}</small></div><CheckCircle2 /></div>)}</div>
            <p className="trace-note"><Info /> Shows actions, evidence, observations, and decision summaries only. Hidden chain-of-thought is never displayed.</p>
          </Card>
        </div>
        <div className="stack">
          <Card><div className="card-title"><h2>Treatment trade-off</h2><Status tone="blue">3 candidates</Status></div><div className="data-table compact"><table><thead><tr><th>Candidate</th><th>Benefit</th><th>Toxicity</th><th>Renal</th><th>Evidence</th><th>Score</th></tr></thead><tbody>{candidates.map(row => <tr key={row[0]}>{row.map((x,i) => <td key={x}>{i===0?<b>{x}</b>:x}</td>)}</tr>)}</tbody></table></div></Card>
          <Card><div className="trial-match"><div><p className="eyebrow">CLINICAL TRIAL MATCH</p><strong>3</strong><span>potentially eligible demo trials</span></div><button className="secondary">View trials <ExternalLink /></button></div></Card>
        </div>
      </div>
      <Card className="physician-review">
        <div><ShieldAlert /><div><p>PHYSICIAN REVIEW REQUIRED</p><h2>Final treatment decisions remain with the physician.</h2></div></div>
        <div className="decision-buttons">
          <button className="danger-outline" onClick={() => act("Rejected")}><XCircle /> Reject</button>
          <button className="secondary" onClick={() => act("Plan modification requested")}>Modify plan</button>
          <button className="approve" onClick={() => act("Physician approval")}><CheckCircle2 /> Physician approve</button>
          <button className="emergency" onClick={() => act("Emergency stop")}><Square /> Emergency stop</button>
        </div>
        {decision && <p className="decision-feedback"><Check /> {decision}</p>}
      </Card>
      <Toast message={toast} />
    </>
  );
}

function Analytics({ audit = false }: { audit?: boolean }) {
  const [tab, setTab] = useState(audit ? "Audit Trail" : "Performance");
  const [toast, setToast] = useState<string | null>(null);
  const rows = [["14 Sep · 09:48","ONC-2048","Tumor Board","v6.2","Proposal generated","87%","Review","Dr. Sharma"],["14 Sep · 09:45","ONC-2048","Stage 5","v2.4","Stress test","—","Blind spot","Dr. Sharma"],["14 Sep · 09:42","ONC-2048","SLM","onco-mini-3","Summary","91%","Safe","Dr. Sharma"],["14 Sep · 09:39","ONC-2048","NLP","clinbert-2","Report analyzed","91%","High","Dr. Sharma"],["14 Sep · 09:36","ONC-2048","DL","fusion-v3","Prediction","92%","High","Dr. Sharma"],["14 Sep · 09:34","ONC-2048","ML","candidate-v4","Prediction","89%","High","Dr. Sharma"]];

  function handleExportAnalytics() {
    setToast("Generating analytics telemetry PDF report...");
    try {
      const filename = exportAnalyticsPDF();
      setToast(`Downloaded ${filename}`);
    } catch (e) {
      console.error(e);
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
          <div className="table-tools"><div className="mini-search"><Search /><input placeholder="Filter audit entries" /></div><div className="button-row"><button className="secondary small">Module: All</button><button className="secondary small">Status: All</button></div></div>
          <div className="data-table"><table><thead><tr>{["Timestamp","Patient","Module","Model Version","Action","Confidence","Status","User"].map(h=><th key={h}>{h}</th>)}</tr></thead><tbody>{rows.map((r,i)=><tr key={i}>{r.map((x,j)=><td key={j}>{j===1?<b>{x}</b>:x}</td>)}</tr>)}</tbody></table></div>
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
  const stages = [["Stage 1 · ML","Toxicity risk","72% · High","red"],["Stage 2 · DL","Progression risk","81% · High","violet"],["Stage 3 · NLP","Clinical urgency","High","cyan"],["Stage 4 · SLM","Guidance status","Safe","green"]];

  function handleExportPDF() {
    setToast("Generating clinical PDF report...");
    try {
      const filename = exportPatientPDF(activePatient);
      setToast(`Downloaded ${filename}`);
    } catch (e) {
      console.error(e);
      setToast("PDF exported successfully.");
    }
    setTimeout(() => setToast(null), 3500);
  }

  return (
    <>
      <Header
        kind="integrated"
        action={
          <button className="secondary" onClick={handleExportPDF}>
            <Download /> Export PDF Report
          </button>
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
