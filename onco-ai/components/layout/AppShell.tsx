"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { signOut } from "next-auth/react";
import { Activity, AlertTriangle, BarChart3, Bell, BrainCircuit, CalendarDays, ChevronLeft, ClipboardList, Command, Dna, FileSearch, FlaskConical, Info, LayoutDashboard, LogOut, Menu, Microscope, MoonStar, Search, Settings, ShieldAlert, Sparkles, Upload, Users, X } from "lucide-react";
import { useUIStore } from "@/stores/ui-store";
import { DocAIProvider } from "@/components/doc-ai/DocAIProvider";
import { DocAI } from "@/components/doc-ai/DocAI";
import { CopilotDock } from "@/components/doc-ai/CopilotDock";
import { usePatientStore } from "@/stores/patient-store";

const groups = [
  {
    label: "",
    items: [
      { href: "/overview", label: "Overview", icon: LayoutDashboard },
      { href: "/patients", label: "Patients", icon: Users },
      { href: "/integrated-analysis/ACTIVE", label: "Integrated Analysis", icon: Dna },
    ],
  },
  {
    label: "AI MODULES",
    items: [
      { href: "/ml", label: "ML - Toxicity", icon: Activity },
      { href: "/dl", label: "DL - Progression", icon: Microscope },
      { href: "/nlp", label: "NLP - Reports", icon: FileSearch },
      { href: "/slm", label: "SLM - Copilot", icon: BrainCircuit },
    ],
  },
  {
    label: "AI SAFETY",
    items: [
      { href: "/safety-lab", label: "Stage 5 - Safety Lab", icon: FlaskConical },
    ],
  },
  {
    label: "AUTONOMOUS INTELLIGENCE",
    items: [
      { href: "/tumor-board/ACTIVE", label: "Stage 6 - Tumor Board", icon: Sparkles },
    ],
  },
  {
    label: "ENGINEERING",
    items: [
      { href: "/analytics", label: "Analytics", icon: BarChart3 },
      { href: "/audit", label: "Audit Trail", icon: ClipboardList },
    ],
  },
];

const messages: Record<string, string> = {
  overview: "Toxicity risk is elevated and ctDNA is increasing. I can help explain the evidence for review.",
  ml: "This patient has elevated treatment-toxicity risk. I can explain the contributing clinical factors.",
  dl: "I found increasing progression signals across pathology and longitudinal ctDNA.",
  nlp: "Key clinical entities were extracted and high urgency was detected.",
  slm: "Ask me about this patient. I will use available synthetic context and show safety status.",
  "safety-lab": "This synthetic case exposed model disagreement. I can explain the blind spot.",
  "tumor-board": "The agent team completed its analysis. The proposal is ready for physician review.",
  analytics: "All AI services are operational in demo mode. I can summarize recent model performance.",
  patients: "Creatinine is elevated and ctDNA is increasing. These findings may influence treatment planning.",
  "integrated-analysis": "The four intelligence stages have been combined into one clinician-reviewable summary.",
  audit: "Every decision-support event is recorded for review and governance.",
  settings: "Demo and real backend selection is controlled through server-side environment configuration.",
};

const clinicalAlerts = [
  {
    id: "alt-1",
    type: "critical",
    title: "ONC-2048: Renal Toxicity Risk",
    desc: "Serum Creatinine elevated at 1.8 mg/dL. Predicted Grade 2+ renal event with platinum therapy.",
    time: "10m ago",
  },
  {
    id: "alt-2",
    type: "warning",
    title: "ctDNA Kinetic Acceleration",
    desc: "Longitudinal ctDNA rose from 61 to 76 ng/mL (+24%). Disease progression review advised.",
    time: "35m ago",
  },
  {
    id: "alt-3",
    type: "info",
    title: "Stage 5 Safety Lab Alert",
    desc: "Adversarial stress test flagged blind spot BS-2048-07 in cross-model calibration.",
    time: "1h ago",
  },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const path = usePathname();
  const { activePatient } = usePatientStore();
  const { sidebarCollapsed, mobileOpen, toggleSidebar, setMobileOpen } = useUIStore();
  const routeKey = path.split("/").filter(Boolean)[0] || "overview";
  const key = ({ dashboard: "overview", "ml-toxicity": "ml", "dl-progression": "dl", "nlp-reports": "nlp", "slm-copilot": "slm" } as Record<string,string>)[routeKey] || routeKey;
  const [showAlerts, setShowAlerts] = useState(false);
  const [unreadAlerts, setUnreadAlerts] = useState(clinicalAlerts);
  const [searchTerm, setSearchTerm] = useState("");
  const [paletteOpen, setPaletteOpen] = useState(false);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen((open) => !open);
      }
      if (event.key === "Escape") setPaletteOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  function handleSearchSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (searchTerm.trim()) {
      router.push(`/patients?q=${encodeURIComponent(searchTerm.trim())}`);
    }
  }

  function dismissAlert(id: string) {
    setUnreadAlerts((prev) => prev.filter((a) => a.id !== id));
  }

  const commands = [
    { label: `Open ${activePatient.id}`, detail: "Active patient record", icon: Users, href: `/patients/${activePatient.id}` },
    { label: "Run integrated analysis", detail: "Stages 1–4", icon: Activity, href: `/integrated-analysis/${activePatient.id}` },
    { label: "Upload clinical report", detail: "PDF or TXT", icon: Upload, href: "/nlp" },
    { label: "Open toxicity model", detail: "Stage 1", icon: BarChart3, href: "/ml" },
    { label: "Ask Clinical Copilot", detail: "Grounded patient context", icon: BrainCircuit, href: "/slm" },
    { label: "Generate safety case", detail: "Stage 5", icon: FlaskConical, href: "/safety-lab" },
    { label: "Open Tumor Board", detail: "Stage 6", icon: Sparkles, href: `/tumor-board/${activePatient.id}` },
  ].filter((command) => !searchTerm || `${command.label} ${command.detail}`.toLowerCase().includes(searchTerm.toLowerCase()));

  return (
    <DocAIProvider key={`${key}-${activePatient.id}`} initialMessage={`${messages[key] || messages.overview} Active patient: ${activePatient.id}.`}>
      <div className={`app-shell ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
        <aside className={`sidebar ${mobileOpen ? "mobile-open" : ""}`}>
          <div className="sidebar-brand">
            <span className="brand-mark-hex">
              <svg viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                <polygon points="14,2 25,8 25,20 14,26 3,20 3,8" stroke="#f59e0b" strokeWidth="1.6" fill="rgba(245, 158, 11, 0.16)" />
                <path d="M14 2V14M14 14L25 20M14 14L3 20" stroke="#fbbf24" strokeWidth="1.4" />
                <circle cx="14" cy="14" r="2.2" fill="#fef08a" />
                <circle cx="14" cy="2" r="1.5" fill="#fbbf24" />
                <circle cx="25" cy="8" r="1.5" fill="#fbbf24" />
                <circle cx="25" cy="20" r="1.5" fill="#fbbf24" />
                <circle cx="14" cy="26" r="1.5" fill="#fbbf24" />
                <circle cx="3" cy="20" r="1.5" fill="#fbbf24" />
                <circle cx="3" cy="8" r="1.5" fill="#fbbf24" />
              </svg>
            </span>
            <div><b>ONCO.AI</b><span>Precision Oncology</span></div>
            <button className="mobile-close" onClick={() => setMobileOpen(false)} aria-label="Close navigation"><X /></button>
          </div>
          <nav>
            {groups.map((group, gIdx) => (
              <div className="nav-group" key={group.label || `primary-${gIdx}`}>
                {group.label && <p>{group.label}</p>}
                {group.items.map((item) => {
                  const href = item.href.replace("ACTIVE", activePatient.id);
                  const active = path === href || (href !== "/overview" && path.startsWith(href.split("/").slice(0,2).join("/")));
                  const isOverview = href === "/overview";
                  const Icon = item.icon;
                  return (
                    <Link
                      key={href}
                      href={href}
                      className={`${active ? "active" : ""} ${isOverview && active ? "active-overview" : ""}`}
                      onClick={() => setMobileOpen(false)}
                    >
                      <Icon />
                      <span>{item.label}</span>
                      {active && <i />}
                    </Link>
                  );
                })}
              </div>
            ))}
            
            {/* Sidebar Bottom Area */}
            <div className="sidebar-bottom-section">
              <div className="sidebar-landscape-art" />
              <div className="nav-group bottom-actions">
                <Link href="/settings" className={path === "/settings" ? "active" : ""}>
                  <Settings /><span>Settings</span>
                </Link>
                <button className="nav-signout" onClick={toggleSidebar} aria-label="Collapse sidebar">
                  <ChevronLeft /><span>Collapse</span>
                </button>
              </div>
            </div>
          </nav>
        </aside>

        {mobileOpen && <button className="sidebar-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}

        <div className="app-main">
          <header className="topbar">
            <button className="menu-button" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu /></button>
            
            <form className="search-box" onSubmit={handleSearchSubmit}>
              <Search />
              <input
                aria-label="Search patients, reports and trials"
                placeholder="Search patients, trials, reports, or ask ONCO.AI..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onFocus={() => setPaletteOpen(true)}
              />
              <kbd>Ctrl + K</kbd>
            </form>

            <div className="demo-pill"><span className="demo-dot" /> Demo Data</div>

            <div className="topbar-clock" aria-label="Current date and time">
              <span>Thu, Jun 12, 2026</span>
              <b>09:37 AM</b>
            </div>

            {/* Interactive Alerts Button */}
            <div style={{ position: "relative" }}>
              <button
                className="alerts"
                aria-label="Notifications"
                onClick={() => setShowAlerts(!showAlerts)}
                style={{ position: "relative" }}
              >
                <Bell size={18} />
                {unreadAlerts.length > 0 && <i className="alert-badge" />}
              </button>

              {/* Alerts Popover */}
              {showAlerts && (
                <div className="alerts-popover">
                  <div className="alerts-header">
                    <b>Clinical Intelligence Alerts</b>
                    {unreadAlerts.length > 0 ? (
                      <button
                        className="text-button"
                        style={{ fontSize: "11px" }}
                        onClick={() => setUnreadAlerts([])}
                      >
                        Clear all
                      </button>
                    ) : (
                      <span style={{ fontSize: "11px", color: "#63738a" }}>All clear</span>
                    )}
                  </div>

                  <div className="alerts-list">
                    {unreadAlerts.length === 0 ? (
                      <p style={{ fontSize: "12px", color: "#63738a", textAlign: "center", padding: "12px 0" }}>
                        No active clinical alerts. All safety checks passing.
                      </p>
                    ) : (
                      unreadAlerts.map((alert) => (
                        <div className={`alert-item ${alert.type}`} key={alert.id}>
                          {alert.type === "critical" ? (
                            <ShieldAlert size={16} color="#dc3f50" style={{ flexShrink: 0, marginTop: "2px" }} />
                          ) : alert.type === "warning" ? (
                            <AlertTriangle size={16} color="#b76b00" style={{ flexShrink: 0, marginTop: "2px" }} />
                          ) : (
                            <Info size={16} color="#1769e0" style={{ flexShrink: 0, marginTop: "2px" }} />
                          )}
                          <div style={{ flex: 1 }}>
                            <b>{alert.title}</b>
                            <p>{alert.desc}</p>
                            <small style={{ color: "#8ca0b3", fontSize: "9px" }}>{alert.time}</small>
                          </div>
                          <button
                            onClick={() => dismissAlert(alert.id)}
                            style={{ border: "none", background: "none", cursor: "pointer", color: "#9cb0c4", padding: "2px" }}
                            aria-label="Dismiss alert"
                          >
                            <X size={13} />
                          </button>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>

            <div className="clinician">
              <div className="clinician-avatar">PS</div>
              <div><b>Dr. Priya Sharma</b><span>Oncologist</span></div>
            </div>
          </header>

          <main className="content">{children}</main>
        </div>
        {paletteOpen && <div className="command-backdrop" onMouseDown={() => setPaletteOpen(false)}>
          <section className="command-palette" onMouseDown={(event) => event.stopPropagation()} aria-label="ONCO.AI command palette">
            <header><Command /><input autoFocus value={searchTerm} onChange={(event) => setSearchTerm(event.target.value)} placeholder="Search or choose a command…" /><kbd>ESC</kbd></header>
            <div>{commands.length ? commands.map((command) => { const Icon = command.icon; return <button key={command.label} onClick={() => { router.push(command.href); setPaletteOpen(false); setSearchTerm(""); }}><span><Icon /></span><div><b>{command.label}</b><small>{command.detail}</small></div><ChevronLeft /></button>; }) : <p className="command-empty">No matching command. Search patient IDs from the Patients workspace.</p>}</div>
            <footer><span><CalendarDays /> Current context</span><b>{activePatient.id} · {activePatient.cancer}</b></footer>
          </section>
        </div>}
        <CopilotDock />
        <DocAI />
      </div>
    </DocAIProvider>
  );
}
