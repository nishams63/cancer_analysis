"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { signOut } from "next-auth/react";
import { Activity, AlertTriangle, BarChart3, Bell, BrainCircuit, Check, ChevronLeft, ClipboardList, Dna, FileSearch, FlaskConical, Info, LayoutDashboard, LogOut, Menu, Microscope, Search, Settings, ShieldAlert, ShieldCheck, Sparkles, Users, X } from "lucide-react";
import { useUIStore } from "@/stores/ui-store";
import { DocAIProvider } from "@/components/doc-ai/DocAIProvider";
import { DocAI } from "@/components/doc-ai/DocAI";

const groups = [
  { label: "", items: [{ href: "/overview", label: "Overview", icon: LayoutDashboard }] },
  { label: "PATIENT", items: [{ href: "/patients", label: "Patients", icon: Users }, { href: "/integrated-analysis/ONC-2048", label: "Integrated Analysis", icon: Activity }] },
  { label: "AI MODULES", items: [{ href: "/ml", label: "ML · Toxicity", icon: BarChart3 }, { href: "/dl", label: "DL · Progression", icon: Microscope }, { href: "/nlp", label: "NLP · Reports", icon: FileSearch }, { href: "/slm", label: "SLM · Copilot", icon: BrainCircuit }] },
  { label: "AI SAFETY", items: [{ href: "/safety-lab", label: "Stage 5 · Safety Lab", icon: FlaskConical }] },
  { label: "AUTONOMOUS INTELLIGENCE", items: [{ href: "/tumor-board/ONC-2048", label: "Stage 6 · Tumor Board", icon: Sparkles }] },
  { label: "ENGINEERING", items: [{ href: "/analytics", label: "Analytics", icon: BarChart3 }, { href: "/audit", label: "Audit Trail", icon: ClipboardList }] },
];

const messages: Record<string, string> = {
  overview: "All AI modules are ready. Review each model or run a complete patient analysis.",
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
  const { sidebarCollapsed, mobileOpen, toggleSidebar, setMobileOpen } = useUIStore();
  const key = path.split("/").filter(Boolean)[0] || "overview";
  const [showAlerts, setShowAlerts] = useState(false);
  const [unreadAlerts, setUnreadAlerts] = useState(clinicalAlerts);
  const [searchTerm, setSearchTerm] = useState("");

  function handleSearchSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (searchTerm.trim()) {
      router.push(`/patients?q=${encodeURIComponent(searchTerm.trim())}`);
    }
  }

  function dismissAlert(id: string) {
    setUnreadAlerts((prev) => prev.filter((a) => a.id !== id));
  }

  return (
    <DocAIProvider initialMessage={messages[key] || messages.overview}>
      <div className={`app-shell ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
        <aside className={`sidebar ${mobileOpen ? "mobile-open" : ""}`}>
          <div className="sidebar-brand">
            <span className="brand-mark"><Dna /></span>
            <div><b>ONCO.AI</b><span>Precision Oncology</span></div>
            <button className="mobile-close" onClick={() => setMobileOpen(false)} aria-label="Close navigation"><X /></button>
          </div>
          <nav>
            {groups.map((group) => (
              <div className="nav-group" key={group.label || "primary"}>
                {group.label && <p>{group.label}</p>}
                {group.items.map((item) => {
                  const active = path === item.href || (item.href !== "/overview" && path.startsWith(item.href));
                  const Icon = item.icon;
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      className={active ? "active" : ""}
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
            <div className="nav-group">
              <Link href="/settings" className={path === "/settings" ? "active" : ""}>
                <Settings /><span>Settings</span>
              </Link>
              <button className="nav-signout" onClick={() => signOut({ callbackUrl: "/login" })}>
                <LogOut /><span>Sign out</span>
              </button>
            </div>
          </nav>
          <button className="collapse" onClick={toggleSidebar} aria-label="Toggle sidebar"><ChevronLeft /></button>
        </aside>

        {mobileOpen && <button className="sidebar-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}

        <div className="app-main">
          <header className="topbar">
            <button className="menu-button" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu /></button>
            
            <form className="search-box" onSubmit={handleSearchSubmit}>
              <Search />
              <input
                aria-label="Search patients, reports and trials"
                placeholder="Search patients, reports, trials… (Press Enter)"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
              <kbd>↵ Enter</kbd>
            </form>

            <div className="demo-pill"><span /> DEMO DATA</div>

            {/* Interactive Alerts Button */}
            <div style={{ position: "relative" }}>
              <button
                className="alerts"
                aria-label="Notifications"
                onClick={() => setShowAlerts(!showAlerts)}
                style={{ position: "relative" }}
              >
                <Bell size={18} />
                {unreadAlerts.length > 0 && <i />}
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
        <DocAI />
      </div>
    </DocAIProvider>
  );
}
