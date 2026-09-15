"use client";

import { useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Expand, Focus, RotateCcw, ScanLine } from "lucide-react";

type Organ = "Lungs" | "Brain" | "Liver" | "Bone" | "Nodes";
type ViewMode = "3D View" | "CT Scan" | "Pathology" | "Gene Map";

const organPoints: Record<Organ, { x: number; y: number; label: string; detail: string }> = {
  Lungs: { x: 50, y: 38, label: "Primary tumor", detail: "Detected · right upper lobe" },
  Brain: { x: 50, y: 13, label: "Brain", detail: "No active demo signal" },
  Liver: { x: 43, y: 59, label: "Liver", detail: "Monitoring" },
  Bone: { x: 50, y: 74, label: "Bone", detail: "Low metastatic signal" },
  Nodes: { x: 61, y: 43, label: "Lymph node", detail: "Involvement detected" },
};

export function PatientDigitalTwin({ patientId }: { patientId: string }) {
  const [organ, setOrgan] = useState<Organ>("Lungs");
  const [mode, setMode] = useState<ViewMode>("3D View");
  const [rotation, setRotation] = useState({ x: -4, y: -8 });
  const [zoom, setZoom] = useState(1);
  const drag = useRef<{ x: number; y: number } | null>(null);
  const point = organPoints[organ];

  function reset() {
    setRotation({ x: -4, y: -8 });
    setZoom(1);
    setOrgan("Lungs");
  }

  function fullScreen() {
    document.querySelector(".digital-twin-card")?.requestFullscreen?.();
  }

  return (
    <section className={`card digital-twin-card twin-mode-${mode.toLowerCase().replaceAll(" ", "-")}`}>
      <div className="twin-toolbar">
        <div>
          <p className="eyebrow">PATIENT DIGITAL TWIN</p>
          <h2>Interactive disease map</h2>
        </div>
        <div className="twin-actions">
          <span className="live-signal"><i /> LIVE CONTEXT · {patientId}</span>
          <button onClick={reset} aria-label="Reset digital twin view" title="Reset camera"><RotateCcw /></button>
          <button onClick={fullScreen} aria-label="Open digital twin fullscreen" title="Fullscreen"><Expand /></button>
        </div>
      </div>

      <div className="twin-modes" role="tablist" aria-label="Visualization modes">
        {(["3D View", "CT Scan", "Pathology", "Gene Map"] as ViewMode[]).map((item) => (
          <button key={item} className={mode === item ? "active" : ""} onClick={() => setMode(item)} role="tab" aria-selected={mode === item}>{item}</button>
        ))}
      </div>

      <div className="twin-stage">
        <div className="twin-grid" aria-hidden="true" />
        <div className="twin-ambient ambient-one" aria-hidden="true" />
        <div className="twin-ambient ambient-two" aria-hidden="true" />

        <div
          className="human-scene"
          style={{ transform: `perspective(900px) rotateX(${rotation.x}deg) rotateY(${rotation.y}deg) scale(${zoom})` }}
          onPointerDown={(event) => { drag.current = { x: event.clientX, y: event.clientY }; event.currentTarget.setPointerCapture(event.pointerId); }}
          onPointerMove={(event) => {
            if (!drag.current) return;
            const dx = event.clientX - drag.current.x;
            const dy = event.clientY - drag.current.y;
            drag.current = { x: event.clientX, y: event.clientY };
            setRotation((value) => ({ x: Math.max(-22, Math.min(22, value.x - dy * .22)), y: value.y + dx * .28 }));
          }}
          onPointerUp={() => { drag.current = null; }}
          onPointerCancel={() => { drag.current = null; }}
          onWheel={(event) => { event.preventDefault(); setZoom((value) => Math.max(.82, Math.min(1.35, value - event.deltaY * .001))); }}
          aria-label="Interactive educational patient torso. Drag to rotate and scroll to zoom."
          role="img"
        >
          <svg viewBox="0 0 320 520" aria-hidden="true">
            <defs>
              <linearGradient id="bodyGlass" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stopColor="#8de4ee" stopOpacity=".34" />
                <stop offset=".48" stopColor="#3a7b87" stopOpacity=".13" />
                <stop offset="1" stopColor="#d9a85f" stopOpacity=".18" />
              </linearGradient>
              <radialGradient id="organGlow"><stop offset="0" stopColor="#f0c98a" stopOpacity=".9"/><stop offset="1" stopColor="#d9a85f" stopOpacity=".06"/></radialGradient>
              <filter id="softGlow"><feGaussianBlur stdDeviation="7" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
            </defs>
            <ellipse cx="160" cy="55" rx="49" ry="54" fill="url(#bodyGlass)" stroke="#8edce8" strokeOpacity=".46" />
            <path d="M105 114 C70 133 60 188 66 247 L88 392 L115 480 L148 480 L154 292 L166 292 L172 480 L205 480 L232 392 L254 247 C260 188 250 133 215 114 C193 102 127 102 105 114Z" fill="url(#bodyGlass)" stroke="#70c7d2" strokeOpacity=".48" />
            <path d="M95 131 C65 156 45 219 40 308" fill="none" stroke="#70c7d2" strokeOpacity=".32" strokeWidth="22" strokeLinecap="round" />
            <path d="M225 131 C255 156 275 219 280 308" fill="none" stroke="#70c7d2" strokeOpacity=".32" strokeWidth="22" strokeLinecap="round" />
            <path d="M160 109 L160 454" stroke="#59d7e8" strokeOpacity=".2" strokeDasharray="5 7" />
            <g className={organ === "Lungs" ? "organ active" : "organ"}>
              <path d="M151 151 C118 144 96 166 97 215 C98 244 116 253 145 237 L153 201Z" fill="#59d7e8" fillOpacity=".33" stroke="#72eef5" />
              <path d="M169 151 C202 144 224 166 223 215 C222 244 204 253 175 237 L167 201Z" fill="#59d7e8" fillOpacity=".33" stroke="#72eef5" />
            </g>
            <g className={organ === "Brain" ? "organ active" : "organ"}><path d="M126 54 C129 24 190 22 194 54 C197 82 176 91 160 89 C142 92 123 78 126 54Z" fill="#8b7cf6" fillOpacity=".35" stroke="#a69afb" /></g>
            <g className={organ === "Liver" ? "organ active" : "organ"}><path d="M112 260 C134 243 196 245 211 262 C204 291 170 301 121 285Z" fill="#d9a85f" fillOpacity=".32" stroke="#f0c98a" /></g>
            <g className={organ === "Bone" ? "organ active" : "organ"}><path d="M157 292 L157 452 M163 292 L163 452" stroke="#f0c98a" strokeOpacity=".68" strokeWidth="5" /></g>
            <g className={organ === "Nodes" ? "organ active" : "organ"}><circle cx="196" cy="218" r="8" fill="#8b7cf6"/><circle cx="203" cy="235" r="5" fill="#8b7cf6"/></g>
            <g className="tumor-marker" filter="url(#softGlow)"><circle cx="196" cy="183" r="22" fill="url(#organGlow)"/><circle cx="196" cy="183" r="7" fill="#f16d78"/><circle cx="196" cy="183" r="13" fill="none" stroke="#f0c98a" strokeDasharray="3 4" /></g>
          </svg>
        </div>

        <AnimatePresence mode="wait">
          <motion.div key={organ} className="medical-annotation" style={{ left: `${point.x}%`, top: `${point.y}%` }} initial={{ opacity: 0, scale: .9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}>
            <i /><span><b>{point.label}</b><small>{point.detail}</small></span>
          </motion.div>
        </AnimatePresence>

        <div className="twin-readouts">
          <span><ScanLine /> Treatment response <b>Monitoring</b></span>
          <span><Focus /> Metastatic signal <b>Low</b></span>
        </div>
        <small className="twin-hint">Drag to rotate · Scroll to zoom · Educational visualization</small>
      </div>

      <div className="organ-selector" aria-label="Choose organ focus">
        {(["Lungs", "Brain", "Liver", "Bone", "Nodes"] as Organ[]).map((item) => (
          <button key={item} className={organ === item ? "active" : ""} onClick={() => setOrgan(item)}><span />{item}</button>
        ))}
      </div>
    </section>
  );
}
